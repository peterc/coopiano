// Included by coopiano.cpp after its live adapter. Diagnostic CLI only.
#ifndef SYNTHPIANO_LIVE_BENCH_HPP
#define SYNTHPIANO_LIVE_BENCH_HPP
namespace live {
template<class T> T device_property(AudioDeviceID device, AudioObjectPropertySelector selector) {
    AudioObjectPropertyAddress address{selector,kAudioObjectPropertyScopeGlobal,kAudioObjectPropertyElementMain};
    T value{}; UInt32 size=sizeof value;
    audio_check(AudioObjectGetPropertyData(device,&address,0,nullptr,&size,&value),"read device property");
    return value;
}
class OverloadListener {
    AudioDeviceID device_;
    CallbackMonitor& monitor_;
    AudioObjectPropertyAddress address_{kAudioDeviceProcessorOverload,kAudioObjectPropertyScopeGlobal,kAudioObjectPropertyElementMain};
    static OSStatus notify(AudioObjectID, UInt32 count, const AudioObjectPropertyAddress* addresses, void* data) {
        auto& monitor=*static_cast<CallbackMonitor*>(data);
        for (UInt32 i=0;i<count;++i)
            if (addresses[i].mSelector==kAudioDeviceProcessorOverload)
                monitor.overloads.fetch_add(1,std::memory_order_relaxed);
        return noErr;
    }
public:
    OverloadListener(AudioDeviceID device, CallbackMonitor& monitor) : device_(device),monitor_(monitor) {
        audio_check(AudioObjectAddPropertyListener(device_,&address_,notify,&monitor_),"listen for device overloads");
    }
    ~OverloadListener() { AudioObjectRemovePropertyListener(device_,&address_,notify,&monitor_); }
    OverloadListener(const OverloadListener&) = delete;
    OverloadListener& operator=(const OverloadListener&) = delete;
};
static int benchmark_main(const std::string& timeline_path, const std::string& prefix, bool mute, unsigned host_rate=0) {
    unsigned rate=host_rate ? host_rate : piano::FS;
    auto replay=Replay::load(timeline_path,rate);
    // Fail before opening a device if reports cannot be written.
    std::ofstream summary(prefix+"-summary.csv"), details(prefix+"-callbacks.csv"), info(prefix+"-device.txt");
    if (!summary || !details || !info) throw std::runtime_error("cannot create live benchmark reports");
    Engine engine;
    if (!engine.core.has_strike_cache()) {
        std::printf("Building exact strike cache for callback replay...\n"); std::fflush(stdout);
        engine.core.build_strike_cache(engine.velocity_gamma);
    }
    std::unique_ptr<piano_host::Adapter> adapter;
    if (host_rate) {
        adapter=std::make_unique<piano_host::Adapter>(engine.core);
        adapter->prepare(rate,8192,replay.events.size());
        engine.host_adapter=adapter.get(); engine.host_events.reserve(replay.events.size());
    }
    const uint64_t total=replay.frames+uint64_t(10)*rate+(adapter ? adapter->latency_samples() : BLK);
    CallbackMonitor monitor(total,rate); monitor.mute=mute;
    engine.replay=&replay; engine.monitor=&monitor;
    g_quit.store(false); g_engine=&engine;
    // This guard outlives the output so the callback target remains valid until
    // AudioUnit disposal, including all exception and cancellation paths.
    struct ClearEngine { ~ClearEngine() { g_engine=nullptr; } } clear;
    AudioOutput output(audio_cb,nullptr,rate);
    auto device=output.device();
    CFStringRef name_ref=device_property<CFStringRef>(device,kAudioObjectPropertyName);
    std::string name=cf_str(name_ref); if (name_ref) CFRelease(name_ref);
    double hardware_rate=device_property<Float64>(device,kAudioDevicePropertyNominalSampleRate);
    UInt32 hardware_frames=device_property<UInt32>(device,kAudioDevicePropertyBufferFrameSize);
    info << "device=" << name << "\ndevice_id=" << device << "\nhardware_rate=" << hardware_rate
         << "\nhardware_buffer_frames=" << hardware_frames << "\nclient_rate=" << rate
         << "\nhost_adapter=" << bool(adapter)
         << "\nreported_latency_frames=" << (adapter ? adapter->latency_samples() : 0)
         << "\nengine_block_frames=" << BLK << "\nmuted_after_render=" << mute
         << "\nvelocity_gamma=" << engine.velocity_gamma << "\noutput_gain=" << instrument_settings().gain
         << "\nresponse_storage_bytes=" << engine.core.response_storage_bytes()
         << "\noverload_listener=registered\n";
    info.flush();
    std::printf("Live replay: %s, %.1f seconds + 10-second tail; %s.\nDevice: %s, %.0f Hz, %u hardware frames; client %u Hz.\n",
        replay.name.c_str(),replay.frames/double(rate),mute?"muted after synthesis":"audible",
        name.c_str(),hardware_rate,hardware_frames,rate); std::fflush(stdout);
    bool interrupted=false;
    {
        OverloadListener listener(device,monitor);
        auto previous_signal=std::signal(SIGINT,on_sigint);
        struct RestoreSignal {
            decltype(previous_signal) handler;
            ~RestoreSignal() { std::signal(SIGINT,handler); }
        } restore{previous_signal};
        // Ensure stop precedes listener teardown even on exceptions.
        struct StopOutput { AudioOutput& output; ~StopOutput() { output.stop(); } } stop{output};
        output.start();
        auto last_progress=std::chrono::steady_clock::now();
        uint64_t previous=0,last_logged=0;
        while (monitor.progress.load(std::memory_order_relaxed)<total && !g_quit.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            uint64_t position=monitor.progress.load(std::memory_order_relaxed);
            auto now=std::chrono::steady_clock::now();
            if (position!=previous) { last_progress=now; previous=position; }
            if (now-last_progress>std::chrono::seconds(5)) throw std::runtime_error("audio callback stopped progressing");
            if (output.device()!=device
                || device_property<Float64>(device,kAudioDevicePropertyNominalSampleRate)!=hardware_rate
                || device_property<UInt32>(device,kAudioDevicePropertyBufferFrameSize)!=hardware_frames)
                throw std::runtime_error("audio device configuration changed during benchmark");
            if (position/(30*rate)>last_logged) {
                last_logged=position/(30*rate);
                std::printf("  %.1f s replayed; device overload notifications %llu\n",position/double(rate),
                    (unsigned long long)monitor.overloads.load()); std::fflush(stdout);
            }
        }
        interrupted=g_quit.load();
        output.stop();
    }
    // Callback-owned state is read only after audio has stopped. Formatting,
    // sorting and file I/O never happen on the audio thread.
    std::vector<double> times;
    times.reserve(monitor.stored);
    UInt32 min_frames=UINT32_MAX,max_frames=0;
    details << "position_seconds,frames,wall_us,budget_us,voices,timestamp_valid,gap_frames\n" << std::fixed << std::setprecision(6);
    for (size_t i=0;i<monitor.stored;++i) {
        const auto& r=monitor.records[i]; times.push_back(r.wall_us);
        min_frames=std::min(min_frames,r.frames); max_frames=std::max(max_frames,r.frames);
        details << r.position/double(rate) << ',' << r.frames << ',' << r.wall_us << ',' << r.budget_us
                << ',' << r.voices << ',' << r.timestamp_valid << ',' << r.gap_frames << '\n';
    }
    std::sort(times.begin(),times.end());
    uint64_t held=0;
    for (int ch=0;ch<16;++ch) for (int key=0;key<128;++key) held+=engine.core.held_count(ch,key);
    bool complete=!interrupted && monitor.frames>=total && replay.next==replay.events.size() && held==0 && engine.processing_failures==0;
    summary << "piece,complete,notes,callbacks,rendered_seconds,min_frames,max_frames,mean_us,p99_us,max_us,callback_overruns,device_overloads,timestamp_gaps,invalid_timestamps,missing_records,peak_voices,dropped_notes,held_keys,tail_voices,output_peak,processing_failures,stolen_voices,stolen_held_voices\n"
            << std::fixed << std::setprecision(6) << replay.name << ',' << complete << ',' << replay.notes
            << ',' << monitor.callbacks << ',' << monitor.frames/double(rate) << ',' << min_frames << ',' << max_frames
            << ',' << (monitor.callbacks?monitor.sum_wall_us/monitor.callbacks:0)
            << ',' << (times.empty()?0:times[times.size()*99/100]) << ',' << monitor.max_wall_us
            << ',' << monitor.overruns << ',' << monitor.overloads.load() << ',' << monitor.gaps
            << ',' << monitor.invalid_stamps << ',' << (monitor.callbacks-monitor.stored)
            << ',' << engine.core.peak_voices() << ',' << engine.core.dropped_notes()+engine.command_drops.load()
            << ',' << held << ',' << engine.core.active_voices() << ',' << engine.core.output_peak() << ',' << engine.processing_failures
            << ',' << engine.core.stolen_voices() << ',' << engine.core.stolen_held_voices() << '\n';
    summary.close(); details.close(); info.close();
    if (!summary || !details || !info) throw std::runtime_error("could not finish live benchmark reports");
    std::printf("Finished %s: %llu callbacks, %llu callback overruns, %llu device overloads, %llu timestamp gaps; peak %zu voices; %llu notes dropped; %llu voices stolen.\n",
        replay.name.c_str(),(unsigned long long)monitor.callbacks,(unsigned long long)monitor.overruns,
        (unsigned long long)monitor.overloads.load(),(unsigned long long)monitor.gaps,
        engine.core.peak_voices(),(unsigned long long)engine.core.dropped_notes(),(unsigned long long)engine.core.stolen_voices());
    return complete && monitor.stored==monitor.callbacks ? 0 : 2;
}
} // namespace live
#endif
