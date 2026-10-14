// macOS app-only replay and callback measurements. No synthesis changes.
#ifndef SYNTHPIANO_LIVE_DIAGNOSTICS_HPP
#define SYNTHPIANO_LIVE_DIAGNOSTICS_HPP
#include <AudioToolbox/AudioToolbox.h>
#include <CoreAudio/CoreAudio.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace live {
struct ReplayEvent {
    uint64_t frame = 0, seed = 1;
    int type = 0, key = 0, velocity = 0, channel = 0, down = 0;
};
struct Replay {
    std::string name;
    uint64_t frames = 0, notes = 0, generated = 0;
    size_t next = 0;
    std::vector<ReplayEvent> events;
    static Replay load(const std::string& path, unsigned rate=44100) {
        std::ifstream in(path);
        Replay result;
        std::string line, extra;
        if (!std::getline(in,line)) throw std::runtime_error("cannot read timeline: "+path);
        std::istringstream header(line);
        if (!(header >> result.name >> result.frames >> result.notes) || (header >> extra)
            || result.frames > uint64_t(rate)*86400)
            throw std::runtime_error("invalid timeline header");
        uint64_t notes = 0;
        while (std::getline(in,line)) {
            std::istringstream row(line); ReplayEvent e;
            if (!(row >> e.frame >> e.type >> e.key >> e.velocity >> e.channel >> e.seed >> e.down)
                || (row >> extra) || e.frame > result.frames
                || (!result.events.empty() && e.frame < result.events.back().frame)
                || e.type < 0 || e.type > 2 || e.key < 0 || e.key > 127
                || e.velocity < 0 || e.velocity > 127 || e.channel < 0 || e.channel > 15
                || e.down < 0 || e.down > 1 || (e.type == 0 && e.velocity == 0))
                throw std::runtime_error("invalid timeline event");
            notes += e.type == 0;
            result.events.push_back(e);
        }
        if (!in.eof() || notes != result.notes) throw std::runtime_error("invalid timeline note count");
        return result;
    }
    // Match the app's live semantics: commands apply at the first engine block
    // boundary at/after their timestamp. Never pull future events forward.
    template<class Apply> void dispatch(Apply apply, size_t block) {
        while (next < events.size() && events[next].frame <= generated) apply(events[next++]);
        generated += block;
    }
};
struct CallbackRecord {
    uint64_t position = 0;
    uint32_t frames = 0;
    double wall_us = 0, budget_us = 0, gap_frames = 0;
    size_t voices = 0;
    bool timestamp_valid = false;
};
struct CallbackMonitor {
    // Allocate and touch bounded storage before starting the device. At 16+
    // client frames/callback this fits the entire run; smaller buffers are
    // counted but explicitly report missing detailed rows instead of allocating.
    std::vector<CallbackRecord> records;
    size_t stored = 0;
    uint64_t callbacks = 0, frames = 0, overruns = 0, gaps = 0, invalid_stamps = 0;
    double expected_sample = 0, max_wall_us = 0, sum_wall_us = 0;
    bool have_sample = false, mute = true;
    std::atomic<uint64_t> progress{0}, overloads{0};
    unsigned sample_rate;
    explicit CallbackMonitor(uint64_t total_frames, unsigned rate=44100) : records(total_frames/16 + 8192),sample_rate(rate) {
        if (!rate) throw std::invalid_argument("callback rate must be positive");
    }
    void record(const AudioTimeStamp* stamp, uint32_t count, double elapsed, size_t voices) noexcept {
        CallbackRecord row;
        row.position = frames; row.frames = count; row.wall_us = elapsed;
        row.budget_us = 1e6*count/sample_rate; row.voices = voices;
        row.timestamp_valid = stamp && (stamp->mFlags & kAudioTimeStampSampleTimeValid)
                              && std::isfinite(stamp->mSampleTime);
        if (row.timestamp_valid) {
            if (have_sample) {
                row.gap_frames = stamp->mSampleTime - expected_sample;
                if (std::abs(row.gap_frames) > .5) ++gaps;
            }
            expected_sample = stamp->mSampleTime + count;
        } else ++invalid_stamps;
        have_sample = row.timestamp_valid;
        ++callbacks; frames += count;
        overruns += elapsed > row.budget_us;
        max_wall_us = std::max(max_wall_us, elapsed); sum_wall_us += elapsed;
        if (stored < records.size()) records[stored++] = row;
        progress.store(frames,std::memory_order_relaxed);
    }
};
static_assert(std::atomic<uint64_t>::is_always_lock_free, "callback counters must be lock-free");

inline void audio_check(OSStatus status, const char* operation) {
    if (status != noErr) throw std::runtime_error(std::string(operation)+": OSStatus "+std::to_string(status));
}
// Shared setup for normal live playback and the diagnostic replay. Does not
// change the device's global sample rate or buffer size.
class AudioOutput {
    AudioUnit unit_ = nullptr;
    bool started_ = false;
public:
    AudioOutput(AURenderCallback callback, void* context = nullptr, unsigned sample_rate=44100) {
        try {
            AudioComponentDescription d{};
            d.componentType=kAudioUnitType_Output; d.componentSubType=kAudioUnitSubType_DefaultOutput;
            d.componentManufacturer=kAudioUnitManufacturer_Apple;
            auto component=AudioComponentFindNext(nullptr,&d);
            if (!component) throw std::runtime_error("default output AudioUnit unavailable");
            audio_check(AudioComponentInstanceNew(component,&unit_),"create output");
            AudioStreamBasicDescription fmt{};
            fmt.mSampleRate=sample_rate; fmt.mFormatID=kAudioFormatLinearPCM;
            fmt.mFormatFlags=kAudioFormatFlagsNativeFloatPacked|kAudioFormatFlagIsNonInterleaved;
            fmt.mChannelsPerFrame=2; fmt.mBitsPerChannel=32; fmt.mBytesPerFrame=4;
            fmt.mBytesPerPacket=4; fmt.mFramesPerPacket=1;
            audio_check(AudioUnitSetProperty(unit_,kAudioUnitProperty_StreamFormat,
                        kAudioUnitScope_Input,0,&fmt,sizeof fmt),"set output format");
            AURenderCallbackStruct cb{callback,context};
            audio_check(AudioUnitSetProperty(unit_,kAudioUnitProperty_SetRenderCallback,
                        kAudioUnitScope_Input,0,&cb,sizeof cb),"set output callback");
            audio_check(AudioUnitInitialize(unit_),"initialize output");
        } catch (...) { if (unit_) AudioComponentInstanceDispose(unit_); throw; }
    }
    AudioOutput(const AudioOutput&) = delete;
    AudioOutput& operator=(const AudioOutput&) = delete;
    ~AudioOutput() { stop(); if (unit_) AudioComponentInstanceDispose(unit_); }
    void start() { audio_check(AudioOutputUnitStart(unit_),"start output"); started_=true; }
    void stop() noexcept { if (started_) AudioOutputUnitStop(unit_); started_=false; }
    AudioDeviceID device() const {
        AudioDeviceID result=0; UInt32 size=sizeof result;
        audio_check(AudioUnitGetProperty(unit_,kAudioOutputUnitProperty_CurrentDevice,
                    kAudioUnitScope_Global,0,&result,&size),"get output device");
        return result;
    }
};
} // namespace live
#endif
