// coopiano.cpp — command-line/MIDI host for the piano.hpp synthesis library.
// Build: make
// Usage: ./coopiano [note] [velocity 0..1] [duration] [hold]
//        ./coopiano demo
//        ./coopiano file.mid [velocity_scale] [output.wav]
//        ./coopiano live [--selftest]   (macOS)

#include "instrument.hpp"
#include "preset.hpp"

#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

// Set once by main before any workers or devices start. Voices copy resolved
// parameters during preparation; the audio callback never reads preset files.
static piano::Voicing host_voicing;
static room::Options host_reverb;

using instrument::OutputEffects;
using instrument::ResonanceState;

// Only the host translates process settings into per-render library options.
static const piano::Options& piano_options() {
    static const piano::Options options = [] {
        piano::Options o;
        if (const char* value = std::getenv("GP_DAMP_SCALE"))
            o.damper_scale = preset::number(value);
        if (const char* value = std::getenv("GP_KNOCK_GAIN"))
            o.knock_gain = preset::number(value);
        o.knock = !std::getenv("GP_NO_KNOCK");
        o.scrape = !std::getenv("GP_NO_SCRAPE");
        o.damper_noise = !std::getenv("GP_NO_DAMPNOISE");
        if (const char* value = std::getenv("GP_SOFT_LONGITUDINAL")) {
            std::string flag(value);
            if (flag != "0" && flag != "1")
                throw std::invalid_argument("GP_SOFT_LONGITUDINAL must be 0 or 1");
            o.soften_longitudinal_highs = flag == "1";
        }
        return o;
    }();
    return options;
}

static const piano::ResonanceOptions& resonance_options() {
    return host_voicing.resonance;
}

static void configure_voicing(const std::string& path) {
    auto candidate = path.empty() ? preset::Preset{} : preset::load(path);
    // Explicit environment controls keep their existing override semantics.
    if (const char* value = std::getenv("GP_RESONANCE_GAIN"))
        candidate.voicing.resonance.gain = preset::number(value);
    if (std::getenv("GP_NO_RESONANCE")) candidate.voicing.resonance.gain = 0.0;
    if (const char* value = std::getenv("GP_REVERB_MIX")) candidate.reverb.mix = preset::number(value);
    if (std::getenv("GP_NO_REVERB")) candidate.reverb.mix = 0.0;
    preset::validate(candidate);
    host_voicing = candidate.voicing;
    host_reverb = candidate.reverb;
}

static double note_to_midi(const std::string& note) {
    auto sem = [](char c) -> int {
        switch ((char)std::toupper((unsigned char)c)) {
            case 'C': return 0;  case 'D': return 2;  case 'E': return 4;
            case 'F': return 5;  case 'G': return 7;  case 'A': return 9;
            case 'B': return 11; default: throw std::invalid_argument("bad note");
        }
    };
    if (note.size() < 2) throw std::invalid_argument("bad note");
    int s = sem(note[0]);
    size_t pos = 1;
    while (pos < note.size() && (note[pos] == '#' || note[pos] == 'b')) {
        s += (note[pos] == '#') ? 1 : -1;
        ++pos;
    }
    size_t used = 0;
    int octave = std::stoi(note.substr(pos), &used);
    if (pos + used != note.size()) throw std::invalid_argument("bad note: " + note);
    return 12.0 * (static_cast<double>(octave) + 1.0) + s;
}

static double write_wav_stereo(const std::string& path,
                             const std::vector<double>& L,
                             const std::vector<double>& R) {
    double peak = 1e-12;
    for (size_t i = 0; i < L.size(); ++i)
        peak = std::max({peak, std::abs(L[i]), std::abs(R[i])});
    double scale = 0.89 / peak;

    std::vector<int16_t> pcm(L.size() * 2);
    for (size_t i = 0; i < L.size(); ++i) {
        pcm[2*i]   = (int16_t)(std::clamp(L[i] * scale, -1.0, 1.0) * 32767.0);
        pcm[2*i+1] = (int16_t)(std::clamp(R[i] * scale, -1.0, 1.0) * 32767.0);
    }
    uint32_t data_bytes = (uint32_t)(pcm.size() * 2);
    uint32_t chunk = 36 + data_bytes, fmt_size = 16, sr = piano::FS, br = piano::FS * 4;
    uint16_t fmt = 1, ch = 2, ba = 4, bits = 16;
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write WAV '" + path + "'");
    f.write("RIFF", 4); f.write((char*)&chunk, 4); f.write("WAVE", 4);
    f.write("fmt ", 4); f.write((char*)&fmt_size, 4);
    f.write((char*)&fmt, 2); f.write((char*)&ch, 2);
    f.write((char*)&sr, 4); f.write((char*)&br, 4);
    f.write((char*)&ba, 2); f.write((char*)&bits, 2);
    f.write("data", 4); f.write((char*)&data_bytes, 4);
    f.write((char*)pcm.data(), data_bytes);
    f.close();
    if (!f) throw std::runtime_error("could not finish WAV '" + path + "'");
    return scale;
}

// ---------------------------------------------------------------------------

static std::string midi_to_name(int m) {
    static const char* names[12] = {"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"};
    return std::string(names[((m % 12) + 12) % 12]) + std::to_string(m / 12 - 1);
}

// ---------------------------------------------------------------------------
// Performances: physical key events and pedal changes, rendered through the
// live instrument engine faster than real time. There is no separate offline
// renderer: voices, damping, sustain, resonance, soundboard and room are the
// same code the live player and the plugin run.

struct Note { double t; double midi; double vel; double hold; int channel = 0; }; // hold: physical key hold
struct PedalEvent { double t; int channel; double position; }; // 0 up .. 1 down (CC64/127)
struct Performance { std::vector<Note> notes; std::vector<PedalEvent> pedals; };

static void save_render_settings(const std::string& outfile, const Performance& performance,
                                 size_t frames, double pcm_gain) {
    std::ofstream file(outfile + ".ini");
    if (!file) throw std::runtime_error("cannot write settings '" + outfile + ".ini'");
    file << std::setprecision(17)
         << "# Resolved coopiano voicing; load this file with --preset.\n"
         << "# Render: sample_rate=" << piano::FS << " frames=" << frames
         << " pcm_gain=" << pcm_gain << '\n'
         << "# Soundboard seed=0xb0a4d5eed; note seeds use event index and pitch.\n";
    const auto& o = piano_options();
    file << "# Voice options: damper_scale=" << o.damper_scale << " knock_gain=" << o.knock_gain
         << " knock=" << o.knock << " scrape=" << o.scrape << " damper_noise=" << o.damper_noise
         << " soften_longitudinal_highs=" << o.soften_longitudinal_highs << '\n';
    for (const auto& n : performance.notes)
        file << "# Note: t=" << n.t << " midi=" << n.midi << " velocity=" << n.vel
             << " hold=" << n.hold << " channel=" << n.channel << '\n';
    for (const auto& p : performance.pedals)
        file << "# Pedal: t=" << p.t << " channel=" << p.channel << " position=" << p.position << '\n';
    preset::write(file, preset::Preset{host_voicing, host_reverb});
    file.close();
    if (!file) throw std::runtime_error("could not finish settings '" + outfile + ".ini'");
}

// Notes are prepared in parallel up front (the contact simulation), then the
// engine streams the key releases and pedal changes at their own times. A
// positive total_dur renders exactly that long plus the room tail; otherwise
// rendering continues until every voice has been culled and the output,
// sympathetic ring included, has fallen 100 dB under the render's peak for the
// length of the room tail.
static void render(const Performance& performance, double total_dur,
                   const std::string& outfile, bool verbose) {
    using Clock = std::chrono::steady_clock;
    const auto& notes = performance.notes;
    instrument::Settings settings;
    settings.voicing = host_voicing; settings.voice_options = piano_options(); settings.reverb = host_reverb;
    settings.maximum_voices = 1024;     // no deadline offline: never steal
    settings.gain *= 0.25;              // keep the output limiter linear; the WAV is normalized
    instrument::Engine engine(settings);

    auto t0 = Clock::now();
    std::vector<std::unique_ptr<instrument::PreparedStrike>> strikes(notes.size());
    std::atomic<size_t> next{0};
    unsigned nt = std::max(1u, std::min(std::thread::hardware_concurrency(), (unsigned)std::max<size_t>(notes.size(), 1)));
    std::vector<std::thread> pool;
    for (unsigned w = 0; w < nt; ++w) pool.emplace_back([&] {
        for (size_t i; (i = next.fetch_add(1)) < notes.size();) {
            const Note& n = notes[i];
            uint64_t seed = 0x9e3779b97f4a7c15ULL ^ (i * 0x100000001b3ULL)
                          ^ ((uint64_t)std::lround(n.midi * 64.0) << 32);
            strikes[i] = std::make_unique<instrument::PreparedStrike>(
                engine.prepare((int)std::lround(n.midi), n.vel, seed));
        }
    });
    for (auto& t : pool) t.join();
    double prepare_seconds = std::chrono::duration<double>(Clock::now() - t0).count();
    if (verbose && notes.size() <= 24)
        for (size_t i = 0; i < notes.size(); ++i) {
            const auto& d = strikes[i]->note().note_diagnostics();
            auto key = piano::make_key(notes[i].midi, host_voicing);
            std::printf("%-3s f0=%7.2f Hz  v_h=%.2f m/s  contact=%.2f ms  Fpeak=%.1f N  modes=%d  B=%.5f  T60_1=%.1f s\n",
                        midi_to_name((int)std::lround(notes[i].midi)).c_str(), key.frequency_hz(),
                        d.v_h, d.contact_ms, d.peak_force, d.n_phasors, key.parameters().B, key.parameters().T60_1);
        }

    struct Control { int64_t sample; int kind; size_t index; int channel; double position; }; // kind: 0 strike, 1 release, 2 pedal
    std::vector<Control> controls;
    for (size_t i = 0; i < notes.size(); ++i) {
        const Note& n = notes[i];
        controls.push_back({(int64_t)(n.t * piano::FS), 0, i, n.channel, 0.0});
        controls.push_back({(int64_t)((n.t + n.hold) * piano::FS), 1, i, n.channel, 0.0});
    }
    for (const auto& p : performance.pedals)
        controls.push_back({(int64_t)(p.t * piano::FS), 2, 0, p.channel, p.position});
    // On one sample, a pedal change applies before a key release (the release
    // is caught), and a release before a strike of the same key (the release
    // belongs to the earlier strike).
    std::stable_sort(controls.begin(), controls.end(), [](const Control& a, const Control& b) {
        return a.sample != b.sample ? a.sample < b.sample : a.kind > b.kind; });

    // Unbounded renders stop once no voice is active and the output (which
    // includes any sympathetic ring) has stayed below the render's own peak by
    // RENDER_FLOOR for the room tail. The bank's worst-case decay bound is not
    // used: it pads seconds of silence when the resonators hold no energy.
    const int64_t room_tail = (int64_t)instrument::OutputEffects(host_reverb).reverb.tail_samples();
    const double RENDER_FLOOR = 1e-5; // -100 dB, below the 16-bit floor
    const int64_t last_control = controls.empty() ? 0 : controls.back().sample;
    const int64_t fixed = total_dur > 0 ? (int64_t)(total_dur * piano::FS) + room_tail : -1;
    const int64_t limit = fixed > 0 ? fixed : last_control + (int64_t)(200.0 * piano::FS); // safety bound
    std::vector<double> L, R;
    std::vector<float> l(4096), r(4096);
    t0 = Clock::now();
    int64_t position = 0, silent_since = -1; size_t control = 0;
    double peak = 0;
    for (;;) {
        while (control < controls.size() && controls[control].sample <= position) {
            const auto& c = controls[control++];
            if (c.kind == 0) { if (!engine.start(*strikes[c.index], c.channel)) throw std::runtime_error("engine rejected a prepared note"); }
            else if (c.kind == 1) engine.note_off((int)std::lround(notes[c.index].midi), c.channel);
            else engine.pedal(c.channel, c.position);
        }
        if (position >= limit) break;
        if (fixed < 0 && control == controls.size() && silent_since >= 0 && position - silent_since >= room_tail) break;
        int64_t end = std::min(limit, control < controls.size() ? controls[control].sample : position + (int64_t)l.size());
        size_t count = (size_t)std::min<int64_t>(end - position, (int64_t)l.size());
        if (!engine.process(l.data(), r.data(), count)) throw std::runtime_error("engine process failed");
        if (fixed < 0) {
            double block_peak = 0;
            for (size_t i = 0; i < count; ++i) block_peak = std::max({block_peak, std::fabs((double)l[i]), std::fabs((double)r[i])});
            peak = std::max(peak, block_peak);
            const bool quiet = engine.active_voices() == 0 && block_peak <= peak * RENDER_FLOOR;
            if (!quiet) silent_since = -1;
            else if (silent_since < 0) silent_since = position;
        }
        L.insert(L.end(), l.begin(), l.begin() + count);
        R.insert(R.end(), r.begin(), r.begin() + count);
        position += count;
    }
    double stream_seconds = std::chrono::duration<double>(Clock::now() - t0).count();
    piano::fade_tail(L, 0.05); piano::fade_tail(R, 0.05);
    double gain = write_wav_stereo(outfile, L, R);
    if (const char* save = std::getenv("GP_SAVE_SETTINGS"); save && std::string(save) == "1")
        save_render_settings(outfile, performance, L.size(), gain);
    if (verbose)
        std::printf("%zu notes prepared in %.2f s on %u threads; %.1f s of audio streamed in %.2f s (%.1fx realtime); peak voices %zu\n",
                    notes.size(), prepare_seconds, nt, L.size() / (double)piano::FS, stream_seconds,
                    L.size() / (double)piano::FS / std::max(stream_seconds, 1e-9), engine.peak_voices());
    std::cout << "Saved: " << outfile << "\n";
}

// ---------------------------------------------------------------------------
// Standard MIDI File reader. Every channel and patch is treated as piano.
// CC64 (sustain) defers each note's damper engagement to the pedal-up.

static Performance parse_midi(const std::string& path, double vel_scale) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<uint8_t> d((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
    auto u32 = [&](size_t q) { return ((uint32_t)d[q]<<24)|((uint32_t)d[q+1]<<16)
                                     |((uint32_t)d[q+2]<<8)|d[q+3]; };
    auto u16 = [&](size_t q) { return (uint32_t)((d[q]<<8)|d[q+1]); };
    if (d.size() < 14 || u32(0) != 0x4D546864) throw std::runtime_error("not a MIDI file");
    int ntrk = (int)u16(10);
    int division = (int)u16(12);
    size_t p = 8 + u32(4);

    struct On    { long tick; int ch, key, vel; };
    struct Off   { long tick; int ch, key; };
    struct Pedal { long tick; int ch; double position; };
    struct Tempo { long tick; double us_per_qn; };
    std::vector<On> ons; std::vector<Off> offs;
    std::vector<Pedal> pedals; std::vector<Tempo> tempos;
    long last_tick = 0;

    for (int tr = 0; tr < ntrk && p + 8 <= d.size(); ++tr) {
        if (u32(p) != 0x4D54726B) throw std::runtime_error("bad track header");
        size_t end = p + 8 + u32(p + 4);
        p += 8;
        long tick = 0;
        uint8_t status = 0;
        auto vlq = [&]() { long v = 0; uint8_t b;
            do { b = d[p++]; v = (v << 7) | (b & 0x7f); } while (b & 0x80);
            return v; };
        while (p < end) {
            tick += vlq();
            last_tick = std::max(last_tick, tick);
            if (d[p] & 0x80) status = d[p++];
            int ch = status & 0x0f;
            switch (status & 0xf0) {
                case 0x90: { int key = d[p], v = d[p+1]; p += 2;
                    if (v > 0) ons.push_back({tick, ch, key, v});
                    else       offs.push_back({tick, ch, key});
                    break; }
                case 0x80: offs.push_back({tick, ch, d[p]}); p += 2; break;
                case 0xB0: { int c = d[p], v = d[p+1]; p += 2;
                    if (c == 64) pedals.push_back({tick, ch, v / 127.0}); // continuous: half-pedal preserved
                    break; }
                case 0xA0: case 0xE0: p += 2; break;
                case 0xC0: case 0xD0: p += 1; break;
                default:
                    if (status == 0xFF) {
                        uint8_t mt = d[p++];
                        long l = vlq();
                        if (mt == 0x51 && l == 3)
                            tempos.push_back({tick,
                                (double)((d[p]<<16)|(d[p+1]<<8)|d[p+2])});
                        p += l;
                    } else if (status == 0xF0 || status == 0xF7) {
                        p += vlq();
                    } else throw std::runtime_error("unsupported MIDI event");
            }
        }
        p = end;
    }

    // tick -> seconds through the tempo map
    std::sort(tempos.begin(), tempos.end(),
              [](const Tempo& a, const Tempo& b){ return a.tick < b.tick; });
    struct Seg { long tick; double sec, spt; };
    std::vector<Seg> segs;
    if (division & 0x8000) {                   // SMPTE: fixed seconds per tick
        double fps = 256.0 - ((division >> 8) & 0xFF);
        segs.push_back({0, 0.0, 1.0 / (fps * (division & 0xFF))});
    } else {
        double sec = 0.0, spt = 500000e-6 / division;
        long prev = 0;
        segs.push_back({0, 0.0, spt});
        for (const auto& tm : tempos) {
            sec += (tm.tick - prev) * segs.back().spt;
            prev = tm.tick;
            segs.push_back({tm.tick, sec, tm.us_per_qn * 1e-6 / division});
        }
    }
    auto tick_sec = [&](long t) {
        const Seg* s = &segs[0];
        for (const auto& g : segs) { if (g.tick <= t) s = &g; else break; }
        return s->sec + (t - s->tick) * s->spt;
    };

    std::stable_sort(ons.begin(), ons.end(),
                     [](const On& a, const On& b){ return a.tick < b.tick; });
    std::stable_sort(offs.begin(), offs.end(),
                     [](const Off& a, const Off& b){ return a.tick < b.tick; });
    std::stable_sort(pedals.begin(), pedals.end(),
                     [](const Pedal& a, const Pedal& b){ return a.tick < b.tick; });
    Performance result;
    for (const auto& pedal : pedals)
        result.pedals.push_back({tick_sec(pedal.tick), pedal.ch, pedal.position});
    // An unclosed pedal is released at the end of the file.
    for (int channel = 0; channel < 16; ++channel)
        result.pedals.push_back({tick_sec(last_tick), channel, 0.0});

    // Damper matching per (channel, key): a note ends at the first release of
    // its key after its start, so restrikes while the key is down end together
    // with the earlier strike, as one damper would end them. A release on the
    // same tick as a strike belongs to the earlier strike.
    std::map<int, std::deque<size_t>> off_q;
    for (size_t i = 0; i < offs.size(); ++i)
        off_q[(offs[i].ch << 8) | offs[i].key].push_back(i);

    result.notes.reserve(ons.size());
    for (const auto& on : ons) {   // sorted by tick, so consumed releases stay behind
        auto& q = off_q[(on.ch << 8) | on.key];
        while (!q.empty() && offs[q.front()].tick <= on.tick) q.pop_front();
        long t_off = q.empty() ? last_tick : offs[q.front()].tick;
        double t0 = tick_sec(on.tick);
        // Pitches outside the piano sound at the nearest key but keep their own
        // MIDI ownership, as in live playback. The engine handles the pedal.
        result.notes.push_back({t0, (double)on.key,
                                std::clamp(vel_scale * on.vel / 127.0, 0.02, 1.0),
                                std::max(tick_sec(t_off) - t0, 0.02), on.ch});
    }
    std::stable_sort(result.notes.begin(), result.notes.end(),
                     [](const Note& a, const Note& b){ return a.t < b.t; });
    return result;
}

// ---------------------------------------------------------------------------
// Live mode (macOS): CoreMIDI in -> voice engine -> partitioned soundboard
// convolution -> CoreAudio out. Note preparation (the contact simulation)
// runs on the MIDI thread; the audio thread only advances phasors. The two
// communicate over single-producer single-consumer rings, so the audio
// callback never locks or allocates on the steady path.

#if defined(__APPLE__)
#include <CoreMIDI/CoreMIDI.h>
#include "live_diagnostics.hpp"
#include "piano_host.hpp"
#include <AudioToolbox/AudioToolbox.h>
#include <csignal>

namespace live {

static constexpr int BLK = piano::BLOCK_SIZE;          // engine block (samples)

template <typename T, size_t N>
struct SpscRing {
    T buf[N];
    std::atomic<size_t> w{0}, r{0};
    bool has_space() const noexcept {
        return w.load(std::memory_order_relaxed)-r.load(std::memory_order_acquire) < N;
    }
    bool empty() const noexcept {
        return r.load(std::memory_order_relaxed) == w.load(std::memory_order_acquire);
    }
    bool push(const T& v) {
        size_t W = w.load(std::memory_order_relaxed);
        if (W - r.load(std::memory_order_acquire) >= N) return false;
        buf[W % N] = v;
        w.store(W + 1, std::memory_order_release);
        return true;
    }
    bool pop(T& v) {
        size_t R = r.load(std::memory_order_relaxed);
        if (R == w.load(std::memory_order_acquire)) return false;
        v = buf[R % N];
        r.store(R + 1, std::memory_order_release);
        return true;
    }
};

struct Command {
    enum Type { NOTE_ON, NOTE_OFF, PEDAL, CACHED_NOTE_ON } type;
    int key = 0, channel = 0;
    double position = 0; // pedal: 0 up .. 1 down
    int velocity = 0;
    uint64_t seed = 1;
    instrument::PreparedStrike* strike = nullptr;
};

static instrument::Settings instrument_settings() {
    instrument::Settings s;
    s.voicing = host_voicing; s.voice_options = piano_options(); s.reverb = host_reverb;
    if (const char* value = std::getenv("GP_GAIN")) s.gain *= preset::number(value);
    return s;
}

// Device adapter only: command handoff, callback-size FIFO and diagnostics.
// All synthesis, voice ownership and pedal scheduling live in instrument.hpp.
struct Engine {
    instrument::Engine core;
    SpscRing<Command,1024> cmds;
    SpscRing<instrument::PreparedStrike*,1024> trash;
    std::mutex midi_mutex; // serializes producers; never acquired by audio thread
    std::atomic<bool> panic{false};
    bool discarding = false;
    double velocity_gamma = 1;
    uint64_t note_sequence = 1;
    uint8_t midi_status = 0;
    float fifoL[BLK], fifoR[BLK];
    size_t fifo_w = 0, fifo_r = 0;
    std::atomic<long> max_block_us{0};
    std::atomic<int> command_drops{0};
    struct CaptureBlock { std::array<float,BLK> left, right; };
    SpscRing<CaptureBlock,128> captured;
    CaptureBlock capture_block;
    std::vector<float> capL, capR; // written only by the non-audio collector
    bool capture = false;
    Replay* replay = nullptr; // configured before starting the device
    CallbackMonitor* monitor = nullptr;
    piano_host::Adapter* host_adapter = nullptr;
    std::vector<instrument::MidiEvent> host_events; // reserved before audio starts
    size_t processing_failures = 0;
    size_t capture_dropped = 0;

    explicit Engine(const instrument::Settings& settings = instrument_settings()) : core(settings) {
        if (const char* value = std::getenv("GP_VEL_GAMMA")) velocity_gamma = preset::number(value);
        if (!std::isfinite(velocity_gamma) || velocity_gamma <= 0)
            throw std::invalid_argument("GP_VEL_GAMMA must be finite and positive");
        if (const char* value = std::getenv("GP_LIVE_CACHE"); value && std::string(value) == "1") {
            std::printf("Building exact live strike cache...\n"); std::fflush(stdout);
            auto start = std::chrono::steady_clock::now();
            core.build_strike_cache(velocity_gamma);
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            std::printf("Strike cache ready: %.1f s, %.1f MiB response storage.\n", seconds,
                        core.response_storage_bytes()/1048576.0);
        }
    }
    // Call only after callbacks/producers have stopped.
    ~Engine() {
        Command c; while (cmds.pop(c)) if (c.type == Command::NOTE_ON) delete c.strike;
        instrument::PreparedStrike* strike; while (trash.pop(strike)) delete strike;
    }
    void apply(const Command& c) {
        switch (c.type) {
            case Command::NOTE_ON: if (c.strike) core.start(*c.strike,c.channel); break;
            case Command::CACHED_NOTE_ON: core.note_on(c.key,c.velocity,c.seed,c.channel); break;
            case Command::NOTE_OFF: core.note_off(c.key,c.channel); break;
            case Command::PEDAL: core.pedal(c.channel,c.position); break;
        }
    }
    void enqueue(Command c) {
        if (!cmds.push(c)) {
            if (c.type == Command::NOTE_ON) delete c.strike;
            ++command_drops;
            panic.store(true,std::memory_order_release);
        }
    }
    void renderBlock() {
        auto t0 = std::chrono::steady_clock::now();
        if (replay) replay->dispatch([&](const ReplayEvent& e) {
            Command c; c.key=e.key; c.velocity=e.velocity; c.channel=e.channel;
            c.seed=e.seed; c.position=e.down ? 1.0 : 0.0;
            c.type=e.type==0 ? Command::CACHED_NOTE_ON : e.type==1 ? Command::NOTE_OFF : Command::PEDAL;
            apply(c);
        }, BLK);
        discarding |= panic.exchange(false,std::memory_order_acq_rel);
        Command c;
        // Back-pressure prevents ever losing a prepared strike when reclamation
        // is delayed. Consume at most one queue's capacity per audio block.
        for (size_t n = 0; n < 1024 && trash.has_space() && cmds.pop(c); ++n) {
            if (!discarding) apply(c);
            if (c.type == Command::NOTE_ON) trash.push(c.strike);
        }
        if (discarding) { core.reset(); discarding = !cmds.empty(); }
        core.render_block(fifoL,fifoR);
        fifo_r = 0; fifo_w = BLK;
        if (capture) {
            std::copy(fifoL,fifoL+BLK,capture_block.left.begin());
            std::copy(fifoR,fifoR+BLK,capture_block.right.begin());
            if (!captured.push(capture_block)) capture_dropped += BLK;
        }
        long us = (long)std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::steady_clock::now() - t0).count();
        max_block_us.store(std::max(max_block_us.load(),us));
    }
    // Non-audio thread only; permits arbitrarily long captures without growing
    // vectors in the callback. Drain once more after the audio thread stops.
    void collect_capture() {
        CaptureBlock block;
        while (captured.pop(block)) {
            capL.insert(capL.end(),block.left.begin(),block.left.end());
            capR.insert(capR.end(),block.right.begin(),block.right.end());
        }
    }
    bool pull(float* left, float* right, int count) {
        if (host_adapter) {
            host_events.clear();
            uint64_t position=host_adapter->position();
            if (replay) while (replay->next<replay->events.size()
                              && replay->events[replay->next].frame<=position+count) {
                const auto& e=replay->events[replay->next++];
                instrument::MidiEvent event;
                event.offset=size_t(e.frame-position); event.key=e.key; event.velocity=e.velocity;
                event.channel=e.channel; event.seed=e.seed; event.down=e.down!=0;
                event.type=e.type==0 ? instrument::MidiEvent::Type::note_on
                          : e.type==1 ? instrument::MidiEvent::Type::note_off : instrument::MidiEvent::Type::pedal;
                host_events.push_back(event);
            }
            return host_adapter->process(left,right,size_t(count),host_events.data(),host_events.size());
        }
        for (int i = 0; i < count; ++i) {
            if (fifo_r == fifo_w) renderBlock();
            left[i] = fifoL[fifo_r]; right[i] = fifoR[fifo_r]; ++fifo_r;
        }
        return true;
    }
};

static Engine* g_engine = nullptr;
static std::atomic<bool> g_quit{false};

static void handle_midi_bytes(const uint8_t* data, int len) {
    if (!g_engine) return;
    std::lock_guard<std::mutex> producer(g_engine->midi_mutex);
    const double gamma = g_engine->velocity_gamma;
    auto& status = g_engine->midi_status;
    int i = 0;
    while (i < len) {
        uint8_t b = data[i];
        if (b & 0x80) { status = b; ++i; continue; }
        switch (status & 0xf0) {
            case 0x90: case 0x80: {
                if (i + 1 >= len) return;
                int key = data[i], vel = data[i + 1];
                i += 2;
                bool on = ((status & 0xf0) == 0x90) && vel > 0;
                if (on) {
                    double velocity = std::clamp(
                        std::pow(vel / 127.0, gamma), 0.02, 1.0);
                    uint64_t seed = 0x9e3779b97f4a7c15ULL
                                  ^ (g_engine->note_sequence++ * 0x100000001b3ULL)
                                  ^ ((uint64_t)key << 32);
                    Command c; c.channel = status & 0x0f; c.key = key;
                    if (g_engine->core.has_strike_cache()) {
                        c.type = Command::CACHED_NOTE_ON; c.velocity = vel; c.seed = seed;
                    } else {
                        c.type = Command::NOTE_ON;
                        c.strike = new instrument::PreparedStrike(g_engine->core.prepare(key,velocity,seed));
                    }
                    g_engine->enqueue(c);
                } else {
                    Command c; c.type = Command::NOTE_OFF; c.key = key; c.channel = status & 0x0f;
                    g_engine->enqueue(c);
                }
                break;
            }
            case 0xB0: {
                if (i + 1 >= len) return;
                int cc = data[i], val = data[i + 1];
                i += 2;
                if (cc == 64) {
                    Command c; c.type = Command::PEDAL; c.position = val / 127.0; c.channel = status & 0x0f;
                    g_engine->enqueue(c);
                }
                break;
            }
            case 0xC0: case 0xD0: i += 1; break;
            case 0xA0: case 0xE0: i += 2; break;
            default: ++i; break;
        }
    }
}

static void midi_read(const MIDIPacketList* pl, void*, void*) {
    const MIDIPacket* p = &pl->packet[0];
    for (UInt32 i = 0; i < pl->numPackets; ++i) {
        try { handle_midi_bytes(p->data, p->length); }
        catch (...) { if (g_engine) g_engine->panic.store(true,std::memory_order_release); }
        p = MIDIPacketNext(p);
    }
}

static OSStatus audio_cb(void*, AudioUnitRenderActionFlags*,
                         const AudioTimeStamp* stamp, UInt32, UInt32 frames,
                         AudioBufferList* io) {
    float* L = (float*)io->mBuffers[0].mData;
    float* R = (float*)io->mBuffers[io->mNumberBuffers > 1 ? 1 : 0].mData;
    auto* monitor = g_engine->monitor;
    auto start = monitor ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    bool ok=g_engine->pull(L, R, (int)frames);
    if (!ok) {
        ++g_engine->processing_failures;
        std::fill_n(L,frames,0.0f); std::fill_n(R,frames,0.0f);
        g_quit.store(true);
    }
    if (monitor) {
        if (monitor->mute) { std::fill_n(L,frames,0.0f); std::fill_n(R,frames,0.0f); }
        double elapsed = std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
        monitor->record(stamp,frames,elapsed,g_engine->core.active_voices());
    }
    return ok ? noErr : kAudio_ParamError;
}

static std::string cf_str(CFStringRef s) {
    if (!s) return "?";
    char buf[256];
    if (CFStringGetCString(s, buf, sizeof buf, kCFStringEncodingUTF8)) return buf;
    return "?";
}

static void on_sigint(int) { g_quit.store(true); }

static int live_main(bool selftest) {
    g_quit.store(false);
    Engine eng;
    g_engine = &eng;
    if (std::getenv("GP_CAPTURE") || selftest) {
        eng.capture = true;
        eng.capL.reserve(piano::FS * 120);
        eng.capR.reserve(piano::FS * 120);
    }
    // MIDI
    MIDIClientRef client;
    MIDIClientCreate(CFSTR("coopiano"), nullptr, nullptr, &client);
    MIDIPortRef inport;
    MIDIInputPortCreate(client, CFSTR("in"), midi_read, nullptr, &inport);
    MIDIEndpointRef vsrc = 0;
    if (selftest)
        MIDISourceCreate(client, CFSTR("coopiano-selftest"), &vsrc);
    MIDIEndpointRef vdst;
    MIDIDestinationCreate(client, CFSTR("coopiano"), midi_read, nullptr, &vdst);

    ItemCount nsrc = MIDIGetNumberOfSources();
    std::printf("MIDI sources connected:\n");
    for (ItemCount i = 0; i < nsrc; ++i) {
        MIDIEndpointRef src = MIDIGetSource(i);
        MIDIPortConnectSource(inport, src, nullptr);
        CFStringRef name = nullptr;
        MIDIObjectGetStringProperty(src, kMIDIPropertyDisplayName, &name);
        std::printf("  - %s\n", cf_str(name).c_str());
        if (name) CFRelease(name);
    }
    std::printf("(also listening as virtual destination \"coopiano\")\n");

    // Audio
    AudioOutput output(audio_cb);
    output.start();

    std::signal(SIGINT, on_sigint);
    std::printf("live. play your keyboard; Ctrl-C to quit.\n");

    // Collector: reclaims prepared strikes and grows capture storage off audio.
    std::thread reaper([&]() {
        while (!g_quit.load()) {
            instrument::PreparedStrike* strike;
            while (eng.trash.pop(strike)) delete strike;
            eng.collect_capture();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });

    std::thread selftest_thread;
    if (selftest && vsrc) {
        selftest_thread = std::thread([vsrc]() {
            auto ms = [](int remaining) {
                while (remaining > 0 && !g_quit.load()) {
                    int step = std::min(remaining,10);
                    std::this_thread::sleep_for(std::chrono::milliseconds(step));
                    remaining -= step;
                }
            };
            ms(400);
            struct Ev { int wait_ms; std::vector<uint8_t> bytes; };
            // arpeggio at rising velocity, pedal chord, staccato, big bass ff
            std::vector<Ev> seq = {
                {0,    {0x90, 48, 40}}, {350, {0x80, 48, 0}},
                {0,    {0x90, 52, 64}}, {350, {0x80, 52, 0}},
                {0,    {0x90, 55, 90}}, {350, {0x80, 55, 0}},
                {0,    {0x90, 60, 115}}, {350, {0x80, 60, 0}},
                {100,  {0xB0, 64, 127}},                      // pedal down
                {0,    {0x90, 36, 100}}, {120, {0x80, 36, 0}},
                {0,    {0x90, 43, 80}},  {120, {0x80, 43, 0}},
                {0,    {0x90, 48, 80}},  {120, {0x80, 48, 0}},
                {0,    {0x90, 52, 75}},  {120, {0x80, 52, 0}},
                {1200, {0xB0, 64, 0}},                        // pedal up
                {300,  {0x90, 33, 120}}, {0, {0x90, 45, 95}}, // A1+A2 ff
                {1500, {0x80, 33, 0}},   {0, {0x80, 45, 0}},
                {200,  {0x90, 84, 70}},  {200, {0x80, 84, 0}},
            };
            for (auto& e : seq) {
                ms(e.wait_ms);
                if (g_quit.load()) break;
                Byte plbuf[128];
                MIDIPacketList* pl = (MIDIPacketList*)plbuf;
                MIDIPacket* pk = MIDIPacketListInit(pl);
                MIDIPacketListAdd(pl, sizeof plbuf, pk, 0,
                                  e.bytes.size(), e.bytes.data());
                MIDIReceived(vsrc, pl);
            }
            ms(2500);
            g_quit.store(true);
        });
    }

    while (!g_quit.load())
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

    if (selftest_thread.joinable()) selftest_thread.join();
    output.stop();
    // Disconnect producers before reclaiming queued preparations / engine state.
    MIDIClientDispose(client);
    reaper.join();
    eng.collect_capture();
    g_engine = nullptr;

    std::printf("stats: peak voices %d, max block %ld us (budget %d us), "
                "dropped %d, stolen %llu (%llu held), pre-gain peak %.3f\n",
                (int)eng.core.peak_voices(), eng.max_block_us.load(),
                (int)(1e6 * BLK / piano::FS), (int)eng.core.dropped_notes() + eng.command_drops.load(),
                (unsigned long long)eng.core.stolen_voices(), (unsigned long long)eng.core.stolen_held_voices(),
                eng.core.output_peak());
    if (eng.capture_dropped) std::printf("capture full: omitted %zu samples\n",eng.capture_dropped);
    if (eng.capture && !eng.capL.empty()) {
        std::vector<double> L(eng.capL.begin(), eng.capL.end());
        std::vector<double> R(eng.capR.begin(), eng.capR.end());
        const char* path = std::getenv("GP_CAPTURE");
        std::string out = path ? path : "selftest.wav";
        // write at unity: the capture is already post-gain
        double peak = 1e-12;
        for (size_t i = 0; i < L.size(); ++i)
            peak = std::max({peak, std::abs(L[i]), std::abs(R[i])});
        std::printf("capture peak %.3f -> %s\n", peak, out.c_str());
        write_wav_stereo(out, L, R);
    }
    return 0;
}

} // namespace live
#include "live_bench.hpp"
#endif // __APPLE__

// ---------------------------------------------------------------------------

struct Audition {
    std::vector<double> notes{45, 48, 52};
    std::vector<double> velocities{.25, .45, .65, .85};
    double hold = 1.1, gap = .55, tail = 1.15;
    std::string output = "audition.wav";
};

static std::vector<std::string> comma_list(const std::string& value) {
    std::vector<std::string> parts;
    size_t start = 0;
    for (;;) {
        size_t end = value.find(',', start);
        auto part = preset::trim(value.substr(start, end == std::string::npos ? end : end - start));
        if (part.empty()) throw std::invalid_argument("empty item in comma-separated list");
        parts.push_back(part);
        if (end == std::string::npos) return parts;
        start = end + 1;
    }
}

static Audition parse_audition(int argc, char* argv[]) {
    Audition a;
    std::set<std::string> seen;
    for (int i = 2; i < argc; ++i) {
        std::string flag = argv[i];
        if (!seen.insert(flag).second) throw std::invalid_argument("duplicate option " + flag);
        if (flag != "--chord" && flag != "--velocities" && flag != "--hold"
            && flag != "--gap" && flag != "--tail" && flag != "--output")
            throw std::invalid_argument("unknown audition option " + flag);
        if (++i == argc) throw std::invalid_argument("missing value for " + flag);
        std::string value = argv[i];
        if (flag == "--chord") {
            a.notes.clear();
            for (const auto& note : comma_list(value)) {
                double midi = note_to_midi(note);
                piano::make_key(midi, host_voicing); // Validate before rendering.
                a.notes.push_back(midi);
            }
        } else if (flag == "--velocities") {
            a.velocities.clear();
            for (const auto& item : comma_list(value)) {
                double velocity = preset::number(item);
                if (velocity < .02 || velocity > 1)
                    throw std::invalid_argument("audition velocities must be in [0.02,1]");
                a.velocities.push_back(velocity);
            }
        } else if (flag == "--output") {
            if (value.empty()) throw std::invalid_argument("output path is empty");
            a.output = value;
        } else {
            double seconds = preset::number(value);
            if (seconds < 0 || seconds > 30 || (flag == "--hold" && seconds < .02))
                throw std::invalid_argument(flag + " must be 0..30 seconds (hold at least 0.02)");
            if (flag == "--hold") a.hold = seconds;
            else if (flag == "--gap") a.gap = seconds;
            else a.tail = seconds;
        }
    }
    if (a.notes.size() > 12 || a.velocities.size() > 32)
        throw std::invalid_argument("audition supports up to 12 chord notes and 32 velocities");
    return a;
}

static int run_cli(int argc, char* argv[]) {
    std::string arg1 = (argc > 1) ? argv[1] : "A4";

    if (arg1 == "--help" || arg1 == "help") {
        std::cout << "Usage: coopiano [--preset FILE] NOTE [velocity] [duration] [hold]\n"
                  << "       coopiano [--preset FILE] demo\n"
                  << "       coopiano [--preset FILE] FILE.mid [velocity_scale] [output.wav]\n"
                  << "       coopiano [--preset FILE] live [--selftest]  (macOS)\n"
                  << "       coopiano [--preset FILE] live-bench TIMELINE REPORT_PREFIX [--audible] [--host-rate HZ] (macOS)\n"
                  << "       coopiano [--preset FILE] audition [options]\n\n"
                  << "Audition: --chord A2,C3,E3 --velocities 0.25,0.45,0.65,0.85\n"
                  << "          --hold 1.1 --gap 0.55 --tail 1.15 --output audition.wav\n"
                  << "Durations are seconds. Chord notes sound together. One gain is applied\n"
                  << "to the whole velocity progression; chords are not individually normalized.\n"
                  << "GP_SAVE_SETTINGS=1 optionally writes a settings sidecar: output.wav.ini.\n"
                  << "Edit presets and restart playback; live reloading is not supported.\n"
                  << "Room reverb is on by default. GP_NO_REVERB=1 disables it for any command.\n"
                  << "Use [reverb] in presets or GP_REVERB_MIX=0..1 to set its wet level.\n"
                  << "Upper longitudinal tones are softened by default; GP_SOFT_LONGITUDINAL=0 disables it.\n"
                  << "Unset it or use 1 for the default sound. Saved settings record it as a comment.\n";
        return 0;
    }

    if (arg1 == "audition") {
        auto a = parse_audition(argc, argv);
        Performance performance;
        for (size_t i = 0; i < a.velocities.size(); ++i)
            for (double note : a.notes)
                performance.notes.push_back({i * (a.hold + a.gap), note, a.velocities[i], a.hold});
        double duration = (a.velocities.size() - 1) * (a.hold + a.gap) + a.hold + a.tail;
        render(performance, duration, a.output, true);
        return 0;
    }

#if defined(__APPLE__)
    if (arg1 == "live-bench") {
        if (argc<4) throw std::invalid_argument("usage: live-bench TIMELINE REPORT_PREFIX [--audible] [--host-rate HZ]");
        bool mute=true; unsigned rate=0;
        for (int i=4;i<argc;++i) {
            std::string flag=argv[i];
            if (flag=="--audible" && mute) mute=false;
            else if (flag=="--host-rate" && !rate && i+1<argc) {
                std::string value=argv[++i]; size_t used=0;
                unsigned long parsed=std::stoul(value,&used);
                if (used!=value.size() || parsed>96000 || !piano_host::supported_rate(unsigned(parsed)))
                    throw std::invalid_argument("host rate must be 44100, 48000, 88200 or 96000");
                rate=unsigned(parsed);
            } else throw std::invalid_argument("unknown, duplicate or incomplete live-bench option: "+flag);
        }
        return live::benchmark_main(argv[2],argv[3],mute,rate);
    }
    if (arg1 == "live") {
        bool selftest = argc > 2 && std::string(argv[2]) == "--selftest";
        return live::live_main(selftest);
    }
#endif

    if (arg1.size() > 4 && arg1.substr(arg1.size() - 4) == ".mid") {
        double vel_scale = (argc > 2) ? std::stod(argv[2]) : 1.0;
        std::string out = (argc > 3) ? argv[3]
                        : arg1.substr(0, arg1.size() - 4) + ".wav";
        auto performance = parse_midi(arg1, vel_scale);
        double dur = performance.notes.empty() ? 0.0 : performance.notes.back().t;
        std::printf("%s: %zu notes, %.1f s, velocity scale %.2f\n",
                    arg1.c_str(), performance.notes.size(), dur, vel_scale);
        render(performance, -1.0, out, true);
        return 0;
    }

    if (arg1 == "demo") {
        auto N = [](double t, const char* nm, double v, double h) {
            return Note{t, note_to_midi(nm), v, h};
        };
        Performance demo; demo.notes = {
            N(0.00, "C2", 0.62, 7.5), N(0.00, "G2", 0.50, 7.5),
            N(0.45, "C3", 0.52, 7.0), N(0.90, "E3", 0.50, 6.6),
            N(1.35, "G3", 0.55, 6.2), N(1.80, "B3", 0.52, 5.7),
            N(2.25, "D4", 0.55, 5.3), N(2.70, "E4", 0.58, 4.9),
            N(3.40, "C5", 0.65, 4.2), N(3.40, "E4", 0.45, 4.2),
            N(3.40, "G4", 0.48, 4.2), N(3.40, "B4", 0.50, 4.2),
            N(5.60, "C6", 0.40, 2.5),
            N(7.60, "A1", 0.92, 5.0), N(7.60, "A2", 0.78, 5.0),
            N(9.80, "E2", 0.55, 3.0),
        };
        render(demo, 13.5, "demo_piano.wav", true);
        return 0;
    }

    double velocity = (argc > 2) ? std::stod(argv[2]) : 0.5;
    double duration = (argc > 3) ? std::stod(argv[3]) : 8.0;
    double hold     = (argc > 4) ? std::stod(argv[4]) : duration;

    Performance single; single.notes = {{0.0, note_to_midi(arg1), velocity, hold}};
    render(single, duration, arg1 + "_gp.wav", true);
    return 0;
}

int main(int argc, char* argv[]) {
    try {
        std::vector<std::string> arguments{argv[0]};
        std::string path;
        bool preset_seen = false;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--preset") {
                if (preset_seen) throw std::invalid_argument("--preset may only appear once");
                preset_seen = true;
                if (++i == argc || std::string(argv[i]).empty())
                    throw std::invalid_argument("--preset requires a file path");
                path = argv[i];
            } else arguments.emplace_back(argv[i]);
        }
        configure_voicing(path);
        // Resolve and validate per-voice environment controls before workers start.
        const auto& options = piano_options();
        if (!std::isfinite(options.damper_scale) || options.damper_scale <= 0
            || !std::isfinite(options.knock_gain) || options.knock_gain < 0)
            throw std::invalid_argument("GP_DAMP_SCALE must be positive and GP_KNOCK_GAIN nonnegative; both finite");
        std::vector<char*> raw;
        for (auto& argument : arguments) raw.push_back(argument.data());
        return run_cli((int)raw.size(), raw.data());
    } catch (const std::exception& error) {
        std::cerr << "coopiano: " << error.what() << '\n';
        return 1;
    }
}
