// Platform-independent instrument orchestration for the app and future hosts.
// No files, environment variables, devices, threads, or plugin SDK dependencies.
#ifndef SYNTHPIANO_INSTRUMENT_HPP
#define SYNTHPIANO_INSTRUMENT_HPP
#include "piano.hpp"
#include "reverb.hpp"
#include "streaming_soundboard.hpp"

namespace instrument {
struct Settings {
    piano::Voicing voicing;
    piano::Options voice_options;
    room::Options reverb;
    size_t maximum_voices = 160;
    double gain = .0011;
};

// Key/pedal state shared by the engine and its sympathetic bank.
// All channels share one instrument body; their primary sustain remains separate.
struct ResonanceState {
    piano::SympatheticResonance bank;
    std::array<std::array<unsigned, 88>, 16> held{};
    std::array<double, 16> pedals{}; // sustain pedal position per channel, 0 up .. 1 down
    explicit ResonanceState(const piano::Voicing& voicing) : bank(voicing) {}
    void key(int channel, int midi, bool down) {
        if (channel < 0 || channel >= 16 || midi < 0 || midi > 127)
            throw std::invalid_argument("instrument: invalid MIDI key/channel");
        int k = std::clamp(midi, 21, 108) - 21;
        auto& count = held[channel][k];
        if (down) {
            if (count == std::numeric_limits<unsigned>::max())
                throw std::length_error("instrument: held-key count overflow");
            ++count;
        } else count = 0; // one damper per key: any release damps the string
        bool any = false;
        for (const auto& channel_keys : held) any |= channel_keys[k] != 0;
        bank.set_key_held(k + 21, any);
    }
    void pedal(int channel, double position) {
        if (channel < 0 || channel >= 16 || !std::isfinite(position) || position < 0 || position > 1)
            throw std::invalid_argument("instrument: invalid MIDI channel or pedal position");
        pedals[channel] = position;
        // The bank is one instrument: its dampers lift as far as any channel's pedal.
        bank.set_pedal_lift(1.0 - piano::detail::damper_contact(*std::max_element(pedals.begin(), pedals.end())));
    }
    void pedal(int channel, bool down) { pedal(channel, down ? 1.0 : 0.0); }
    void reset() noexcept { bank.reset(); held = {}; pedals = {}; }
};

// Shared output effects after either whole-buffer or partitioned convolution.
struct OutputEffects {
    piano::DcBlocker dc[2];
    room::StereoReverb reverb;
    explicit OutputEffects(const room::Options& options) : reverb(piano::FS, options) {}
    void process(double* left, double* right, size_t count) {
        piano::detail::validate_buffer(left, count);
        piano::detail::validate_buffer(right, count);
        if (piano::detail::overlaps({left,count}, {right,count}))
            throw std::invalid_argument("instrument: output channels overlap");
        for (size_t i = 0; i < count; ++i) {
            left[i] = dc[0].process(left[i]);
            right[i] = dc[1].process(right[i]);
        }
        reverb.process(left, right, count);
    }
    void reset() noexcept { dc[0].reset(); dc[1].reset(); reverb.reset(); }
};

inline void make_soundboard(std::vector<double>& left, std::vector<double>& right,
                            const piano::Voicing& voicing) {
    piano::make_soundboard_ir(left, right, 0xb0a4d5eedULL, voicing);
}

// Preparation is explicit and may allocate/run the nonlinear contact solver.
// start() consumes neither this object nor its storage. Keep preparation and
// destruction off the audio callback. Reusing it deliberately repeats its seed.
class PreparedStrike {
    int key_;
    piano::PreparedNote note_;
    PreparedStrike(int key, piano::PreparedNote note) : key_(key), note_(std::move(note)) {}
    friend class Engine;
public:
    int key() const noexcept { return key_; }
    const piano::PreparedNote& note() const noexcept { return note_; }
};

// One immutable response for every physical key and incoming MIDI velocity.
// The velocity curve is baked into the bank: exact mapped values, no rounding
// to a different velocity or interpolation. Workspaces belong to the audio owner.
class StrikeBank {
    std::vector<piano::ContactResponse> responses_;
    std::vector<std::unique_ptr<piano::StrikeWorkspace>> workspaces_;
public:
    StrikeBank(const std::vector<piano::Key>& keys, const piano::Options& options, double gamma) {
        if (keys.size() != 88 || !std::isfinite(gamma) || gamma <= 0)
            throw std::invalid_argument("instrument: invalid strike bank setup");
        responses_.reserve(88*127); workspaces_.reserve(88);
        for (const auto& key : keys) {
            workspaces_.push_back(std::make_unique<piano::StrikeWorkspace>(key, options));
            for (int velocity = 1; velocity <= 127; ++velocity)
                responses_.emplace_back(key, std::clamp(std::pow(velocity/127.0, gamma), .02, 1.0));
        }
    }
    const piano::PreparedNote* prepare(int physical_key, int velocity, uint64_t seed) noexcept {
        if (physical_key < 21 || physical_key > 108 || velocity < 1 || velocity > 127) return nullptr;
        size_t key = (size_t)(physical_key-21);
        auto& workspace = *workspaces_[key];
        if (!workspace.prepare(responses_[key*127+(size_t)velocity-1], seed)) return nullptr;
        return &workspace.note();
    }
    size_t response_storage_bytes() const noexcept {
        size_t bytes = responses_.capacity()*sizeof(piano::ContactResponse);
        for (const auto& response : responses_) bytes += response.storage_bytes();
        return bytes; // excludes workspaces, shared keys and allocator overhead
    }
};

struct MidiEvent {
    enum class Type { note_on, note_off, pedal };
    size_t offset = 0; // samples from the start of this process() call
    Type type = Type::note_on;
    int key = 60, velocity = 64, channel = 0;
    uint64_t seed = 1;
    bool down = false;      // pedal: used when position is negative
    double position = -1;   // pedal: continuous 0 (up) .. 1 (down), e.g. CC64/127
};

class Engine {
    struct Slot {
        piano::Voice voice;
        int key = 0, channel = 0;
        uint64_t order = 0;
        // Mean-square bridge force over the voice's last completed level
        // window, the measure used to choose the quietest voice to steal.
        // Windows follow the voice's own sample clock, so the choice does not
        // depend on host buffer boundaries. Negative until one window completes.
        double level = -1, energy = 0;
        size_t energy_count = 0;
        bool held = false, occupied = false;
    };
    Settings settings_;
    std::vector<piano::Key> keys_;
    std::vector<Slot> slots_;
    std::unique_ptr<StrikeBank> strike_bank_;
    std::vector<size_t> active_; // capacity fixed during construction
    ResonanceState resonance_;
    piano::PartitionedConvolver board_;
    detail::StreamingSoundboard stream_board_;
    // Spatial imaging: one soundboard per bridge region, fed by the voices
    // routed by key position. Empty when spread and position_depth are zero,
    // in which case the single shared soundboard above is used.
    std::vector<piano::PartitionedConvolver> region_boards_;
    std::vector<detail::StreamingSoundboard> region_streams_;
    std::vector<double> region_force_; // SOUNDBOARD_REGIONS rows of one block
    std::array<std::pair<int,double>,88> key_region_{}; // lower region and crossfade toward the next
    std::array<double,piano::BLOCK_SIZE> region_left_{}, region_right_{};
    OutputEffects effects_;
    uint64_t order_ = 0, dropped_ = 0, stolen_ = 0, stolen_held_total_ = 0;
    size_t peak_voices_ = 0;
    double output_peak_ = 0;
    std::array<double,piano::BLOCK_SIZE> force_{}, left_{}, right_{}, voice_scratch_{};
    // Each physical key's own share of the force, 88 rows of one block, for a
    // sympathetic bank that withholds a key's own drive (resonance.own_drive < 1).
    std::vector<double> own_force_;
    // A stolen voice is rendered 10 ms ahead under a raised-cosine fade into
    // this ring, then its slot is reused immediately. Bounded work per steal;
    // no reserve slots, allocation or delay for the incoming note.
    static constexpr size_t STEAL_FADE_SAMPLES = (size_t)(0.010*piano::FS);
    static constexpr size_t LEVEL_WINDOW_SAMPLES = 2048; // about 46 ms
    std::array<double,STEAL_FADE_SAMPLES> fade_window_{}, fade_scratch_{}, fade_tail_{};
    size_t fade_tail_position_ = 0, fade_tail_remaining_ = 0;
    size_t stream_position_ = 0;
    enum class RenderMode { unused, block, stream };
    RenderMode render_mode_ = RenderMode::unused;

    static Settings checked(Settings s) {
        piano::validate_voicing(s.voicing);
        piano::detail::validate_options(s.voice_options);
        room::validate_options(s.reverb);
        if (s.maximum_voices < 1 || s.maximum_voices > 1024)
            throw std::invalid_argument("instrument: maximum_voices must be in [1,1024]");
        if (!std::isfinite(s.gain) || s.gain < 0)
            throw std::invalid_argument("instrument: gain must be finite and nonnegative");
        return s;
    }
    static bool valid_channel(int channel) noexcept { return channel >= 0 && channel < 16; }
    static bool valid_key(int key) noexcept { return key >= 0 && key <= 127; }
public:
    // Setup only. Fixed 44.1 kHz; convolution uses 256-sample blocks. Every instance owns
    // its settings, reserved voices, resonance, convolution, and effect history.
    explicit Engine(const Settings& settings = {})
        : settings_(checked(settings)), resonance_(settings_.voicing), effects_(settings_.reverb) {
        piano::VoiceCapacity capacity;
        keys_.reserve(88);
        for (int key = 21; key <= 108; ++key) {
            keys_.push_back(piano::make_key(key, settings_.voicing));
            capacity.include(piano::voice_capacity(keys_.back()));
        }
        slots_.resize(settings_.maximum_voices);
        active_.reserve(settings_.maximum_voices);
        if (settings_.voicing.resonance.own_drive < 1.0 && settings_.voicing.resonance.gain != 0.0)
            own_force_.assign(88 * piano::BLOCK_SIZE, 0.0);
        for (auto& slot : slots_) slot.voice.reserve(capacity);
        for (size_t i = 0; i < STEAL_FADE_SAMPLES; ++i)
            fade_window_[i] = 0.5*(1.0+std::cos(piano::detail::PI*(double)i/STEAL_FADE_SAMPLES));
        const auto& sb = settings_.voicing.soundboard;
        if (sb.spread > 0 || sb.position_depth > 0) {
            // Region heads carry the image; the shared board takes the tail.
            std::vector<std::vector<double>> lefts, rights;
            std::vector<double> tail_left, tail_right;
            piano::make_soundboard_regions(lefts, rights, 0xb0a4d5eedULL, settings_.voicing, &tail_left, &tail_right);
            region_boards_.resize(lefts.size()); region_streams_.resize(lefts.size());
            for (size_t r = 0; r < lefts.size(); ++r) { region_boards_[r].init(lefts[r], rights[r]); region_streams_[r].init(lefts[r], rights[r]); }
            board_.init(tail_left, tail_right);
            stream_board_.init(tail_left, tail_right);
            region_force_.assign(lefts.size() * piano::BLOCK_SIZE, 0.0);
            for (int key = 0; key < 88; ++key) {
                double x = piano::detail::bridge_position(21 + key) * piano::detail::SOUNDBOARD_REGIONS - 0.5;
                int lower = (int)std::floor(std::clamp(x, 0.0, (double)piano::detail::SOUNDBOARD_REGIONS - 1));
                key_region_[key] = {lower, std::clamp(x - lower, 0.0, 1.0)};
            }
        } else {
            std::vector<double> ir_left, ir_right;
            make_soundboard(ir_left, ir_right, settings_.voicing);
            board_.init(ir_left, ir_right);
            stream_board_.init(ir_left, ir_right);
        }
    }
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&) = delete;
    Engine& operator=(Engine&&) = delete;

    const Settings& settings() const noexcept { return settings_; }
    // Read-only, safe concurrently with rendering: only immutable setup is read.
    PreparedStrike prepare(int midi_key, double velocity, uint64_t seed) const {
        if (!valid_key(midi_key)) throw std::invalid_argument("instrument: MIDI key must be in [0,127]");
        return PreparedStrike(midi_key, piano::prepare_strike(
            keys_[(size_t)(std::clamp(midi_key,21,108)-21)], velocity, seed, settings_.voice_options));
    }
    // Audio thread: bounded scans/copies, no allocation, destruction or contact
    // simulation. False means rejected; invalid input leaves state unchanged.
    // Prepared strikes from another instrument may be used if capacity permits;
    // their note voicing is retained, while this engine supplies the body/room.
    bool start(const PreparedStrike& strike, int channel = 0) noexcept {
        return start_note(strike.note_, strike.key_, channel);
    }
    // Setup/stopped engine only. A replacement is built in full before it is
    // installed; failure preserves the current bank. Voicing/options are fixed
    // for the Engine lifetime. To change them, construct a new Engine off audio.
    void build_strike_cache(double velocity_gamma = 1) {
        auto replacement = std::make_unique<StrikeBank>(keys_, settings_.voice_options, velocity_gamma);
        strike_bank_.swap(replacement);
    }
    bool has_strike_cache() const noexcept { return bool(strike_bank_); }
    size_t response_storage_bytes() const noexcept {
        return strike_bank_ ? strike_bank_->response_storage_bytes() : 0;
    }
    // Audio owner only. No allocation, lock, solver or fallback. Zero velocity
    // follows MIDI note-off semantics. Out-of-range pitches retain their MIDI
    // ownership but sound at the nearest physical key, just like prepare().
    bool note_on(int key, int velocity, uint64_t seed, int channel = 0) noexcept {
        if (!valid_key(key) || !valid_channel(channel) || velocity < 0 || velocity > 127) return false;
        if (!velocity) return note_off(key, channel);
        if (!strike_bank_) return false;
        const auto* note = strike_bank_->prepare(std::clamp(key,21,108), velocity, seed);
        return note ? start_note(*note, key, channel) : reject_note(key,channel);
    }
    // Fade out the quietest stealable voice now: released or pedal-sustained
    // voices first, a physically held voice only as a last resort, never a
    // voice that has not rendered yet. Hosts sharing one voice budget across
    // several engines use this before sending a note elsewhere. Audio owner
    // only; bounded work, no allocation. False when nothing can be stolen.
    bool steal_voice() noexcept {
        size_t victim = select_victim();
        if (victim == active_.size()) return false;
        steal(victim);
        return true;
    }
    static constexpr size_t steal_fade_samples() noexcept { return STEAL_FADE_SAMPLES; }
private:
    // A rejected strike never sounded and needs no bookkeeping: its note-off
    // damps the key like any other (see note_off).
    bool reject_note(int, int) noexcept { ++dropped_; return false; }
    // Index into active_, or active_.size() when no voice is stealable. A
    // voice younger than one level window (about 46 ms) has no measured level
    // and is never stolen, so a chord cannot steal from itself.
    size_t select_victim() const noexcept {
        size_t victim = active_.size();
        int best_tier = 0; double best_level = 0;
        for (size_t i = 0; i < active_.size(); ++i) {
            const auto& slot = slots_[active_[i]];
            if (slot.level < 0) continue;
            int tier = slot.held ? 1 : 0;
            if (victim == active_.size() || tier < best_tier
                || (tier == best_tier && slot.level < best_level)) {
                victim = i; best_tier = tier; best_level = slot.level;
            }
        }
        return victim;
    }
    void steal(size_t victim) noexcept {
        auto& slot = slots_[active_[victim]];
        fade_scratch_.fill(0.0);
        slot.voice.render_bridge_force({fade_scratch_.data(),STEAL_FADE_SAMPLES});
        size_t position = fade_tail_position_;
        for (size_t i = 0; i < STEAL_FADE_SAMPLES; ++i) {
            fade_tail_[position] += fade_scratch_[i]*fade_window_[i];
            if (++position == STEAL_FADE_SAMPLES) position = 0;
        }
        fade_tail_remaining_ = STEAL_FADE_SAMPLES;
        // A stolen held key stays held in the sympathetic bank until its
        // note-off damps it, like every other voice of that key.
        if (slot.held) ++stolen_held_total_;
        slot.voice.stop();
        slot.occupied = slot.held = false;
        active_[victim] = active_.back(); active_.pop_back();
        ++stolen_;
    }
    // A free slot index, stealing one when the pool is full; slots_.size()
    // means the note is rejected.
    size_t acquire_slot() noexcept {
        size_t free = 0;
        while (free < slots_.size() && slots_[free].occupied) ++free;
        if (free < slots_.size()) return free;
        size_t victim = select_victim();
        if (victim == active_.size()) return slots_.size();
        size_t index = active_[victim];
        steal(victim);
        return index;
    }
    bool start_note(const piano::PreparedNote& note, int key, int channel) noexcept {
        if (!valid_channel(channel) || !note.valid()) return false;
        // Every slot reserves the same capacity; check before stealing for it.
        if (!slots_.front().voice.can_start(note)) return false;
        size_t index = acquire_slot();
        if (index == slots_.size()) return reject_note(key,channel);
        auto& slot = slots_[index];
        if (!slot.voice.start(note)) return false;
        slot.key = key; slot.channel = channel; slot.order = order_++;
        slot.level = -1; slot.energy = 0; slot.energy_count = 0;
        slot.held = slot.occupied = true;
        active_.push_back(index);
        resonance_.key(channel, slot.key, true);
        peak_voices_ = std::max(peak_voices_, active_.size());
        return true;
    }
public:
    // A released key's damper follows the channel's pedal: seated with the
    // pedal up, lifted with it down, partially in between (half-pedal).
    double released_contact(int channel) const noexcept {
        return piano::detail::damper_contact(resonance_.pedals[channel]);
    }
    // Damper semantics: a key has one damper, so its note-off releases every
    // held voice of that key/channel at once (the pedal still sustains them),
    // including restrikes that arrived while the key was down. There is no
    // per-note FIFO, and a repeated or unmatched note-off is harmless.
    bool note_off(int key, int channel = 0) noexcept {
        if (!valid_key(key) || !valid_channel(channel)) return false;
        const double contact = released_contact(channel);
        for (size_t index : active_) {
            auto& slot = slots_[index];
            if (slot.key != key || slot.channel != channel || !slot.held) continue;
            slot.held = false;
            slot.voice.damper(contact);
        }
        resonance_.key(channel, key, false);
        return true;
    }
    // Continuous sustain pedal, 0 (up) to 1 (down). Every released voice on
    // the channel that is still sounding follows it: a re-pressed pedal
    // catches them, lifting their dampers again from wherever the damping had
    // got to and keeping the remaining amplitude and phase; an intermediate
    // position damps them partially. Held keys are unaffected.
    bool pedal(int channel, double position) noexcept {
        if (!valid_channel(channel) || !std::isfinite(position) || position < 0 || position > 1) return false;
        resonance_.pedal(channel, position);
        const double contact = released_contact(channel);
        for (size_t index : active_) {
            auto& slot = slots_[index];
            if (slot.channel == channel && !slot.held) slot.voice.damper(contact);
        }
        return true;
    }
    bool pedal(int channel, bool down) noexcept { return pedal(channel, down ? 1.0 : 0.0); }
    // MIDI All Notes Off: every key on the channel is released as if its
    // note-off arrived now; the pedal still sustains the voices.
    bool all_notes_off(int channel) noexcept {
        if (!valid_channel(channel)) return false;
        const double contact = released_contact(channel);
        for (size_t index : active_) {
            auto& slot = slots_[index];
            if (slot.channel != channel || !slot.held) continue;
            slot.held = false;
            slot.voice.damper(contact);
        }
        for (int key = 21; key <= 108; ++key) resonance_.key(channel, key, false);
        return true;
    }
private:
    void render_force(size_t offset, size_t count) {
        double* force = force_.data()+offset;
        std::fill_n(force, count, 0.0);
        const bool own = !own_force_.empty(), regions = !region_force_.empty();
        const size_t K = region_boards_.size();
        if (own) for (int key = 0; key < 88; ++key)
            std::fill_n(own_force_.data()+key*piano::BLOCK_SIZE+offset, count, 0.0);
        if (regions) for (size_t r = 0; r < K; ++r)
            std::fill_n(region_force_.data()+r*piano::BLOCK_SIZE+offset, count, 0.0);
        for (size_t i = 0; i < active_.size();) {
            auto& slot = slots_[active_[i]];
            // Each voice renders into a zeroed scratch so its level can be
            // measured; the summed force is bit-identical to direct rendering.
            std::fill_n(voice_scratch_.data(), count, 0.0);
            bool sounding = slot.voice.render_bridge_force({voice_scratch_.data(),count});
            const int physical = std::clamp(slot.key,21,108)-21;
            double* own_row = own ? own_force_.data()+physical*piano::BLOCK_SIZE+offset : nullptr;
            double* lower_row = nullptr; double* upper_row = nullptr; double upper_weight = 0;
            if (regions) {
                const auto [lower, fraction] = key_region_[physical];
                lower_row = region_force_.data()+lower*piano::BLOCK_SIZE+offset;
                if (fraction > 0 && lower+1 < (int)K) { upper_row = region_force_.data()+(lower+1)*piano::BLOCK_SIZE+offset; upper_weight = fraction; }
            }
            const double lower_weight = 1.0-upper_weight;
            for (size_t j = 0; j < count; ++j) {
                force[j] += voice_scratch_[j];
                if (own_row) own_row[j] += voice_scratch_[j];
                if (lower_row) { lower_row[j] += lower_weight*voice_scratch_[j]; if (upper_row) upper_row[j] += upper_weight*voice_scratch_[j]; }
                slot.energy += voice_scratch_[j]*voice_scratch_[j];
                if (++slot.energy_count == LEVEL_WINDOW_SAMPLES) {
                    slot.level = slot.energy/(double)LEVEL_WINDOW_SAMPLES;
                    slot.energy = 0; slot.energy_count = 0;
                }
            }
            if (!sounding && !slot.held) {
                slot.voice.stop(); slot.occupied = false;
                active_[i] = active_.back(); active_.pop_back();
            } else ++i;
        }
        if (fade_tail_remaining_) {
            size_t length = std::min(count, fade_tail_remaining_);
            for (size_t j = 0; j < length; ++j) {
                force[j] += fade_tail_[fade_tail_position_];
                fade_tail_[fade_tail_position_] = 0.0;
                if (++fade_tail_position_ == STEAL_FADE_SAMPLES) fade_tail_position_ = 0;
            }
            fade_tail_remaining_ -= length;
        }
        const piano::ConstAudioBuffer own_buffer = own
            ? piano::ConstAudioBuffer{own_force_.data()+offset, own_force_.size()-offset}
            : piano::ConstAudioBuffer{};
        resonance_.bank.process({force,count}, own_buffer, piano::BLOCK_SIZE);
        if (regions) {
            // Whatever the regions do not account for (the sympathetic wet
            // signal and stolen-voice fades) is spread evenly across them, so
            // the regions always sum to the force the bank produced.
            for (size_t j = 0; j < count; ++j) {
                double routed = 0;
                for (size_t r = 0; r < K; ++r) routed += region_force_[r*piano::BLOCK_SIZE+offset+j];
                const double share = (force[j]-routed)/(double)K;
                for (size_t r = 0; r < K; ++r) region_force_[r*piano::BLOCK_SIZE+offset+j] += share;
            }
        }
    }
    // Soundboard stage for one span: the shared board, or every region board
    // summed. Region outputs accumulate into left_/right_ at the same offset.
    void convolve_span(size_t offset, size_t count, bool streaming) {
        const size_t K = region_boards_.size();
        if (!K) {
            if (streaming)
                stream_board_.process({force_.data()+offset,count}, {left_.data()+offset,count}, {right_.data()+offset,count});
            else
                board_.process({force_.data(),force_.size()}, {left_.data(),left_.size()}, {right_.data(),right_.size()});
            return;
        }
        const size_t span_offset = streaming ? offset : 0, span = streaming ? count : force_.size();
        // Shared tail of the summed force first, then each region's head.
        if (streaming)
            stream_board_.process({force_.data()+span_offset,span}, {left_.data()+span_offset,span}, {right_.data()+span_offset,span});
        else
            board_.process({force_.data(),force_.size()}, {left_.data(),left_.size()}, {right_.data(),right_.size()});
        for (size_t r = 0; r < K; ++r) {
            const double* input = region_force_.data()+r*piano::BLOCK_SIZE+span_offset;
            if (streaming) region_streams_[r].process({input,span}, {region_left_.data(),span}, {region_right_.data(),span});
            else region_boards_[r].process({input,span}, {region_left_.data(),span}, {region_right_.data(),span});
            for (size_t j = 0; j < span; ++j) { left_[span_offset+j] += region_left_[j]; right_[span_offset+j] += region_right_[j]; }
        }
    }
    void finish_block() {
        convolve_span(0, piano::BLOCK_SIZE, false);
        effects_.process(left_.data(),right_.data(),left_.size());
        for (size_t i = 0; i < left_.size(); ++i)
            output_peak_ = std::max({output_peak_,std::abs(left_[i]),std::abs(right_[i])});
    }
    static bool valid_outputs(float* left, float* right, size_t count) noexcept {
        if (!count) return true;
        if (!left || !right) return false;
        std::less<const float*> before;
        return !(before(left, right+count) && before(right, left+count));
    }
public:
    // Legacy immediate block interface. Do not mix it with process() between
    // resets. Both paths use the same voice/body/effect processing.
    void render_block(float* left, float* right) {
        if (!valid_outputs(left,right,piano::BLOCK_SIZE))
            throw std::invalid_argument("instrument: invalid output channels");
        if (render_mode_ == RenderMode::stream)
            throw std::logic_error("instrument: reset before switching render interfaces");
        render_mode_ = RenderMode::block;
        render_force(0, piano::BLOCK_SIZE);
        finish_block();
        for (size_t i = 0; i < left_.size(); ++i) {
            left[i] = (float)piano::limit_output(left_[i]*settings_.gain);
            right[i] = (float)piano::limit_output(right_[i]*settings_.gain);
        }
    }
    // Arbitrary buffers at FS (44.1 kHz). Events are sorted by offset, with
    // caller order retained for ties; offset==count applies at the next boundary.
    // Immediate soundboard head plus partitioned tail adds no buffering delay.
    // Hosts query latency_samples(); device and rate-converter delays are separate.
    // Invalid arguments reject the whole call without changing state or output.
    // A full pool steals the quietest stealable voice (stolen_voices()); a
    // rejection is recorded in dropped_notes(). Neither is a process failure.
    static constexpr size_t latency_samples() noexcept { return 0; }
    bool process(float* left, float* right, size_t count,
                 const MidiEvent* events = nullptr, size_t event_count = 0) {
        if (render_mode_ == RenderMode::block || !valid_outputs(left,right,count)
            || (event_count && !events)) return false;
        for (size_t i = 0; i < event_count; ++i) {
            const auto& event = events[i];
            if (event.offset > count || (i && event.offset < events[i-1].offset)
                || !valid_channel(event.channel)) return false;
            switch (event.type) {
                case MidiEvent::Type::note_on:
                    if (!valid_key(event.key) || event.velocity < 0 || event.velocity > 127
                        || (event.velocity && !strike_bank_)) return false;
                    break;
                case MidiEvent::Type::note_off:
                    if (!valid_key(event.key)) return false;
                    break;
                case MidiEvent::Type::pedal:
                    if (event.position >= 0 && !(event.position <= 1)) return false;
                    if (event.position < 0 && !(event.position >= -1)) return false; // NaN
                    break;
                default: return false;
            }
        }
        render_mode_ = RenderMode::stream;
        size_t position = 0, event_index = 0;
        for (;;) {
            while (event_index < event_count && events[event_index].offset == position) {
                const auto& event = events[event_index++];
                switch (event.type) {
                    case MidiEvent::Type::note_on: note_on(event.key,event.velocity,event.seed,event.channel); break;
                    case MidiEvent::Type::note_off: note_off(event.key,event.channel); break;
                    case MidiEvent::Type::pedal:
                        if (event.position >= 0) pedal(event.channel,event.position);
                        else pedal(event.channel,event.down);
                        break;
                }
            }
            if (position == count) break;
            size_t end = event_index < event_count ? events[event_index].offset : count;
            size_t length = std::min(end-position, (size_t)piano::BLOCK_SIZE-stream_position_);
            render_force(stream_position_,length);
            convolve_span(stream_position_, length, true);
            effects_.process(left_.data()+stream_position_,right_.data()+stream_position_,length);
            for (size_t i = 0; i < length; ++i) {
                const double l = left_[stream_position_+i], r = right_[stream_position_+i];
                output_peak_ = std::max({output_peak_,std::abs(l),std::abs(r)});
                left[position+i] = (float)piano::limit_output(l*settings_.gain);
                right[position+i] = (float)piano::limit_output(r*settings_.gain);
            }
            position += length; stream_position_ += length;
            if (stream_position_ == piano::BLOCK_SIZE) stream_position_ = 0;
        }
        return true;
    }
    // Immediate silence/reset, retaining all preparation and reserved storage.
    // Caller synchronizes reset/events/rendering; no internal locks are taken.
    void reset() noexcept {
        for (auto& slot : slots_) {
            slot.voice.stop(); slot.occupied = slot.held = false;
            slot.level = -1; slot.energy = 0; slot.energy_count = 0;
        }
        active_.clear();
        resonance_.reset(); board_.reset(); stream_board_.reset(); effects_.reset();
        for (auto& board : region_boards_) board.reset();
        for (auto& stream : region_streams_) stream.reset();
        std::fill(region_force_.begin(), region_force_.end(), 0.0);
        force_.fill(0); left_.fill(0); right_.fill(0);
        fade_tail_.fill(0); fade_tail_position_ = fade_tail_remaining_ = 0;
        stream_position_ = 0; render_mode_ = RenderMode::unused;
        order_ = dropped_ = stolen_ = stolen_held_total_ = 0; peak_voices_ = 0; output_peak_ = 0;
    }
    size_t active_voices() const noexcept { return active_.size(); }
    size_t peak_voices() const noexcept { return peak_voices_; }
    // Rejected notes never sounded; stolen voices were faded out for a new note.
    uint64_t dropped_notes() const noexcept { return dropped_; }
    uint64_t stolen_voices() const noexcept { return stolen_; }
    uint64_t stolen_held_voices() const noexcept { return stolen_held_total_; }
    double output_peak() const noexcept { return output_peak_; }
    unsigned held_count(int channel, int key) const noexcept {
        return valid_channel(channel) && valid_key(key)
            ? resonance_.held[channel][std::clamp(key,21,108)-21] : 0;
    }
    // Diagnostics borrow buffers until the next render/reset; no mutable state
    // escapes. As with other diagnostics, read on the owning audio thread.
    piano::ConstAudioBuffer bridge_force() const noexcept { return {force_.data(),force_.size()}; }
};
} // namespace instrument
#endif
