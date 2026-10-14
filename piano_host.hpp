// Host-rate adapter for instrument::Engine. No OS, devices, files or plugin SDK.
#ifndef SYNTHPIANO_HOST_HPP
#define SYNTHPIANO_HOST_HPP
#include "instrument.hpp"
#include "piano_resampler.hpp"
#include <limits>
#include <memory>

namespace piano_host {
class Adapter {
    instrument::Engine& engine_; // borrowed; must outlive this adapter
    struct State {
        unsigned rate;
        size_t max_frames,max_events;
        detail::StereoResampler resampler;
        std::vector<float> left,right;
        std::vector<instrument::MidiEvent> events;
        State(unsigned r,size_t frames,size_t event_limit)
            : rate(r),max_frames(frames),max_events(event_limit),
              resampler(r,frames,instrument::Engine::latency_samples()) {
            if (event_limit>65536) throw std::invalid_argument("piano host: event capacity exceeds 65536");
            size_t native=(frames*uint64_t(piano::FS)+r-1)/r;
            left.resize(native); right.resize(native); events.reserve(event_limit);
        }
    };
    std::unique_ptr<State> state_;
public:
    explicit Adapter(instrument::Engine& engine) noexcept : engine_(engine) {}
    Adapter(const Adapter&)=delete;
    Adapter& operator=(const Adapter&)=delete;
    // Setup/stopped audio only. Failed preparation preserves the old state.
    // Build the engine's strike cache separately before sending note-ons.
    void prepare(double rate,size_t maximum_frames,size_t maximum_events=1024) {
        if (!supported_rate(rate)) throw std::invalid_argument("piano host: unsupported sample rate");
        auto replacement=std::make_unique<State>(unsigned(rate),maximum_frames,maximum_events);
        engine_.reset(); state_=std::move(replacement);
    }
    void reset() noexcept {
        engine_.reset();
        if (state_) { state_->resampler.reset(); state_->events.clear(); }
    }
    unsigned sample_rate() const noexcept { return state_ ? state_->rate : 0; }
    size_t latency_samples() const noexcept { return state_ ? state_->resampler.latency_samples() : 0; }
    uint64_t position() const noexcept { return state_ ? state_->resampler.output_position() : 0; }
    // All offsets/counts/latency are in HOST frames. Events at the end boundary
    // and zero-frame calls are supported. Native event placement rounds upward
    // by <1 source sample, preserving caller order even when events coincide.
    // Invalid calls leave engine, timeline and output untouched. Oversized event
    // lists are rejected whole, never partially consumed or silently truncated.
    // One audio owner; do not drive the borrowed engine separately between resets.
    bool process(float* left,float* right,size_t count,
                 const instrument::MidiEvent* events=nullptr,size_t event_count=0) {
        if (!state_) return false;
        auto& s=*state_;
        if (count>s.max_frames || event_count>s.max_events || (event_count && !events)) return false;
        if (count) {
            if (!left || !right) return false;
            std::less<const float*> before;
            if (before(left,right+count) && before(right,left+count)) return false;
        }
        // Bound the lifetime before multiplying absolute sample positions.
        constexpr uint64_t limit=uint64_t(std::numeric_limits<int64_t>::max())/piano::FS-2;
        uint64_t start=s.resampler.output_position();
        if (start>limit-count) return false;
        for (size_t i=0;i<event_count;++i) {
            const auto& e=events[i];
            if (e.offset>count || (i && e.offset<events[i-1].offset) || e.channel<0 || e.channel>15) return false;
            switch (e.type) {
                case instrument::MidiEvent::Type::note_on:
                    if (e.velocity<0 || e.velocity>127 || (e.velocity && !engine_.has_strike_cache())) return false;
                    [[fallthrough]];
                case instrument::MidiEvent::Type::note_off:
                    if (e.key<0 || e.key>127) return false;
                    break;
                case instrument::MidiEvent::Type::pedal: break;
                default: return false;
            }
        }
        size_t native_count=s.resampler.input_needed(count);
        s.events.clear();
        for (size_t i=0;i<event_count;++i) {
            auto event=events[i];
            event.offset=size_t(s.resampler.native_frame(start+event.offset)-s.resampler.input_position());
            s.events.push_back(event);
        }
        if (!engine_.process(s.left.data(),s.right.data(),native_count,s.events.data(),s.events.size())) return false;
        // Native-rate conversion is an exact float copy, with no filter/delay.
        if (count) s.resampler.process(s.left.data(),s.right.data(),native_count,left,right,count);
        return true;
    }
};
} // namespace piano_host
#endif
