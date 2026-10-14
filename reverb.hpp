// App-owned stereo room effect. No dependency on the piano synthesis library.
// Eight delay lines with orthogonal Hadamard feedback, damped recirculation,
// input diffusion, and short stereo early reflections.

#ifndef SYNTHPIANO_REVERB_HPP
#define SYNTHPIANO_REVERB_HPP
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace room {
struct Options {
    double mix = 0.3; // wet level relative to unchanged dry signal; zero bypasses
    double decay_seconds = 1.1; // nominal low-frequency T60
    double predelay_ms = 12.0;
    double damping_hz = 5500.0; // feedback lowpass; lower gives a darker tail
};

inline void validate_options(const Options& o) {
    auto range = [](const char* name, double value, double lo, double hi) {
        if (!std::isfinite(value) || value < lo || value > hi)
            throw std::invalid_argument(std::string("reverb.") + name
                + " must be finite and in [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");
    };
    range("mix", o.mix, 0, 1);
    range("decay_seconds", o.decay_seconds, .2, 4);
    range("predelay_ms", o.predelay_ms, 0, 80);
    range("damping_hz", o.damping_hz, 500, 12000);
}

class StereoReverb {
    struct Delay {
        std::vector<double> samples;
        size_t head = 0;
        void init(size_t length) { samples.assign(length, 0.0); head = 0; }
        void reset() noexcept { std::fill(samples.begin(), samples.end(), 0.0); head = 0; }
        double read() const noexcept { return samples[head]; }
        double tap(size_t lag) const noexcept {
            return samples[(head + samples.size() - lag) % samples.size()];
        }
        void push(double value) noexcept {
            samples[head] = std::abs(value) < 1e-30 ? 0.0 : value;
            if (++head == samples.size()) head = 0;
        }
        double process(double value) noexcept {
            if (samples.empty()) return value; // exact zero predelay
            double delayed = read(); push(value); return delayed;
        }
        double allpass(double value) noexcept {
            constexpr double diffusion = .6;
            double output = read() - diffusion * value;
            push(value + diffusion * output);
            return output;
        }
    };
    Options options_;
    double sample_rate_, lowpass_, highpass_;
    std::array<Delay, 8> lines_;
    std::array<Delay, 2> predelay_, early_;
    std::array<Delay, 4> diffusion_;
    std::array<double, 8> feedback_{}, filtered_{};
    std::array<double, 2> previous_input_{}, highpass_state_{};
    std::array<size_t, 6> taps_{};

public:
    // Construct off the audio thread. Parameters are fixed for this instance;
    // callers create a fresh effect to change room settings. No copying/moving
    // of a live effect is supported. Valid process/reset calls never allocate.
    explicit StereoReverb(double sample_rate, const Options& options = {})
        : options_(options), sample_rate_(sample_rate), lowpass_(0), highpass_(0) {
        validate_options(options);
        if (!std::isfinite(sample_rate) || sample_rate < 8000 || sample_rate > 192000)
            throw std::invalid_argument("reverb sample rate must be finite and in [8000,192000]");
        if (options.mix == 0.0) return;
        constexpr double pi = 3.14159265358979323846;
        lowpass_ = std::exp(-2 * pi * options.damping_hz / sample_rate);
        highpass_ = std::exp(-2 * pi * 150.0 / sample_rate);
        auto scaled = [&](double samples_at_44100) {
            return std::max<size_t>(1, (size_t)std::lround(samples_at_44100 * sample_rate / 44100.0));
        };
        const int lengths[] = {1493,1601,1747,1867,1999,2131,2281,2423};
        for (size_t i = 0; i < lines_.size(); ++i) {
            size_t length = scaled(lengths[i]);
            lines_[i].init(length);
            feedback_[i] = std::pow(10.0, -3.0 * length / (sample_rate * options.decay_seconds));
        }
        for (auto& delay : predelay_) delay.init((size_t)std::lround(options.predelay_ms * sample_rate / 1000.0));
        const int diffusion_lengths[] = {191,337,211,373};
        for (size_t i = 0; i < diffusion_.size(); ++i) diffusion_[i].init(scaled(diffusion_lengths[i]));
        const double tap_ms[] = {7,11,17,23,29,31};
        for (size_t i = 0; i < taps_.size(); ++i)
            taps_[i] = std::max<size_t>(1, (size_t)std::lround(tap_ms[i] * sample_rate / 1000.0));
        for (auto& delay : early_) delay.init(taps_.back() + 1);
    }
    StereoReverb(const StereoReverb&) = delete;
    StereoReverb& operator=(const StereoReverb&) = delete;
    StereoReverb(StereoReverb&&) = delete;
    StereoReverb& operator=(StereoReverb&&) = delete;

    const Options& options() const noexcept { return options_; }
    // Conservative tail after the final input: two nominal T60s plus predelay
    // and a small allowance for input diffusion/filter delay. Zero in bypass.
    size_t tail_samples() const noexcept {
        return options_.mix == 0 ? 0 : (size_t)std::ceil(sample_rate_
            * (2.0 * options_.decay_seconds + options_.predelay_ms / 1000.0 + .15));
    }
    void reset() noexcept {
        for (auto& d : lines_) d.reset();
        for (auto& d : predelay_) d.reset();
        for (auto& d : early_) d.reset();
        for (auto& d : diffusion_) d.reset();
        filtered_.fill(0); previous_input_.fill(0); highpass_state_.fill(0);
    }
    // In-place stereo dry + wet. Channels must not overlap; empty calls are
    // no-ops. The caller supplies real storage containing finite audio samples.
    void process(double* left, double* right, size_t count) {
        if (count > (size_t)std::numeric_limits<int>::max())
            throw std::length_error("reverb buffer is too long");
        if (!count) return;
        if (!left || !right) throw std::invalid_argument("reverb requires two non-null channels");
        std::less<const double*> before;
        if (before(left, right + count) && before(right, left + count))
            throw std::invalid_argument("reverb channels must not overlap");
        if (options_.mix == 0) return;
        constexpr double norm = .3535533905932737622; // 1/sqrt(8)
        constexpr double input_right[] = {1,1,1,1,-1,-1,-1,-1};
        constexpr double output_left[] = {1,1,-1,-1,1,1,-1,-1};
        constexpr double output_right[] = {1,-1,1,-1,1,-1,1,-1};
        for (size_t n = 0; n < count; ++n) {
            double input[2] = {left[n], right[n]};
            for (size_t c = 0; c < 2; ++c) {
                double value = input[c] - previous_input_[c] + highpass_ * highpass_state_[c];
                previous_input_[c] = input[c];
                highpass_state_[c] = std::abs(value) < 1e-30 ? 0.0 : value;
                input[c] = predelay_[c].process(highpass_state_[c]);
            }
            double early_l = .55 * early_[0].tap(taps_[0]) + .30 * early_[1].tap(taps_[2])
                           - .15 * early_[0].tap(taps_[4]);
            double early_r = .55 * early_[1].tap(taps_[1]) + .30 * early_[0].tap(taps_[3])
                           - .15 * early_[1].tap(taps_[5]);
            early_[0].push(input[0]); early_[1].push(input[1]);
            double l = diffusion_[1].allpass(diffusion_[0].allpass(input[0]));
            double r = diffusion_[3].allpass(diffusion_[2].allpass(input[1]));
            double mixed[8], late_l = 0, late_r = 0;
            for (size_t i = 0; i < lines_.size(); ++i) {
                double delayed = lines_[i].read();
                late_l += output_left[i] * delayed;
                late_r += output_right[i] * delayed;
                filtered_[i] = (1.0-lowpass_) * delayed + lowpass_ * filtered_[i];
                if (std::abs(filtered_[i]) < 1e-30) filtered_[i] = 0;
                mixed[i] = filtered_[i];
            }
            // Normalised Walsh-Hadamard transform: orthogonal feedback mixing.
            for (size_t width = 1; width < 8; width *= 2)
                for (size_t start = 0; start < 8; start += 2 * width)
                    for (size_t j = 0; j < width; ++j) {
                        double a = mixed[start+j], b = mixed[start+j+width];
                        mixed[start+j] = a+b; mixed[start+j+width] = a-b;
                    }
            for (size_t i = 0; i < lines_.size(); ++i)
                lines_[i].push(norm * (l + input_right[i] * r + feedback_[i] * mixed[i]));
            left[n] += options_.mix * (.45 * early_l + .70 * norm * late_l);
            right[n] += options_.mix * (.45 * early_r + .70 * norm * late_r);
        }
    }
};
} // namespace room
#endif
