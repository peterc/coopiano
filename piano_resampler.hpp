// Internal, platform-independent stereo rate conversion for piano_host.hpp.
#ifndef SYNTHPIANO_RESAMPLER_HPP
#define SYNTHPIANO_RESAMPLER_HPP
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace piano_host {
inline bool supported_rate(double rate) noexcept {
    return rate==44100 || rate==48000 || rate==88200 || rate==96000;
}
namespace detail {
// Rational phases avoid drift and phase-table interpolation. A 257-tap
// Blackman-Harris-windowed sinc preserves the audio band with image rejection.
// The phase origin includes fractional padding so total latency is an exact
// integer number of host frames, including the supplied engine latency.
class StereoResampler {
public:
    static constexpr unsigned source_rate=44100, half=128, taps=2*half+1;
private:
    unsigned rate_, phases_, step_, phase_=0;
    size_t latency_, max_frames_;
    int64_t center_=0;
    uint64_t written_=0, output_=0;
    std::vector<std::array<double,taps>> filters_;
    std::vector<float> left_,right_;
    size_t engine_delay_;
public:
    StereoResampler(unsigned rate, size_t max_frames, size_t engine_delay)
        : rate_(rate),phases_(0),step_(0),latency_(engine_delay),max_frames_(max_frames),engine_delay_(engine_delay) {
        if (!supported_rate(rate) || !max_frames || max_frames>8192 || engine_delay>8192)
            throw std::invalid_argument("piano host: unsupported rate or buffer size");
        if (rate==source_rate) return;
        unsigned gcd=std::gcd(rate,source_rate);
        phases_=rate/gcd; step_=source_rate/gcd;
        latency_=((engine_delay+half)*rate+source_rate-1)/source_rate;
        size_t capacity=(max_frames*source_rate+rate-1)/rate+2*half+4;
        left_.resize(capacity); right_.resize(capacity); filters_.resize(phases_);
        constexpr double pi=3.141592653589793238462643383279502884;
        constexpr double cutoff=.475;
        for (unsigned phase=0;phase<phases_;++phase) {
            double fraction=double(phase)/phases_,sum=0;
            for (unsigned tap=0;tap<taps;++tap) {
                double x=double(tap)-half-fraction, value=0;
                if (std::abs(x)<=half) {
                    double window=.35875+.48829*std::cos(pi*x/half)
                                  +.14128*std::cos(2*pi*x/half)+.01168*std::cos(3*pi*x/half);
                    value=(x==0 ? 2*cutoff : std::sin(2*pi*cutoff*x)/(pi*x))*window;
                }
                filters_[phase][tap]=value; sum+=value;
            }
            for (double& value:filters_[phase]) value/=sum;
        }
        reset();
    }
    void reset() noexcept {
        written_=output_=0;
        std::fill(left_.begin(),left_.end(),0); std::fill(right_.begin(),right_.end(),0);
        if (rate_==source_rate) return;
        int64_t numerator=int64_t(engine_delay_)*rate_-int64_t(latency_)*source_rate;
        center_=numerator/int64_t(rate_);
        int64_t remainder=numerator%int64_t(rate_);
        if (remainder<0) { --center_; remainder+=rate_; }
        phase_=unsigned(remainder)/(rate_/phases_);
    }
    size_t latency_samples() const noexcept { return latency_; }
    // Caller bounds lifetime to INT64_MAX/source_rate, before any multiplication.
    uint64_t native_frame(uint64_t host_frame) const noexcept {
        return (host_frame*source_rate+rate_-1)/rate_;
    }
    size_t input_needed(size_t output_count) const noexcept {
        return size_t(native_frame(output_+output_count)-written_);
    }
    uint64_t input_position() const noexcept { return written_; }
    uint64_t output_position() const noexcept { return output_; }
    // The adapter validates all arguments before rendering the engine. Source
    // arrays contain exactly input_needed(count) frames and are private scratch.
    void process(const float* in_left,const float* in_right,size_t input_count,
                 float* out_left,float* out_right,size_t count) noexcept {
        assert(count<=max_frames_ && input_count==input_needed(count));
        if (rate_==source_rate) {
            std::copy_n(in_left,count,out_left); std::copy_n(in_right,count,out_right);
            written_+=input_count; output_+=count; return;
        }
        for (size_t i=0;i<input_count;++i) {
            size_t index=size_t(written_%left_.size());
            left_[index]=in_left[i]; right_[index]=in_right[i]; ++written_;
        }
        for (size_t frame=0;frame<count;++frame) {
            double l=0,r=0;
            int64_t first=center_-half;
            unsigned tap=first<0 ? unsigned(std::min<int64_t>(-first,taps)) : 0;
            int64_t sample=first+tap;
            if (tap<taps) {
                assert(sample>=0 && uint64_t(sample)<written_);
                assert(written_-uint64_t(sample)<=left_.size());
                size_t index=size_t(uint64_t(sample)%left_.size());
                for (;tap<taps;++tap) {
                    assert(uint64_t(first+tap)<written_);
                    l+=filters_[phase_][tap]*left_[index];
                    r+=filters_[phase_][tap]*right_[index];
                    if (++index==left_.size()) index=0;
                }
            }
            out_left[frame]=float(l); out_right[frame]=float(r);
            phase_+=step_; center_+=phase_/phases_; phase_%=phases_;
        }
        output_+=count;
    }
};
} // namespace detail
} // namespace piano_host
#endif
