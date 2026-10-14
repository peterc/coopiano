// Immediate soundboard convolution for arbitrary streaming buffers.
// The first 256 taps are a direct FIR. The remaining IR is shifted back by
// 256 samples and convolved in blocks, ready before those samples are needed.
// The full impulse response is retained; no future input or MIDI is required.
#ifndef SYNTHPIANO_STREAMING_SOUNDBOARD_HPP
#define SYNTHPIANO_STREAMING_SOUNDBOARD_HPP
#include "piano.hpp"

namespace instrument {
namespace detail {
class StreamingSoundboard {
    static constexpr size_t block = piano::BLOCK_SIZE;
    piano::PartitionedConvolver tail_;
    std::array<double,block> head_left_{}, head_right_{}, input_{}, tail_left_{}, tail_right_{};
    std::array<double,2*block> history_{}; // mirrored for contiguous FIR reads
    size_t head_size_ = 0, position_ = 0, cursor_ = 0;

    void swap(StreamingSoundboard& other) noexcept {
        std::swap(tail_,other.tail_);
        head_left_.swap(other.head_left_); head_right_.swap(other.head_right_);
        input_.swap(other.input_); tail_left_.swap(other.tail_left_); tail_right_.swap(other.tail_right_);
        history_.swap(other.history_);
        std::swap(head_size_,other.head_size_); std::swap(position_,other.position_); std::swap(cursor_,other.cursor_);
    }
    static bool overlaps(const double* a, const double* b, size_t count) noexcept {
        if (!count) return false;
        std::less<const double*> before;
        return before(a,b+count) && before(b,a+count);
    }
public:
    StreamingSoundboard() = default;
    StreamingSoundboard(const StreamingSoundboard&) = delete;
    StreamingSoundboard& operator=(const StreamingSoundboard&) = delete;
    StreamingSoundboard(StreamingSoundboard&& other) noexcept { swap(other); }
    StreamingSoundboard& operator=(StreamingSoundboard&& other) noexcept {
        if (this != &other) { StreamingSoundboard replacement(std::move(other)); swap(replacement); }
        return *this;
    }
    bool ready() const noexcept { return head_size_ != 0; }
    // Setup only. Failed initialization preserves the existing IR and history.
    void init(const std::vector<double>& left, const std::vector<double>& right) {
        piano::detail::validate_ir(left,right);
        StreamingSoundboard replacement;
        replacement.head_size_ = std::min(block,std::max(left.size(),right.size()));
        std::copy_n(left.begin(),std::min(block,left.size()),replacement.head_left_.begin());
        std::copy_n(right.begin(),std::min(block,right.size()),replacement.head_right_.begin());
        if (std::max(left.size(),right.size()) > block) {
            std::vector<double> l(left.begin()+std::min(block,left.size()),left.end());
            std::vector<double> r(right.begin()+std::min(block,right.size()),right.end());
            replacement.tail_.init(l,r);
        }
        swap(replacement);
    }
    // No allocation or deallocation during processing/reset. Input can be exactly
    // either output; partial overlap and overlapping output channels are rejected.
    // All validation precedes mutation. Zero-length views may have null pointers.
    void process(piano::ConstAudioBuffer input, piano::AudioBuffer left, piano::AudioBuffer right) {
        if (!ready()) throw std::logic_error("instrument: soundboard is not initialized");
        if (input.size != left.size || input.size != right.size)
            throw std::invalid_argument("instrument: soundboard buffer sizes differ");
        piano::detail::validate_buffer(input.data,input.size);
        piano::detail::validate_buffer(left.data,left.size);
        piano::detail::validate_buffer(right.data,right.size);
        if (overlaps(left.data,right.data,input.size)
            || (input.data != left.data && overlaps(input.data,left.data,input.size))
            || (input.data != right.data && overlaps(input.data,right.data,input.size)))
            throw std::invalid_argument("instrument: soundboard buffers overlap");
        for (size_t i=0; i<input.size; ++i) {
            const double x = input.data[i];
            input_[position_] = x;
            cursor_ = (cursor_+block-1)%block;
            history_[cursor_] = history_[cursor_+block] = x;
            const double* history = history_.data()+cursor_;
            double l = 0, r = 0;
            for (size_t tap=0; tap<head_size_; ++tap) {
                l += history[tap]*head_left_[tap];
                r += history[tap]*head_right_[tap];
            }
            left.data[i] = l+tail_left_[position_];
            right.data[i] = r+tail_right_[position_];
            if (++position_ == block) {
                if (tail_.ready()) tail_.process({input_.data(),block},
                    {tail_left_.data(),block},{tail_right_.data(),block});
                position_ = 0;
            }
        }
    }
    void reset() noexcept {
        tail_.reset(); input_.fill(0); tail_left_.fill(0); tail_right_.fill(0);
        history_.fill(0); position_ = cursor_ = 0;
    }
};
} // namespace detail
} // namespace instrument
#endif
