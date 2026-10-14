// piano.hpp — single-header C++17 piano synthesis library.
// Supported API: Voicing, validate_voicing, Key, Options, prepare_note/Voice, render_bridge_force,
// prepare_strike/PreparedNote, VoiceCapacity/voice_capacity, ContactResponse/StrikeWorkspace,
// SympatheticResonance, make_soundboard_ir, convolve_stereo,
// PartitionedConvolver, and output helpers.
// All sample buffers contain floating-point model amplitudes at FS (44,100 Hz).
// The host owns gain, PCM encoding, MIDI/pedal scheduling, and audio devices.

#ifndef SYNTHPIANO_PIANO_HPP
#define SYNTHPIANO_PIANO_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace piano {

inline constexpr int BLOCK_SIZE = 256;
inline constexpr int FS = 44100;

// Views carry lengths without allocating or taking ownership.
struct AudioBuffer { double* data; size_t size; };
struct ConstAudioBuffer { const double* data; size_t size; };

struct Options {
    double damper_scale = 1.0;  // positive finite release-time multiplier for strings and phantoms
    double knock_gain = 1.0;   // finite, nonnegative
    bool knock = true;
    bool scrape = true;
    bool damper_noise = true;
    bool soften_longitudinal_highs = true; // standard voicing; false restores the unsoftened layer
};

// Instrument voicing, resolved during key/voice/board preparation. Scale
// controls multiply the existing pitch-dependent curves; they do not replace
// the piano's register variation. Default values preserve the calibrated sound.
struct HammerVoicing {
    double stiffness_scale = 0.90;
    double mass_scale = 1.3;
    double felt_width_scale = 1.4;
    double maximum_velocity_mps = 6.0;
    double velocity_exponent = 1.8;
};
struct StringVoicing {
    double decay_scale = 1.10;
    double partial_loss_scale = 1.0;
    double inharmonicity_scale = 0.85;
    double unison_detune_scale = 0.35;
    double fast_decay_scale = 0.95;
    double horizontal_ratio = 0.4;
    double longitudinal_gain = 0.22;
    double bridge_coupling = 1.0;   // string-soundboard admittance coupling: 0..3
    double treble_decay_scale = 0.38; // natural decay: blends C6..F#6, full above
    double treble_unison_detune = 0.875; // minimum unison detune: blends G5..C6, full above
};
struct ToneVoicing {
    double fundamental_boost_db = 0.0;
    double lower_overtone_cut_db = 0.0;
    double bridge_curve_scale = 1.0;
    double mid_gain_db = -1.5;
    double mid_frequency_hz = 1200.0;
    double mid_width_octaves = 0.9;
};
struct SoundboardVoicing {
    double bass_corner_hz = 70.0;
    double treble_corner_hz = 3800.0;
    double treble_slope = 1.8;
    double decay_scale = 1.10;
    double ripple_fraction = 0.25;
    double low_mode_density = 1.0; // 0 = sparse low modes, 1 = dense; intermediate spacing
    // Spatial imaging by bridge position. The bridge is divided into a few
    // overlapping regions, each with its own stereo soundboard response.
    // spread pans regions across the stereo field (0 = all centred, 1 = full
    // width, bass to the left as the player hears it). position_depth makes
    // each region excite the board's modes differently, as a string's place
    // on the bridge does (0 = identical responses, so only panning remains).
    // Both zero restores the single shared response.
    double spread = 0.7;
    double position_depth = 0.5;
};
struct ResonanceOptions {
    double gain = 0.18; // zero is exact bypass
    double maximum_decay_seconds = 12.0; // open-string T60 cap
    double damper_decay_seconds = 0.18;  // closed-string T60
    double transition_seconds = 0.04;
    // Fraction of a played key's own bridge force that drives that key's own
    // sympathetic resonators. The played strings already model the string, and
    // a nominal-pitch copy of them beats slowly against the detuned unisons
    // (a faint bowing-like wow, about 24-32 dB below the music), so 0 is the
    // physical choice: a key excites every other key while its own resonators
    // respond only to the rest of the instrument. 1 restores the old sound.
    double own_drive = 0.0;
};
struct Voicing {
    HammerVoicing hammer;
    StringVoicing strings;
    ToneVoicing tone;
    SoundboardVoicing soundboard;
    ResonanceOptions resonance;
};

inline void validate_voicing(const Voicing& v) {
    auto range = [](const char* name, double value, double lo, double hi) {
        if (!std::isfinite(value) || value < lo || value > hi)
            throw std::invalid_argument(std::string("piano: ") + name
                + " must be finite and in [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");
    };
    range("hammer.stiffness_scale", v.hammer.stiffness_scale, .25, 4);
    range("hammer.mass_scale", v.hammer.mass_scale, .5, 2);
    range("hammer.felt_width_scale", v.hammer.felt_width_scale, .5, 2);
    range("hammer.maximum_velocity_mps", v.hammer.maximum_velocity_mps, 2, 8);
    range("hammer.velocity_exponent", v.hammer.velocity_exponent, .75, 2.5);
    range("strings.decay_scale", v.strings.decay_scale, .25, 2);
    range("strings.treble_decay_scale", v.strings.treble_decay_scale, .1, 2);
    range("strings.treble_unison_detune", v.strings.treble_unison_detune, 0, 3);
    range("strings.partial_loss_scale", v.strings.partial_loss_scale, .25, 4);
    range("strings.bridge_coupling", v.strings.bridge_coupling, 0, 3);
    range("strings.inharmonicity_scale", v.strings.inharmonicity_scale, 0, 3);
    range("strings.unison_detune_scale", v.strings.unison_detune_scale, 0, 3);
    range("strings.fast_decay_scale", v.strings.fast_decay_scale, .75, 2);
    range("strings.horizontal_ratio", v.strings.horizontal_ratio, 0, 1);
    range("strings.longitudinal_gain", v.strings.longitudinal_gain, 0, .5);
    range("tone.fundamental_boost_db", v.tone.fundamental_boost_db, -6, 12);
    range("tone.lower_overtone_cut_db", v.tone.lower_overtone_cut_db, 0, 12);
    range("tone.bridge_curve_scale", v.tone.bridge_curve_scale, 0, 2);
    range("tone.mid_gain_db", v.tone.mid_gain_db, -9, 9);
    range("tone.mid_frequency_hz", v.tone.mid_frequency_hz, 200, 5000);
    range("tone.mid_width_octaves", v.tone.mid_width_octaves, .2, 2);
    range("soundboard.bass_corner_hz", v.soundboard.bass_corner_hz, 30, 250);
    range("soundboard.treble_corner_hz", v.soundboard.treble_corner_hz, 1500, 10000);
    range("soundboard.treble_slope", v.soundboard.treble_slope, .5, 3);
    range("soundboard.decay_scale", v.soundboard.decay_scale, .25, 2);
    range("soundboard.ripple_fraction", v.soundboard.ripple_fraction, 0, 1);
    range("soundboard.low_mode_density", v.soundboard.low_mode_density, 0, 1);
    range("resonance.gain", v.resonance.gain, 0, 1);
    range("resonance.maximum_decay_seconds", v.resonance.maximum_decay_seconds, 1, 30);
    range("resonance.damper_decay_seconds", v.resonance.damper_decay_seconds, .03, 2);
    range("resonance.transition_seconds", v.resonance.transition_seconds, .001, .2);
    range("resonance.own_drive", v.resonance.own_drive, 0, 1);
    range("soundboard.spread", v.soundboard.spread, 0, 1);
    range("soundboard.position_depth", v.soundboard.position_depth, 0, 1);
}

enum class ContactStatus { not_run, separated, timed_out };

struct NoteDebug {
    double contact_ms, peak_force, v_h;
    int n_phasors;
    ContactStatus contact_status = ContactStatus::not_run;
};

namespace detail {
inline constexpr double PI = 3.14159265358979323846;
inline constexpr double F_A4 = 440.0;
inline constexpr double F_LIMIT = 17500.0;

// Railsback stretch: bass tuned flat, treble sharp. Mild through the middle
// (real curves stay within a few cents from C2 to C6), steep at the ends.
inline double midi_to_freq(double m) {
    double x = (m - 69.0) / (m < 69.0 ? 48.0 : 39.0);
    double dev_cents = 14.0 * x * x * x + 16.0 * std::pow(x, 7.0);
    return F_A4 * std::pow(2.0, (m - 69.0 + dev_cents / 100.0) / 12.0);
}

// ---------------------------------------------------------------------------
// Per-key physical description

struct KeyParams {
    double f0;        // fundamental (Hz)
    double L;         // speaking length (m)
    double T;         // tension (N)
    double mu;        // linear density (kg/m)
    double m_s;       // string mass (kg)
    double B;         // inharmonicity coefficient
    double alpha;     // strike position fraction
    double T60_1;     // fundamental's late-stage T60 (s)
    double c2;        // frequency-dependent damping coefficient
    double m_h;       // hammer mass (kg)
    double K;         // felt stiffness (N/m^p)
    double p;         // felt exponent
    double fast_ratio; // vertical (fast-stage) sigma multiplier
    double det_scale;  // unison detune scale (cents shrink with register)
    double f_long1;   // longitudinal fundamental of the string (Hz)
    double sigma_h;   // hammer felt contact half-width on the string (m)
    double fundamental_gain; // empirical radiation support for the first mode
    double strike_level_gain; // register balance after upper-string excitation correction
    double low_mid_depth_db; // lower-overtone attenuation, before soundboard
    int    n_strings;
    bool   has_damper;
    HammerVoicing hammer;
    StringVoicing strings;
    ToneVoicing tone;
};

// piecewise log-linear interpolation over MIDI note number
inline double interp_log(double m, const std::vector<std::pair<double,double>>& pts) {
    if (m <= pts.front().first) return pts.front().second;
    if (m >= pts.back().first)  return pts.back().second;
    for (size_t i = 1; i < pts.size(); ++i) {
        if (m <= pts[i].first) {
            double t = (m - pts[i-1].first) / (pts[i].first - pts[i-1].first);
            return pts[i-1].second * std::pow(pts[i].second / pts[i-1].second, t);
        }
    }
    return pts.back().second;
}

inline KeyParams make_key_params(double midi, const Voicing& voicing = {}) {
    KeyParams k;
    k.hammer = voicing.hammer;
    k.strings = voicing.strings;
    k.tone = voicing.tone;
    k.f0 = midi_to_freq(midi);

    // Speaking length: concert-grand-like scale.
    k.L = interp_log(midi, {{21, 1.92}, {36, 1.30}, {48, 0.92}, {60, 0.65},
                            {69, 0.41}, {84, 0.185}, {108, 0.052}});
    k.T  = (midi < 40) ? 760.0 : 700.0;
    double c = 2.0 * k.L * k.f0;            // ideal transverse wave speed
    k.mu  = k.T / (c * c);
    k.m_s = k.mu * k.L;

    // Inharmonicity: treble rises with f; wound bass strings add a floor.
    k.B = 7.0e-4 * std::pow(k.f0 / F_A4, 1.4)
        + 1.0e-4 * std::clamp(55.0 / k.f0, 0.0, 2.2);
    k.B *= voicing.strings.inharmonicity_scale;

    // Strike point: ~1/8 of length in bass shrinking towards treble.
    k.alpha = interp_log(midi, {{21, 0.125}, {60, 0.118}, {84, 0.10}, {108, 0.065}});

    // Late-stage decay of the fundamental.
    k.T60_1 = interp_log(midi, {{21, 58.0}, {45, 48.0}, {57, 35.0}, {69, 24.0},
                                {81, 14.0}, {84, 10.0}, {96, 4.2}, {108, 1.2}});
    // sigma_n = sigma_1 + c2*(f_n^2 - f_1^2). Register-dependent: wound bass
    // strings lose their upper partials several times faster than plain
    // middle strings (their sustain carried 4-8 dB too much at 0.6-2.4 kHz
    // against the reference while the middle carried 4-14 dB too little).
    k.c2 = 1.0e-7 * interp_log(midi, {{21, 3.0}, {33, 2.5}, {44, 1.6}, {52, 1.0},
                                       {60, 0.6}, {72, 0.5}, {108, 0.5}});
    // Shorter natural treble ringing, without an EQ change or moving the
    // last damper. Fade in over C6..F#6, with flat slopes at both ends.
    double treble_mix = std::clamp((midi - 84.0) / 6.0, 0.0, 1.0);
    treble_mix *= treble_mix * (3.0 - 2.0 * treble_mix);
    double treble_decay = midi >= 90.0 ? voicing.strings.treble_decay_scale
        : 1.0 + (voicing.strings.treble_decay_scale - 1.0) * treble_mix;
    k.T60_1 *= treble_decay;
    k.T60_1 *= voicing.strings.decay_scale;
    k.c2 *= voicing.strings.partial_loss_scale;

    // Hammer: heavy/soft in the bass, light/hard in the treble.
    double t = std::clamp((midi - 21.0) / 87.0, 0.0, 1.0);
    k.m_h = (11.0 - 7.0 * t) * 1e-3;
    k.p   = 2.85 + 0.85 * t;
    // Felt stiffness scaled so contact time runs ~3ms (bass) -> ~0.5ms (top).
    k.K   = std::pow(10.0, 10.70 + 2.15 * t);
    // Register calibration of the nonlinear felt law. The former K/p curves
    // left upper strings under the hammer for several fundamental cycles,
    // suppressing their pitched attack, especially at soft velocities.
    // Leave C5 and below unchanged; raise effective stiffness progressively
    // to C8. Ease in over three semitones to avoid a slope discontinuity.
    double treble_hammer = std::clamp(midi - 72.0, 0.0, 36.0);
    treble_hammer = treble_hammer < 3.0 ? treble_hammer * treble_hammer / 6.0
                                      : treble_hammer - 1.5;
    k.K *= std::pow(10.0, 4.0 * treble_hammer / 34.5);
    // Experimental gentler upper attack: retain the excitation correction,
    // but reduce effective stiffness to 40% by C7. C6/lower stay unchanged.
    double gentle_treble = std::clamp((midi - 84.0) / 12.0, 0.0, 1.0);
    gentle_treble *= gentle_treble * (3.0 - 2.0 * gentle_treble);
    k.K *= std::pow(10.0, -0.4 * gentle_treble);
    // Recovering pitched excitation is not an equal-loudness target. Retain a
    // falling register level so bell-like upper notes do not dominate music:
    // Base: C5 0 dB, C6 -3 dB, C7 -9 dB, C8 -15 dB. The quadratic onset joins the
    // linear upper slope continuously at C6; this gain never changes contact.
    const double strike_octaves = std::clamp((midi - 72.0) / 12.0, 0.0, 3.0);
    const double strike_db = strike_octaves <= 1.0 ? -3.0 * strike_octaves * strike_octaves
                                                  : 3.0 - 6.0 * strike_octaves;
    // Small listening trim: another 1 dB at C7 and 2 dB at C8, smoothly
    // introduced above C6 without changing the hammer or decay calibration.
    double top_trim = std::clamp((midi - 84.0) / 24.0, 0.0, 1.0);
    top_trim *= top_trim * (3.0 - 2.0 * top_trim);
    // Approximately preserve the accepted mf fundamental level. This is not
    // per-note normalization: softer strikes retain their natural dynamics.
    k.strike_level_gain = std::pow(10.0, (strike_db - 2.0 * top_trim + 1.0 * gentle_treble) / 20.0);
    k.m_h *= voicing.hammer.mass_scale;
    k.K *= voicing.hammer.stiffness_scale;

    k.fast_ratio = interp_log(midi, {{21, 7.0}, {69, 6.5}, {84, 3.0},
                                     {96, 2.0}, {108, 1.5}});
    k.det_scale  = std::clamp(620.0 / k.f0, 0.30, 1.0);
    k.fast_ratio = std::max(1.0, k.fast_ratio * voicing.strings.fast_decay_scale);
    k.det_scale *= voicing.strings.unison_detune_scale;
    // Treble unisons kept a fraction of a cent apart barely beat within a
    // note's life, so high notes faded as one smooth tone. Raise the detune
    // to a floor over G5..C6 (default: outer strings about 1.6 cents apart),
    // giving the quick drop and hanging, wavering tail of a real treble note.
    double treble_unison = std::clamp((midi - 79.0) / 5.0, 0.0, 1.0);
    treble_unison *= treble_unison * (3.0 - 2.0 * treble_unison);
    k.det_scale += treble_unison * std::max(0.0, voicing.strings.treble_unison_detune - k.det_scale);

    // Longitudinal wave speed: ~5100 m/s in plain steel; copper winding adds
    // mass without longitudinal stiffness, slowing the wave in the bass.
    // Calibrated to the measured A1 phantom resonance at ~1442 Hz.
    double wound = interp_log(midi, {{21, 0.75}, {33, 0.79}, {41, 0.86}, {47, 1.0}, {108, 1.0}});
    k.f_long1 = 5100.0 * wound / (2.0 * k.L);

    // Finite hammer width: felt touches centimetres of string, not a point.
    // Spatial lowpass on mode coupling (original-research-notes.md 4.3); without it the strike
    // puts a needle-sharp corner wave into the bridge at every onset.
    k.sigma_h = interp_log(midi, {{21, 0.013}, {60, 0.010}, {84, 0.007}, {108, 0.004}});
    k.sigma_h *= voicing.hammer.felt_width_scale;

    // More foundation in the lower middle, where held chords still leaned
    // heavily on their overtones. This is output voicing, not extra string
    // excitation. Smooth octave-wide fades preserve the deep bass and treble.
    auto smoothstep = [](double x) {
        x = std::clamp(x, 0.0, 1.0);
        return x * x * (3.0 - 2.0 * x);
    };
    double body_register = smoothstep((midi - 33.0) / 12.0)
                         * smoothstep((69.0 - midi) / 12.0);
    k.fundamental_gain = std::pow(10.0, voicing.tone.fundamental_boost_db * body_register / 20.0);
    k.low_mid_depth_db = voicing.tone.lower_overtone_cut_db * body_register;

    k.n_strings  = (midi < 31) ? 1 : (midi < 41) ? 2 : 3;
    k.has_damper = k.f0 < 1400.0;
    return k;
}

// ---------------------------------------------------------------------------
// Hammer-string contact: oversampled symplectic integration of all modes of
// all unison strings against a hysteretic power-law felt (original-research-notes.md §5).

struct StringModes {
    int n_modes = 0;
    std::vector<double> w;       // angular freq
    std::vector<double> wh;      // second polarization's angular freq
    std::vector<double> sig;     // late-stage sigma
    std::vector<double> g;       // force coupling (2/m_s)*sin(n pi alpha)
    std::vector<double> s_a;     // sin(n pi alpha), readout at strike point
    std::vector<double> w_br;    // bridge force weight  T*(n pi/L)*(-1)^n
    std::vector<double> q, qd;   // state
};

// Empirical register-dependent bridge voicing.
inline double bridge_trans(const KeyParams& k, double f) {
    static const double FQ[] = {250, 400, 600, 900, 1250, 1700, 2200, 3000, 4000};
    static const double DB[] = {0,  +1,  +1,  +2,   +2,  +8,   +3,    0,    0};
    double db = 0.0;
    if (f > FQ[0] && f < FQ[8]) {
        int i = 0;
        while (f > FQ[i + 1]) ++i;
        double t = std::log(f / FQ[i]) / std::log(FQ[i + 1] / FQ[i]);
        db = DB[i] + t * (DB[i + 1] - DB[i]);
    }
    // full strength up to A4, fading to nothing by C5
    double s = std::clamp(std::log(523.25 / k.f0) / std::log(523.25 / 440.0), 0.0, 1.0);
    // Bass body and restraint, accepted by listening, fading out by G#3.
    // This is an empirical radiation contour, not a change in string loss.
    double bass = std::clamp(std::log2(midi_to_freq(56.0) / k.f0) / (16.0 / 12.0), 0.0, 1.0);
    bass *= bass * (3.0 - 2.0 * bass);
    double low = std::clamp(std::log2(400.0 / f) / std::log2(400.0 / 150.0), 0.0, 1.0);
    low *= low * (3.0 - 2.0 * low);
    double mid = std::exp(-0.5 * std::pow(std::log2(f / 850.0) / 0.7, 2.0));
    double shaped_db = (s * db * (1.0 - bass) + bass * (6.0 * low - 4.0 * mid))
                     * k.tone.bridge_curve_scale;
    if (k.tone.mid_gain_db != 0.0)
        shaped_db += k.tone.mid_gain_db * std::exp(-0.5 * std::pow(
            std::log2(f / k.tone.mid_frequency_hz) / k.tone.mid_width_octaves, 2.0));
    return std::pow(10.0, shaped_db / 20.0) * k.strike_level_gain;
}

inline double lower_overtone_gain(const KeyParams& k, double f) {
    // Broad, smooth restraint of the lower overtones. The first transverse
    // mode bypasses this curve, even when its frequency lies in this band;
    // higher notes therefore keep their own fundamental body. Preserve the
    // upper detail above 2 kHz. This is empirical output voicing.
    if (k.low_mid_depth_db <= 0.0 || f <= 160.0 || f >= 2000.0) return 1.0;
    auto smoothstep = [](double x) {
        x = std::clamp(x, 0.0, 1.0);
        return x * x * (3.0 - 2.0 * x);
    };
    double rise = smoothstep(std::log2(f / 160.0) / std::log2(220.0 / 160.0));
    double fall = smoothstep(std::log2(2000.0 / f));
    return std::pow(10.0, -k.low_mid_depth_db * rise * fall / 20.0);
}

// Driving-point admittance of the soundboard at the bridge, as a modal sum
// with roughly constant modal density (one mode per ~15 Hz, as for a plate),
// loss factor 0.03 and Rayleigh-distributed mode shapes at the bridge, scaled
// so the mean mobility over 150-1500 Hz is 1.5e-3 s/kg (measured grands sit
// near 1e-3). A string mode at f loses energy to the board at a rate set by
// Re(Y(f)) and is pulled off its harmonic position by Im(Y(f)): the piano's
// partials decay at irregular, mode-specific rates and sit slightly detuned,
// where a rigidly terminated string (a guitar) decays smoothly. The mean of
// Re(Y) is removed because the fitted T60 law already carries the average.
struct BridgeAdmittance {
    static constexpr double F_MAX = 12000.0;
    std::vector<double> re, im;   // per Hz from 0 to F_MAX
    double mean_re = 0;
    BridgeAdmittance() {
        std::mt19937_64 rng{0x6272696467655aULL};
        std::uniform_real_distribution<double> u(0.0, 1.0);
        std::vector<double> fm, sm, am;
        for (double f = 55.0; f < F_MAX * 1.2; f += 15.0 * (0.6 + 0.8 * u(rng))) {
            double phi2 = -std::log(std::max(1e-6, u(rng)));   // exponential (Rayleigh power)
            fm.push_back(f); sm.push_back(PI * 0.03 * f); am.push_back(phi2);
        }
        re.assign((size_t)F_MAX + 1, 0.0); im.assign((size_t)F_MAX + 1, 0.0);
        for (size_t i = 1; i < re.size(); ++i) {
            double w = 2 * PI * (double)i; std::complex<double> y = 0;
            for (size_t m = 0; m < fm.size(); ++m) {
                double wm = 2 * PI * fm[m];
                y += std::complex<double>(0, w) * am[m] / std::complex<double>(wm * wm - w * w, 2 * sm[m] * w);
            }
            re[i] = y.real(); im[i] = y.imag();
        }
        // Bridge mobility is highest in the low hundreds of hertz and falls
        // above the bridge's own cutoff as its mass impedance takes over, so
        // low partials drain into the board quickly while high partials ring
        // on. Normalise to a mean of 1.5e-3 s/kg over 150-300 Hz.
        for (size_t i = 0; i < re.size(); ++i) {
            double trend = 1.0 / (1.0 + (double)i / 400.0);
            re[i] *= trend; im[i] *= trend;
        }
        double sum = 0; int count = 0;
        for (int i = 150; i <= 300; ++i) { sum += re[i]; ++count; }
        double scale = 1.5e-3 / (sum / count);
        for (size_t i = 0; i < re.size(); ++i) { re[i] *= scale; im[i] *= scale; }
        mean_re = 1.5e-3;
    }
    static const BridgeAdmittance& get() { static const BridgeAdmittance table; return table; }
    double real_at(double f) const { return f >= F_MAX ? mean_re : re[(size_t)f] + (f - std::floor(f)) * (re[(size_t)f + 1] - re[(size_t)f]); }
    double imag_at(double f) const { return f >= F_MAX ? 0.0 : im[(size_t)f] + (f - std::floor(f)) * (im[(size_t)f + 1] - im[(size_t)f]); }
};

inline double treble_tilt(const KeyParams& k, double f) {
    double r = std::clamp(std::log2(k.f0 / midi_to_freq(79.0)) * 12.0 / 5.0, 0.0, 1.0);
    r *= r * (3.0 - 2.0 * r);
    double x = std::log2(f / (1.4 * k.f0));
    if (r <= 0.0 || x <= 0.0) return 1.0;
    return std::pow(10.0, -r * std::min(12.0 * x, 18.0) / 20.0);
}

inline StringModes build_string(const KeyParams& k, double detune_cents) {
    StringModes s;
    double f0 = k.f0 * std::pow(2.0, detune_cents / 1200.0);
    double sig1 = std::log(1000.0) / k.T60_1;  // 6.907/T60
    const double coupling = k.strings.bridge_coupling;
    const double Zs = std::sqrt(k.T * k.m_s / k.L);   // string wave impedance
    const BridgeAdmittance& Y = BridgeAdmittance::get();
    for (int n = 1; n <= 600; ++n) {
        double fn = n * f0 * std::sqrt(1.0 + k.B * n * n);
        if (fn > F_LIMIT) break;
        // The quadratic loss is capped above ~2.5 kHz: the reference's bass
        // partials at 2-5 kHz ring on where an f^2 law kills them.
        double sig = sig1 + k.c2 * (fn * fn - f0 * f0) / (1.0 + fn / 2500.0) + 0.8 * std::pow(fn / 4000.0, 3.0);
        if (coupling > 0.0 && fn < BridgeAdmittance::F_MAX) {
            // reflection phase -2 Zs Im(Y) pulls the mode by f0 Zs Im(Y) / pi;
            // the loss Re(Y) acts on the prompt stage, see fast_polarization_decay
            fn += coupling * f0 * Zs * Y.imag_at(fn) / PI;
        }
        s.w.push_back(2 * PI * fn);
        s.sig.push_back(sig);
        double hw = std::exp(-0.5 * std::pow(n * PI * k.sigma_h / k.L, 2.0));
        double sa = std::sin(n * PI * k.alpha) * hw;
        s.s_a.push_back(sa);
        s.g.push_back(2.0 / k.m_s * sa);
        s.w_br.push_back(k.T * (n * PI / k.L) * ((n & 1) ? -1.0 : 1.0)
                         * bridge_trans(k, fn) * treble_tilt(k, fn)
                         * (n == 1 ? k.fundamental_gain : lower_overtone_gain(k, fn)));
        s.q.push_back(0.0);
        s.qd.push_back(0.0);
    }
    s.n_modes = (int)s.w.size();
    return s;
}

struct ContactResult {
    double duration_ms = 0;
    double peak_force  = 0;
    ContactStatus status = ContactStatus::not_run;
};

// Integrate hammer + strings until the hammer separates. The strings keep
// their (q, qd) state for the free-vibration phase.
inline ContactResult run_contact(const KeyParams& k, std::vector<StringModes>& strs,
                                 double v_h0, std::vector<double>& contact_force,
                                 std::array<std::vector<double>,3>* string_force = nullptr) {
    constexpr int OS = 32;                  // oversampling vs FS
    const double dt = 1.0 / (FS * (double)OS);
    const int max_steps = (int)(0.030 * FS * OS);

    double y_h = 0.0, v_h = v_h0;
    bool touched = false;
    ContactResult res;
    contact_force.clear();
    if (string_force) for (auto& force : *string_force) force.clear();
    bool separated = false;
    int step = 0;
    for (; step < max_steps; ++step) {
        // Finish the fractional output sample after separation. The state
        // returned below belongs exactly at contact_force.size() / FS.
        if (separated && step % OS == 0) break;
        // emit the bridge force at output rate so the note's waveform ramps
        // up through the strike instead of starting at a step
        if (step % OS == 0) {
            double fb = 0.0;
            for (const auto& s : strs)
                for (int n = 0; n < s.n_modes; ++n) fb += s.w_br[n] * s.q[n];
            contact_force.push_back(fb);
            if (string_force) for (size_t i = 0; i < strs.size(); ++i) {
                double force = 0;
                for (int n = 0; n < strs[i].n_modes; ++n)
                    force += strs[i].w_br[n] * strs[i].q[n];
                (*string_force)[i].push_back(force);
            }
        }
        double F_tot = 0.0;
        bool any_contact = false;
        for (auto& s : strs) {
            // string displacement under the hammer
            double y_s = 0.0;
            for (int n = 0; n < s.n_modes; ++n) y_s += s.s_a[n] * s.q[n];
            double delta = y_h - y_s;
            double F = 0.0;
            if (!separated && delta > 0.0) {
                any_contact = true;
                // d(delta)/dt for hysteresis
                double yd_s = 0.0;
                for (int n = 0; n < s.n_modes; ++n) yd_s += s.s_a[n] * s.qd[n];
                double ddot = v_h - yd_s;
                F = k.K * std::pow(delta, k.p) * (1.0 + 0.12 * ddot);
                if (F < 0.0) F = 0.0;
            }
            F_tot += F;
            res.peak_force = std::max(res.peak_force, F);
            // advance this string's modes (symplectic Euler)
            for (int n = 0; n < s.n_modes; ++n) {
                s.qd[n] += dt * (-s.w[n] * s.w[n] * s.q[n]
                                 - 2.0 * s.sig[n] * s.qd[n] + s.g[n] * F);
                s.q[n]  += dt * s.qd[n];
            }
        }
        if (any_contact) touched = true;
        if (!separated && touched && !any_contact && v_h < 0.0) {
            separated = true;
            res.duration_ms = (step + 1) * dt * 1000.0;
        }
        v_h -= dt * F_tot / k.m_h;
        y_h += dt * v_h;
    }
    if (!separated) res.duration_ms = step * dt * 1000.0;
    res.status = separated ? ContactStatus::separated : ContactStatus::timed_out;
    return res;
}

// ---------------------------------------------------------------------------
// Free vibration: each (mode x polarization) becomes a complex phasor.
// Vertical polarization carries the strike and decays fast; horizontal gets
// a small share and decays at the fitted late-stage rate -> two-stage decay.

struct Phasor {
    std::complex<double> z;       // current state, output = Re(z)*w_out
    std::complex<double> step;    // per-sample multiplier (open strings)
    std::complex<double> step_r;  // per-sample multiplier (damper fully seated)
    std::complex<double> step_now; // committed blend for the current damper position
    double w_out;                 // bridge weight
    double amp0;                  // |z|*|w_out| at start, for early kill
    double sigma;                 // hold-stage decay rate
    bool longitudinal = false;   // sum-frequency component: smooth its onset
    bool fundamental = false;    // first transverse mode, including both polarizations
};

// Shared lifecycle policy for both rendering loops. Keep the loops separate:
// offline rendering visits one mode at a time; streaming visits one sample.
inline constexpr int DAMPER_TRANSITION_SAMPLES = (int)(0.040 * FS);
inline constexpr int CULL_INTERVAL_SAMPLES = 8192;
// Per-phasor kill threshold relative to the note peak: -90 dB. Measured on
// a full Beethoven movement, -120 dB cost about 5% more CPU and peak voices;
// on the sustain-heavy stress bench the difference was 12-20% of callback time.
inline constexpr double SILENCE_RATIO = 3.1622776601683795e-5;

inline double phasor_sample(const Phasor& p, const std::complex<double>& z,
                            double onset = 1.0) {
    return p.w_out * z.real() * (p.longitudinal ? onset : 1.0);
}

inline std::complex<double> release_multiplier(const Phasor& p, int position) {
    double t = (double)position / DAMPER_TRANSITION_SAMPLES;
    return p.step * (1.0 - t) + p.step_r * t;
}

inline std::int64_t release_sample(std::int64_t requested, int contact_samples) {
    return std::max(requested, (std::int64_t)contact_samples);
}

// Damper contact for a sustain pedal position (0 up .. 1 fully down), as the
// fraction of the seated damper's loss rate: the dampers stay seated over the
// first part of the travel, clear the strings past the middle, and between
// them the felt touches lightly, so the rate fraction is the square of the
// remaining travel (light contact damps much less than half). Hand-tuned; no
// half-pedal reference recordings were available.
inline constexpr double PEDAL_CONTACT_POSITION = 0.25, PEDAL_CLEAR_POSITION = 0.65;
inline double damper_contact(double pedal_position) noexcept {
    double x = (pedal_position - PEDAL_CONTACT_POSITION) / (PEDAL_CLEAR_POSITION - PEDAL_CONTACT_POSITION);
    if (!(x > 0.0)) return 1.0;
    if (x >= 1.0) return 0.0;
    double seated = 1.0 - x * x * (3.0 - 2.0 * x);
    return seated * seated;
}

inline bool below_threshold(const Phasor& p, double threshold) {
    return std::abs(p.z) * std::abs(p.w_out) < threshold;
}

// Bound before converting to int, including extremely long decay lifetimes.
inline int lifetime_samples(double amplitude, double threshold, double sigma, int limit) {
    if (sigma <= 0.0) return limit;
    double samples = std::log(std::max(amplitude / threshold, 1.0)) / sigma * FS;
    if (samples >= limit) return limit;
    return std::min(limit, (int)samples + 1);
}


// Set output gains before contact so the same readout is used throughout
// the note. Retain the calibrated 1.4x free-vibration level, then partition
// the state between polarizations instead of adding energy at separation.
// Unison strings never reach the bridge with equal weight, so a detuned pair
// beats but never cancels. Bichords get a fixed 25% split; trichords keep the
// small random spread only (the middle string is the reference).
inline double unison_asymmetry(size_t n_strings, size_t index) {
    if (n_strings == 2) return index == 0 ? 0.25 : -0.25;
    return 0.0;
}

inline void prepare_readout(std::vector<StringModes>& strs, std::mt19937_64& rng,
                            std::mt19937_64& tuning_rng, double horizontal_ratio = StringVoicing{}.horizontal_ratio) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (size_t i = 0; i < strs.size(); ++i) {
        auto& s = strs[i];
        double gain = (1.0 + horizontal_ratio) * (1.0 + unison_asymmetry(strs.size(), i) + 0.13 * u(rng));
        for (auto& weight : s.w_br) weight *= gain;
        s.wh.reserve(s.w.size());
        for (double w : s.w)
            s.wh.push_back(w * (1.0 + 2.5e-6 * (1.0 + 0.8 * u(tuning_rng))));
    }
}

// Fixed key coefficients and tuning. Strike gains are deliberately absent;
// each preparation scales a private copy, preserving independent seed variation.
inline std::vector<StringModes> prepare_key_strings(const KeyParams& k) {
    std::mt19937_64 tuning_rng{0x7069616e6fULL
        ^ (uint64_t)std::llround(k.f0 * 1048576.0)};
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<StringModes> strings;
    static const double det3[3] = {-1.15, 0.1, 0.7};
    for (int i = 0; i < k.n_strings; ++i) {
        double d = k.n_strings == 1 ? 0.0
            : det3[i] * k.det_scale * (0.8 + 0.4 * u(tuning_rng));
        strings.push_back(build_string(k, d));
    }
    for (auto& string : strings) {
        string.wh.reserve(string.w.size());
        for (double w : string.w)
            string.wh.push_back(w * (1.0 + 2.5e-6 * (1.0 + 0.8 * u(tuning_rng))));
    }
    return strings;
}

// The existing phantom model specifies established sum-frequency phasors,
// not their forced buildup during contact. A short C1-continuous onset
// avoids switching those nonzero states directly into the bridge force.
// This is an onset approximation, not a full longitudinal contact solver.
inline constexpr int LONGITUDINAL_RISE_SAMPLES = (int)(0.002 * FS);
inline double longitudinal_onset(std::int64_t age) {
    constexpr int rise = LONGITUDINAL_RISE_SAMPLES;
    if (age >= rise) return 1.0;
    double t = std::clamp((double)age / rise, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

inline std::complex<double> polarization_state(double q, double qd,
                                               double w, double sigma,
                                               double share) {
    // Re(z) = q, and Re((-sigma + i*w)*z) = qdot.
    return share * std::complex<double>(q, -(qd + sigma * q) / w);
}

inline double aftersound_share(int partial) {
    // The fundamental keeps 0.35 of the register share, the second partial
    // 0.6, the rest the full share: the reference drains each note's
    // fundamental 16 dB from strike to mid-sustain whatever its frequency,
    // while a bass note's partials at the same frequencies hold.
    return partial == 1 ? 0.35 : partial == 2 ? 0.6 : 1.0;
}

inline double fast_polarization_decay(const KeyParams& k, double slow_sigma, double f_hz) {
    // Keep the calibrated low-mode fast/slow contrast, but apply a gentler
    // multiplier to frequency-dependent losses. Multiplying every loss by
    // ~7 in the bass/middle stripped upper harmonics from the early body.
    // The prompt stage is the motion the bridge drains, so its rate follows
    // the board's admittance at the partial: partials near a board resonance
    // decay several times faster than their neighbours, as on a real piano,
    // while the average over partials keeps the calibrated ratio.
    // This is an empirical two-stage decay model, not a coupled bridge solve.
    double base = std::log(1000.0) / k.T60_1;
    double prompt = base * k.fast_ratio;
    if (k.strings.bridge_coupling > 0.0 && f_hz < BridgeAdmittance::F_MAX) {
        // The prompt rate follows the bridge admittance at the partial, both
        // its falling trend with frequency and its mode-to-mode irregularity;
        // the reference low-frequency partial decays three times the old rate.
        const BridgeAdmittance& Y = BridgeAdmittance::get();
        double ratio = std::clamp(3.0 * Y.real_at(f_hz) / Y.mean_re, 0.1, 8.0);
        prompt *= std::pow(ratio, k.strings.bridge_coupling);
    }
    return prompt + std::max(0.0, slow_sigma - base) * std::sqrt(k.fast_ratio);
}

// Phantom partials (original-research-notes.md §11): transverse-to-longitudinal coupling is
// quadratic in string slope, so mode pairs (i,j) drive the longitudinal
// string resonance at f_i + f_j. The product of two decaying phasors is
// itself a decaying phasor: freq f_i+f_j, decay sigma_i+sigma_j, phase
// phi_i+phi_j — so phantoms drop straight into the phasor bank. Quadratic
// amplitude makes them surge at high velocity and vanish at pp for free.
inline constexpr double PHANTOM_FMAX = 4200.0;
inline constexpr double PHANTOM_PARENT_LIMIT = 0.65;

// The one-way quadratic approximation otherwise lets an ff phantom exceed
// the string components driving it (especially the 2nd/3rd partials in the
// bass). Bound that excitation per pair before resonant amplification,
// retaining the quadratic soft response and the established phase/decay.
// This is an empirical bound for the approximate one-way model, not a
// simulation of energy exchange with the transverse strings. The knee leaves small
// components effectively unchanged; it never clips the output waveform.
inline std::complex<double> bound_phantom(std::complex<double> z,
                                         double parent_amplitude) {
    double limit = PHANTOM_PARENT_LIMIT * parent_amplitude;
    if (limit <= 0.0) return 0.0;
    double ratio = std::abs(z) / limit;
    return z * std::pow(1.0 + std::pow(ratio, 8.0), -1.0 / 8.0);
}

// Resonant boost near the string's longitudinal modes
inline double longitudinal_gain(double f, double f_long1) {
    double g = 1.0;
    for (int m = 1; m * f_long1 < PHANTOM_FMAX + 800.0; ++m) {
        double fm = m * f_long1;
        double bw = fm / 50.0;                    // Q ~ 25
        g += 4.6 * bw * bw / ((f - fm) * (f - fm) + bw * bw);
    }
    return g;
}

struct Parent { std::complex<double> a; double w, sig; };

// Smooth amplitude curve: retain E2 and below, reach 30% at C3, then
// reach zero at G#3, avoiding an abrupt tonal change between G3 and G#3.
inline double middle_longitudinal_scale(double frequency) {
    // Exact frequency endpoints avoid inverse-log rounding adding tiny modes
    // at G#3, which would also alter SIMD summation grouping.
    if (frequency <= midi_to_freq(40.0)) return 1.0;
    if (frequency >= midi_to_freq(56.0)) return 0.0;
    double midi = 69.0 + 12.0 * std::log2(frequency / 440.0);
    auto smooth = [](double x) {
        x = std::clamp(x, 0.0, 1.0);
        return x * x * (3.0 - 2.0 * x);
    };
    if (midi <= 40.0) return 1.0;
    if (midi < 48.0) return 1.0 - 0.7 * smooth((midi - 40.0) / 8.0);
    return 0.3 * (1.0 - smooth((midi - 48.0) / 8.0));
}

inline void add_phantoms(const KeyParams& k, const std::vector<Parent>& par,
                         double damp_sig, std::vector<Phasor>& out, double register_gain = 1.0, bool soften_highs = false,
                         bool reserved = false) {
    if (par.empty()) return;
    double maxa = 0.0;
    for (const auto& p : par) maxa = std::max(maxa, std::abs(p.a));
    double thresh = 1e-4 * maxa;
    for (size_t i = 0; i < par.size(); ++i) {
        // Selection rule: pair (i,j) couples through longitudinal terms of
        // order |i-j|, so only near-diagonal pairs contribute audibly — the
        // measured phantoms (2f13, f17+f18, 2f18) are all neighbor pairs.
        for (size_t j = i; j < std::min(i + 4, par.size()); ++j) {
            double w_ph = par[i].w + par[j].w;
            double f_ph = w_ph / (2 * PI);
            if (f_ph > PHANTOM_FMAX) break;     // parents sorted by frequency
            double sel = std::pow(0.55, (double)(j - i));
            std::complex<double> z = k.strings.longitudinal_gain * sel * par[i].a * par[j].a;
            // parents carry bridge_trans at their own frequencies; the
            // phantom radiates at f_i+f_j, so swap the inherited tilt for
            // the one at the phantom's frequency
            z *= bridge_trans(k, f_ph)
                 / (bridge_trans(k, par[i].w / (2 * PI))
                    * bridge_trans(k, par[j].w / (2 * PI)));
            z = bound_phantom(z, std::sqrt(std::abs(par[i].a) * std::abs(par[j].a)));
            // Limit the excitation, not the resonant response: retain the
            // narrow upper-bass bloom around longitudinal string modes.
            z *= longitudinal_gain(f_ph, k.f_long1);
            if (std::abs(z) < thresh) continue;
            // Voice the radiated sum tone after excitation and its bound.
            // Reducing its parents would also change nonlinear generation.
            z *= lower_overtone_gain(k, f_ph);
            // Apply after the nonlinear bound so reductions remain predictable.
            z *= register_gain;
            if (soften_highs) {
                // Retain the lower body, then blend to
                // 25% amplitude above 1 kHz. Only the added sum tones change.
                double band = std::clamp((f_ph - 500.0) / 500.0, 0.0, 1.0);
                band = band * band * (3.0 - 2.0 * band);
                z *= 1.0 - 0.75 * band;
            }
            Phasor p;
            p.longitudinal = true;
            double sg = par[i].sig + par[j].sig;
            p.z = z; p.w_out = 1.0;
            p.sigma = sg;
            p.step   = std::exp(std::complex<double>(-sg / FS, w_ph / FS));
            p.step_r = std::exp(std::complex<double>(-std::max(2.0 * damp_sig, sg) / FS, w_ph / FS));
            p.step_now = p.step;
            p.amp0 = std::abs(z);
            assert(!reserved || out.size() < out.capacity());
            out.push_back(p);
        }
    }
}

inline void collect_phasors(const KeyParams& k, const std::vector<StringModes>& strs,
                            std::vector<Phasor>& out, double* tone_peak = nullptr,
                            const Options& options = {},
                            std::vector<Parent>* scratch_v = nullptr,
                            std::vector<Parent>* scratch_h = nullptr, bool reserved = false) {
    assert(!reserved || (scratch_v && scratch_h));
    if (tone_peak) *tone_peak = 0.0;
    double register_gain = middle_longitudinal_scale(k.f0);
    bool want_phantoms = register_gain > 0.0;
    // Both transverse parents and their longitudinal products use the same
    // release-time scale. add_phantoms doubles this loss for a parent pair.
    const double damp_scale = options.damper_scale;
    const double key_damp_sig = k.has_damper
        ? std::log(1000.0) / std::clamp(damp_scale * 0.25 * std::pow(220.0 / k.f0, 0.3), 0.06 * damp_scale, 0.45 * damp_scale)
        : 0.0;
    for (const auto& s : strs) {
        std::vector<Parent> local_v, local_h;
        auto& par_v = scratch_v ? *scratch_v : local_v;
        auto& par_h = scratch_h ? *scratch_h : local_h;
        par_v.clear(); par_h.clear();
        for (int n = 0; n < s.n_modes; ++n) {
            double w   = s.w[n];
            double sg  = s.sig[n];
            double q   = s.q[n];
            double qd  = s.qd[n];
            if (q == 0.0 && qd == 0.0) continue;

            double damp_sig = k.has_damper ? key_damp_sig : sg;

            // Partition displacement AND velocity. Each component needs
            // its own decay/frequency in the state conversion.
            Phasor pv;
            pv.fundamental = (n == 0);
            double sgv = fast_polarization_decay(k, sg, w / (2 * PI));
            // Accepted slow-tail extension: fade C6–F#6, then extend further C7–C8.
            // Prompt rate was computed above and retains the accepted strike.
            double top_tail = std::clamp(std::log2(k.f0 / midi_to_freq(84.0)) * 2.0, 0.0, 1.0);
            top_tail *= top_tail * (3.0 - 2.0 * top_tail);
            double last_octave = std::clamp(std::log2(k.f0 / midi_to_freq(96.0)), 0.0, 1.0);
            last_octave *= last_octave * (3.0 - 2.0 * last_octave);
            sg *= 1.0 - 0.45 * top_tail - 0.25 * last_octave;

            // Accepted treble voicing: soften upper partials after hammer contact.
            // C5 and below unchanged; smoothly reach full strength at E5.
            double upper = std::clamp(std::log2(k.f0 / midi_to_freq(72.0)) * 3.0, 0.0, 1.0);
            upper *= upper * (3.0 - 2.0 * upper);
            double loss = upper * 0.6 * std::min(n, 3);
            sg += loss;
            sgv += loss;
            // Fast strike-to-tail transition, independent of bridge mobility.
            double high = std::clamp(std::log2(k.f0 / midi_to_freq(76.0)) * 4.0, 0.0, 1.0);
            high *= high * (3.0 - 2.0 * high);
            // The top octave dies away faster after the strike: C8 in the
            // reference loses ~11 dB per 100 ms over its first 0.3 s.
            const double early_rate = (n == 0 ? 10.0 : 5.0 + 2.0 * std::min(n, 3) + 3.0 * high)
                                    * (1.0 + 0.6 * last_octave);
            const double early_mix = n == 0 ? high : upper;
            sgv += early_mix * std::max(0.0, early_rate - sgv);

            // Preserve the calibrated low-partial shares. Above C5, reduce
            // the upper-partial tail alongside the faster prompt decay so
            // the strike settles into a quieter, softer sustained tone.
            const double tail_scale = n == 0 ? 1.0 : n == 1 ? 0.85 - 0.20 * high : 0.75 - 0.25 * high;
            const double share = k.strings.horizontal_ratio * aftersound_share(n + 1)
                               * (1.0 + upper * (tail_scale - 1.0));
            pv.z = polarization_state(q, qd, w, sgv, 1.0 / (1.0 + share));
            pv.w_out = s.w_br[n];
            pv.sigma = sgv;
            pv.step   = std::exp(std::complex<double>(-sgv / FS, w / FS));
            pv.step_r = std::exp(std::complex<double>(-std::max(damp_sig, sgv) / FS, w / FS));
            pv.step_now = pv.step;
            pv.amp0 = std::abs(pv.z) * std::abs(pv.w_out);
            assert(!reserved || out.size() < out.capacity());
            out.push_back(pv);

            // horizontal: small amplitude, slow decay, hair of detune
            Phasor ph;
            ph.fundamental = (n == 0);
            double wh = s.wh[n];
            ph.z = polarization_state(q, qd, wh, sg, share / (1.0 + share));
            ph.w_out = s.w_br[n];
            ph.sigma = sg;
            ph.step   = std::exp(std::complex<double>(-sg / FS, wh / FS));
            ph.step_r = std::exp(std::complex<double>(-std::max(damp_sig, sg) / FS, wh / FS));
            ph.step_now = ph.step;
            ph.amp0 = std::abs(ph.z) * std::abs(ph.w_out);
            assert(!reserved || out.size() < out.capacity());
            out.push_back(ph);

            double voicing = n == 0 ? k.fundamental_gain
                                   : lower_overtone_gain(k, w / (2 * PI));
            double parent_weight = s.w_br[n] / voicing;
            if (tone_peak)
                *tone_peak = std::max({*tone_peak, std::abs(pv.z) * std::abs(parent_weight),
                                                   std::abs(ph.z) * std::abs(parent_weight)});

            // parent pool for phantom generation: bridge-weighted amplitudes
            if (want_phantoms && w / (2 * PI) < 3500.0) {
                // Keep radiation voicing separate from nonlinear excitation
                // and mechanical-noise drive, including the fundamental gain.
                assert(!reserved || par_v.size() < par_v.capacity());
                par_v.push_back({pv.z * parent_weight, w, sgv});
                assert(!reserved || par_h.size() < par_h.capacity());
                par_h.push_back({ph.z * parent_weight, wh, sg});
            }
        }
        if (want_phantoms) {
            // vertical x vertical dominates the attack; horizontal x
            // horizontal carries the (much quieter) phantom aftersound
            add_phantoms(k, par_v, key_damp_sig, out, register_gain, options.soften_longitudinal_highs, reserved);
            add_phantoms(k, par_h, key_damp_sig, out, register_gain, options.soften_longitudinal_highs, reserved);
        }
    }
}

// ---------------------------------------------------------------------------
// Note preparation: everything a note needs before its first sample, shared
// by the offline renderer and the live engine. The attack buffer carries the
// contact-phase bridge force plus action thump and felt scrape; phasors take
// over coff samples in.

struct NoteVoiceData {
    std::vector<double> attack;      // from note start; includes contact force
    int coff = 0;                    // phasor start within the note timeline
    std::vector<Phasor> ph;
    std::vector<double> damp_noise;  // mixed in from the release point
    double peak_amp = 0.0;
    double tone_peak_amp = 0.0;     // transverse excitation before body voicing, for action noise
    NoteDebug dbg{};
};

inline double knock_velocity(double velocity, double f0) {
    // Above mf, the old velocity multiplier made a generic 12 ms thump
    // grow faster than the string body. Compress only this action layer's
    // additional growth in bass/mid keys, preserving the treble voicing.
    double excess = std::max(0.0, velocity - 0.5);
    double restrained = std::min(velocity, 0.5) + 0.12 * std::tanh(excess / 0.12);
    double low_mid = std::clamp(std::log2(880.0 / f0), 0.0, 1.0);
    return velocity + low_mid * (restrained - velocity);
}

// Fixed impact resonances, shared by all keys and strikes. Workspace setup
// warms this table off the audio thread; ordinary preparation may initialize it
// on demand. Only the body/strike gain varies, and no random draws are cached.
inline const std::array<double, (size_t)(0.65 * FS)>& structural_knock_waveform() {
    static const auto waveform = [] {
        std::array<double, (size_t)(0.65 * FS)> samples{};
        const double freq[] = {89.0, 117.0, 163.0, 231.0, 277.0};
        const double weight[] = {1.0, 0.65, 0.32, 0.20, 0.28};
        for (int j = 0; j < 5; ++j) {
            const double decay = 8.0 + 0.01 * freq[j];
            for (size_t i = 0; i < samples.size(); ++i) {
                const double t = (double)i / FS;
                const double rise = 1.0 - std::exp(-t / 0.003);
                const double end = std::clamp((0.65 - t) / 0.04, 0.0, 1.0);
                samples[i] += weight[j] * rise * rise * std::exp(-decay * t)
                    * std::sin(2.0 * PI * freq[j] * t) * end * end;
            }
        }
        return samples;
    }();
    return waveform;
}

// Optional internal stage observer for the preparation benchmark. The default
// is inlined away; the synthesis library never reads clocks or writes reports.
enum class PreparationStage { strings, contact, phasors, noise };
struct NoPreparationTrace {
    void operator()(PreparationStage) const noexcept {}
};

struct DirectContact {
    ContactResult operator()(const KeyParams& key, std::vector<StringModes>& strings,
                             double velocity, std::vector<double>& force) const {
        return run_contact(key, strings, velocity, force);
    }
};

template<class Trace = NoPreparationTrace, class Contact = DirectContact>
inline void prep_note_into(const KeyParams& k, double velocity, uint64_t seed,
                               NoteVoiceData& v, std::vector<StringModes>& strs,
                               std::vector<double>& cforce, std::vector<Parent>& parents_v,
                               std::vector<Parent>& parents_h,
                               const Options& options = {}, Trace trace = {},
                               const std::vector<StringModes>* fixed_strings = nullptr,
                               Contact contact = {}, bool reserved = false) {
    std::mt19937_64 rng{seed};
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    v.ph.clear(); v.attack.clear(); v.damp_noise.clear();
    v.peak_amp = v.tone_peak_amp = 0; v.coff = 0; v.dbg = {};

    velocity = std::clamp(velocity, 0.02, 1.0);
    double v_h = k.hammer.maximum_velocity_mps * std::pow(velocity, k.hammer.velocity_exponent);

    if (fixed_strings) {
#ifndef NDEBUG
        if (reserved) {
            assert(strs.size() == fixed_strings->size());
            for (size_t i = 0; i < strs.size(); ++i) {
                const auto& a = strs[i]; const auto& b = (*fixed_strings)[i];
                for (auto buffers : {std::pair{&a.w,&b.w}, {&a.wh,&b.wh}, {&a.sig,&b.sig},
                                     {&a.g,&b.g}, {&a.s_a,&b.s_a}, {&a.w_br,&b.w_br},
                                     {&a.q,&b.q}, {&a.qd,&b.qd}})
                    assert(buffers.first->capacity() >= buffers.second->size());
            }
        }
#endif
        strs = *fixed_strings;
        for (auto& string : strs) {
            double gain = (1.0 + k.strings.horizontal_ratio) * (1.0 + unison_asymmetry(strs.size(), &string - strs.data()) + 0.13 * u(rng));
            for (auto& weight : string.w_br) weight *= gain;
        }
    } else {
        assert(!reserved); // Realtime workspaces must use precomputed strings.
        strs.clear();
        // Tuning belongs to the key, not the strike. Keep the same unison and
        // polarization frequencies across repeated notes and live/offline seeds.
        std::mt19937_64 tuning_rng{0x7069616e6fULL
            ^ (uint64_t)std::llround(k.f0 * 1048576.0)};
        static const double det3[3] = {-1.15, 0.1, 0.7};
        for (int i = 0; i < k.n_strings; ++i) {
            double d = (k.n_strings == 1) ? 0.0
                     : det3[i] * k.det_scale * (0.8 + 0.4 * u(tuning_rng));
            strs.push_back(build_string(k, d));
        }

        prepare_readout(strs, rng, tuning_rng, k.strings.horizontal_ratio);
    }
    trace(PreparationStage::strings);
    ContactResult con = contact(k, strs, v_h, cforce);
    trace(PreparationStage::contact);
    collect_phasors(k, strs, v.ph, &v.tone_peak_amp, options, &parents_v, &parents_h, reserved);
    trace(PreparationStage::phasors);

    v.coff = (int)cforce.size();
    for (auto& p : v.ph) {
        v.peak_amp = std::max(v.peak_amp, p.amp0);
    }

    // attack buffer: contact force + thump + scrape (they overlap in time)
    {
        std::normal_distribution<double> g(0.0, 1.0);
        // Keybed/board thump.
        double knock_ms = 12.0;
        int klen = (int)(knock_ms * 1e-3 * FS);
        int slen = (int)(0.025 * FS);
        assert(!reserved || (size_t)std::max({v.coff, klen, slen}) <= v.attack.capacity());
        v.attack.assign(std::max({v.coff, klen, slen}), 0.0);
        for (int i = 0; i < v.coff; ++i) v.attack[i] += cforce[i];

        // Tie action noise to the struck transverse strings. A resonantly
        // boosted phantom must not also turn up the unrelated keybed thump.
        double gain = !options.knock ? 0.0
            : 2.2 * options.knock_gain * v.tone_peak_amp * knock_velocity(velocity, k.f0);
        double treble_knock = std::clamp(std::log2(k.f0 / midi_to_freq(84.0)), 0.0, 1.0);
        treble_knock *= treble_knock * (3.0 - 2.0 * treble_knock);
        double top_octave = std::clamp(std::log2(k.f0 / midi_to_freq(96.0)), 0.0, 1.0);
        top_octave *= top_octave * (3.0 - 2.0 * top_octave);
        const double thump = gain * (1.0 + 0.8 * treble_knock * (1.0 - top_octave));
        double lpa = 0.0, lpb = 0.0;
        for (int i = 0; i < klen; ++i) {
            double x = (double)i / klen;
            double pulse = std::sin(PI * x) * std::sin(PI * x);
            lpa = 0.92 * lpa + 0.08 * g(rng);
            lpb = 0.92 * lpb + 0.08 * lpa;
            v.attack[i] += thump * (pulse + 3.4 * lpb * std::sin(PI * x));
        }
        // Short structural resonances excited by the impact.
        // Fade G#4–E5: half excitation at C5, full at E5.
        double body = std::clamp(std::log2(k.f0 / midi_to_freq(68.0)) * 1.5, 0.0, 1.0);
        body *= body * (3.0 - 2.0 * body);
        if (options.knock && body > 0.0) {
            const auto& waveform = structural_knock_waveform();
            assert(!reserved || waveform.size() <= v.attack.capacity());
            v.attack.resize(std::max(v.attack.size(), waveform.size()), 0.0);
            const double scale = 0.20 * body * gain;
            for (size_t i = 0; i < waveform.size(); ++i)
                v.attack[i] += scale * waveform[i];
        }
        double sgain = !options.scrape ? 0.0 : 0.011 * v.tone_peak_amp * std::pow(velocity, 1.8);
        double h1 = 0.0, x1 = 0.0, lp2 = 0.0;
        for (int i = 0; i < slen; ++i) {
            double t = (double)i / slen;
            double x = g(rng);
            lp2 = 0.45 * lp2 + 0.55 * x;          // tame the very top
            double hp = lp2 - x1 + 0.93 * h1;     // strip the lows
            x1 = lp2; h1 = hp;
            double env = std::exp(-7.0 * t) * (1.0 - std::exp(-40.0 * t));
            v.attack[i] += sgain * hp * env;
        }
        if (options.knock && treble_knock > 0.0) {
            double ma = 0.0, mb = 0.0, mc = 0.0;
            const int mlen = klen;
            for (int i = 0; i < mlen; ++i) {
                const double t = (double)i / FS;
                const double env = (1.0 - std::exp(-t / 0.0015)) * std::exp(-t / 0.004);
                ma = 0.8 * ma + 0.2 * g(rng);
                mb = 0.8 * mb + 0.2 * ma;
                mc = 0.8 * mc + 0.2 * mb;
                v.attack[i] += 11.0 * treble_knock * gain * env * mc;
            }
        }
    }

    // damper felt brushing the string, ready for whenever release happens
    if (k.has_damper && options.damper_noise) {
        std::normal_distribution<double> g(0.0, 1.0);
        int len = (int)(0.030 * FS);
        assert(!reserved || (size_t)len <= v.damp_noise.capacity());
        v.damp_noise.assign(len, 0.0);
        double gain = 0.0035 * v.tone_peak_amp * (0.4 + 0.6 * velocity)
                      * std::clamp(std::pow(261.0 / k.f0, 0.3), 0.5, 1.4);
        double lp = 0.0;
        for (int i = 0; i < len; ++i) {
            double t = (double)i / len;
            lp = 0.85 * lp + 0.15 * g(rng);
            v.damp_noise[i] = gain * lp * std::exp(-4.0 * t) * (1 - std::exp(-25 * t));
        }
    }

    v.dbg = {con.duration_ms, con.peak_force, v_h, (int)v.ph.size(), con.status};
    trace(PreparationStage::noise);
}

template<class Trace = NoPreparationTrace, class Contact = DirectContact>
inline NoteVoiceData prep_note(const KeyParams& k, double velocity, uint64_t seed,
                               const Options& options = {}, Trace trace = {},
                               const std::vector<StringModes>* fixed_strings = nullptr,
                               Contact contact = {}) {
    NoteVoiceData note;
    std::vector<StringModes> strings;
    std::vector<double> force;
    std::vector<Parent> parents_v, parents_h;
    prep_note_into(k, velocity, seed, note, strings, force, parents_v, parents_h,
                   options, trace, fixed_strings, contact);
    return note;
}

// Finite user settings can still overflow when combined with model amplitudes.
// Check preparation once, off the audio thread, before exposing/mixing a voice.
inline bool note_data_finite(const NoteVoiceData& note) noexcept {
    auto finite = [](const std::complex<double>& z) {
        return std::isfinite(z.real()) && std::isfinite(z.imag());
    };
    bool valid = std::isfinite(note.peak_amp) && note.peak_amp > 0.0
        && std::isfinite(note.tone_peak_amp);
    for (const auto* buffer : {&note.attack, &note.damp_noise})
        for (double sample : *buffer) valid &= std::isfinite(sample);
    for (const auto& mode : note.ph)
        valid &= finite(mode.z) && finite(mode.step) && finite(mode.step_r)
            && std::isfinite(mode.amp0) && std::isfinite(mode.w_out) && std::isfinite(mode.sigma);
    return valid;
}

inline void validate_note_data(const NoteVoiceData& note) {
    if (!note_data_finite(note))
        throw std::invalid_argument("piano: settings produce an unrepresentable voice");
}

inline NoteVoiceData prepare_note_checked(const KeyParams& key, double velocity,
                                          uint64_t seed, const Options& options,
                                          const std::vector<StringModes>* fixed_strings = nullptr) {
    auto note = prep_note(key, velocity, seed, options, NoPreparationTrace{}, fixed_strings);
    validate_note_data(note);
    return note;
}

// ---------------------------------------------------------------------------
// Render one note's bridge force into buf starting at offset.

inline NoteDebug render_note(const KeyParams& k, double velocity,
                             int hold_samples, std::vector<double>& buf,
                             int offset, uint64_t seed, const Options& options = {},
                             const std::vector<StringModes>* fixed_strings = nullptr) {
    velocity = std::clamp(velocity, 0.02, 1.0);
    NoteVoiceData nv = prepare_note_checked(k, velocity, seed, options, fixed_strings);
    const int coff = nv.coff;
    std::vector<Phasor>& ph = nv.ph;

    for (int i = 0; i < (int)nv.attack.size() && offset + i < (int)buf.size(); ++i)
        buf[offset + i] += nv.attack[i];

    // free vibration continues where the contact waveform left off
    const int n_total = (int)buf.size() - offset - coff;
    if (n_total <= 0) return nv.dbg;
    const double kill_thresh = SILENCE_RATIO;
    double peak_amp = nv.peak_amp;

    int release_at = k.has_damper
        ? (int)std::min(release_sample(hold_samples, coff) - coff, (std::int64_t)n_total) : n_total;
    const int trans = DAMPER_TRANSITION_SAMPLES;

    for (auto& p : ph) {
        auto sample = [&](const std::complex<double>& z, int age) {
            return phasor_sample(p, z, p.longitudinal ? longitudinal_onset(age) : 1.0);
        };
        // lifetime bound from hold-stage decay (release only shortens it)
        int kill = lifetime_samples(p.amp0, peak_amp * kill_thresh, p.sigma, n_total);
        std::complex<double> z = p.z;
        std::complex<double> st = p.step;
        int i = 0;
        int seg1 = std::min(kill, release_at);
        for (; i < seg1; ++i) {
            buf[offset + coff + i] += sample(z, i);
            z *= st;
        }
        if (i >= kill || i >= n_total) continue;
        // damper engages: crossfade the multiplier over the ramp
        for (int j = 0; j < trans && i < n_total; ++j, ++i) {
            std::complex<double> mix = release_multiplier(p, j);
            buf[offset + coff + i] += sample(z, i);
            z *= mix;
        }
        // released tail: bounded by damped decay
        double amp_now = std::abs(z) * std::abs(p.w_out);
        double sig_r = -std::log(std::abs(p.step_r)) * FS;
        int tail = i + lifetime_samples(amp_now, peak_amp * kill_thresh, sig_r, n_total - i);
        for (; i < tail; ++i) {
            buf[offset + coff + i] += sample(z, i);
            z *= p.step_r;
        }
    }

    // damper felt brushing the string at release
    if (k.has_damper && release_at < n_total) {
        for (int i = 0; i < (int)nv.damp_noise.size()
                        && offset + coff + release_at + i < (int)buf.size(); ++i)
            buf[offset + coff + release_at + i] += nv.damp_noise[i];
    }

    return nv.dbg;
}

// ---------------------------------------------------------------------------
// Soundboard impulse response synthesis.
// Discrete low modes blending into a dense modal continuum; frequency-
// dependent decay; board radiation envelope. Stereo via partially
// decorrelated phases (coherent at low f, independent at high f).

inline void fft(std::vector<std::complex<double>>& a, bool inverse);

// Empirical bridge-force -> radiated-pressure envelope. The former 165 Hz
// corner thinned low-note fundamentals; 90 Hz restores some of that body
// without fitting the response to individual notes in a room recording.
inline double board_radiation(double f, const SoundboardVoicing& board = {}) {
    return (f * f / (f * f + board.bass_corner_hz * board.bass_corner_hz))
         / (1.0 + std::pow(f / board.treble_corner_hz, board.treble_slope));
}

// The minimum-phase equalizer that flattens an impulse response toward the
// radiation target, as complex per-bin multipliers of a 2x zero-padded FFT.
inline std::vector<std::complex<double>> ir_correction(const std::vector<double>& ir, const SoundboardVoicing& board = {}) {
    size_t n = 1;
    while (n < ir.size() * 2) n <<= 1;
    std::vector<std::complex<double>> H(n);
    for (size_t i = 0; i < ir.size(); ++i) H[i] = ir[i];
    fft(H, false);

    size_t half = n / 2;
    std::vector<double> logmag(half + 1);
    for (size_t i = 0; i <= half; ++i)
        logmag[i] = std::log(std::abs(H[i]) + 1e-12);

    // constant-Q (1/3 octave) smoothing of log magnitude
    std::vector<double> smooth(half + 1);
    double cum = 0.0;
    std::vector<double> cumsum(half + 2, 0.0);
    for (size_t i = 0; i <= half; ++i) { cum += logmag[i]; cumsum[i + 1] = cum; }
    for (size_t i = 0; i <= half; ++i) {
        size_t lo = (size_t)(i / 1.122), hi = std::min(half, (size_t)(i * 1.122) + 1);
        if (hi <= lo) { smooth[i] = logmag[i]; continue; }
        smooth[i] = (cumsum[hi + 1] - cumsum[lo]) / (hi - lo + 1);
    }

    std::vector<std::complex<double>> correction(n);
    for (size_t i = 0; i <= half; ++i) {
        double f = (double)i * FS / n;
        // A finite, smooth target allows a causal minimum-phase correction.
        double target = std::max(1e-5, board_radiation(f, board))
                      / std::sqrt(1.0 + std::pow(f / 17500.0, 24.0));
        // Keep a quarter of the narrow modal ripple. Only equalizing the
        // smoothed envelope left its peaks/nulls intact. Limit inversion of
        // deep nulls: local correction cannot exceed 15 dB in either direction.
        double ripple = std::clamp(logmag[i] - smooth[i], -std::log(10.0), std::log(10.0));
        // The board's sparse low modes keep their own peaks and nulls (the
        // body of the instrument); above ~450 Hz only ripple_fraction remains.
        double keep = board.ripple_fraction + (1.0 - board.ripple_fraction) * std::clamp((450.0 - f) / 250.0, 0.0, 1.0);
        correction[i] = std::log(target) - smooth[i] - (1.0 - keep) * ripple;
        if (i > 0 && i < half) correction[n - i] = correction[i];
    }

    // Real cepstrum -> causal cepstrum -> minimum-phase equalizer. A real
    // zero-phase gain created negative-time response that was discarded on
    // truncation, changing the intended magnitude. Keep the board's phase
    // structure, adding only the phase needed by this causal correction.
    fft(correction, true);
    for (size_t i = 1; i < half; ++i) correction[i] *= 2.0;
    for (size_t i = half + 1; i < n; ++i) correction[i] = 0.0;
    fft(correction, false);
    for (auto& c : correction) c = std::exp(c);
    return correction;
}
inline void apply_ir_correction(std::vector<double>& ir, const std::vector<std::complex<double>>& multipliers) {
    const size_t n = multipliers.size();
    std::vector<std::complex<double>> H(n);
    for (size_t i = 0; i < ir.size(); ++i) H[i] = ir[i];
    fft(H, false);
    for (size_t i = 0; i < n; ++i) H[i] *= multipliers[i];
    fft(H, true);
    for (size_t i = 0; i < ir.size(); ++i) ir[i] = H[i].real();
}
inline void flatten_ir(std::vector<double>& ir, const SoundboardVoicing& board = {}) {
    apply_ir_correction(ir, ir_correction(ir, board));
}

// Bridge regions for spatial imaging. A key's position along the bridge runs
// from 0 (A0) to 1 (C8); each region covers one equal part, and a key mixes
// into its two nearest regions with a linear crossfade so there are no seams.
inline constexpr int SOUNDBOARD_REGIONS = 4;
inline double bridge_position(double midi) noexcept { return std::clamp((midi - 21.0) / 87.0, 0.0, 1.0); }
inline double region_position(int region) noexcept { return (region + 0.5) / SOUNDBOARD_REGIONS; }

// Per-region stereo soundboard responses. Every region shares the board's
// modes, decays and stereo phase structure; what differs is how strongly each
// mode is excited from that place on the bridge (position_depth: a random
// smooth mode shape whose spatial detail grows with frequency, averaging to
// the shared response across regions) and a constant-power pan (spread,
// low modes stay centred). The equalizer is derived from the shared response
// and applied to all regions, so the average response stays calibrated and
// the mono sum keeps the shared board's phase structure without comb filtering.
// With tail outputs, each region keeps only the first head_seconds of its
// response (raised-cosine fade over the last fifth) and the shared response's
// complement is returned as a single tail for the summed force: the image and
// position colour live in the early response, and the long low-frequency tail
// costs one convolution instead of one per region.
// Interpolate mode spacing, not two audio outputs. Changing the number of
// modes also changes subsequent random draws, so colour need not vary monotonically.
inline double board_mode_spacing(double f, double density) {
    const double dense = 3.2 + f * 0.0042;
    const double sparse = std::max(dense, 24.0 * (1.0 - f / 450.0));
    if (density <= 0.0) return sparse;
    if (density >= 1.0) return dense;
    return sparse + density * (dense - sparse);
}

inline constexpr double SOUNDBOARD_HEAD_SECONDS = 0.2;
inline void make_soundboard_regions(std::vector<std::vector<double>>& left,
                                    std::vector<std::vector<double>>& right,
                                    uint64_t seed, const SoundboardVoicing& board = {},
                                    std::vector<double>* tail_left = nullptr,
                                    std::vector<double>* tail_right = nullptr,
                                    double head_seconds = SOUNDBOARD_HEAD_SECONDS) {
    const int K = SOUNDBOARD_REGIONS;
    const double dur = 1.4 * board.decay_scale;
    const int N = (int)(dur * FS);
    std::vector<double> baseL(N, 0.0), baseR(N, 0.0);
    left.assign(K, std::vector<double>(N, 0.0));
    right.assign(K, std::vector<double>(N, 0.0));
    std::mt19937_64 rng{seed}, spatial{seed ^ 0x9e3779b97f4a7c15ULL};
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::uniform_real_distribution<double> um(-1.0, 1.0);
    std::array<double, SOUNDBOARD_REGIONS> weight{}, gainL{}, gainR{};

    double f = 48.0;
    while (f < 16500.0) {
        // Modal decay: about 0.35 s at 100 Hz, 0.2 s at 200 Hz, 0.1 s at 1 kHz.
        // The earlier 0.75 s low cap made the board a bass reverb after release.
        double T60 = 0.04 + 0.30 * std::pow(100.0 / f, 0.7);
        T60 = std::min(T60, 0.45) * board.decay_scale;
        double sig = std::log(1000.0) / T60;
        double amp = board_radiation(f, board) * std::pow(10.0, um(rng) * 3.0 / 20.0);
        double phL = u01(rng) * 2 * PI;
        double coher = std::clamp(f / 700.0, 0.0, 1.0);
        double phR = phL + coher * um(rng) * PI;
        double ampR = amp * std::pow(10.0, coher * um(rng) * 3.0 / 20.0);
        std::complex<double> zL = amp  * std::exp(std::complex<double>(0, phL));
        std::complex<double> zR = ampR * std::exp(std::complex<double>(0, phR));
        std::complex<double> st = std::exp(std::complex<double>(-sig / FS, 2 * PI * f / FS));
        int len = std::min(N, (int)(T60 * 1.15 * FS));

        // Mode shape along the bridge: more spatial cycles at higher
        // frequencies, zero-mean across regions so the average is unchanged.
        const double cycles = 0.5 + f / 800.0, phase = u01(spatial) * 2 * PI;
        double mean = 0.0;
        for (int r = 0; r < K; ++r) { weight[r] = std::cos(2 * PI * cycles * region_position(r) + phase); mean += weight[r] / K; }
        for (int r = 0; r < K; ++r) weight[r] = 1.0 + board.position_depth * 0.25 * (weight[r] - mean);
        // Constant-power pan by region, fading in above the bass where
        // radiation becomes directional; sqrt(2) keeps unity at the centre.
        const double directional = std::clamp((f - 60.0) / 540.0, 0.0, 1.0);
        for (int r = 0; r < K; ++r) {
            double angle = PI / 4 * (1.0 + board.spread * directional * (2 * region_position(r) - 1.0));
            gainL[r] = std::sqrt(2.0) * std::cos(angle); gainR[r] = std::sqrt(2.0) * std::sin(angle);
        }
        for (int i = 0; i < len; ++i) {
            const double l = zL.imag(), rr = zR.imag();
            baseL[i] += l; baseR[i] += rr;
            for (int r = 0; r < K; ++r) { left[r][i] += weight[r] * gainL[r] * l; right[r][i] += weight[r] * gainR[r] * rr; }
            zL *= st; zR *= st;
        }
        // Adjustable low-mode spacing; accepted default restores the dense layout.
        f += board_mode_spacing(f, board.low_mode_density) * (0.75 + 0.5 * u01(rng));
    }
    auto correctionL = ir_correction(baseL, board), correctionR = ir_correction(baseR, board);
    for (int r = 0; r < K; ++r) { apply_ir_correction(left[r], correctionL); apply_ir_correction(right[r], correctionR); }
    apply_ir_correction(baseL, correctionL); apply_ir_correction(baseR, correctionR);
    double e = 0.0;
    for (int i = 0; i < N; ++i) e += baseL[i] * baseL[i] + baseR[i] * baseR[i];
    double s = 1.0 / std::sqrt(e / 2.0);
    for (int r = 0; r < K; ++r) for (int i = 0; i < N; ++i) { left[r][i] *= s; right[r][i] *= s; }
    if (!tail_left || !tail_right) return;
    const int head = std::clamp((int)(head_seconds * FS), 1, N), fade = std::max(1, head / 5);
    auto window = [&](int i) {
        if (i < head - fade) return 1.0;
        if (i >= head) return 0.0;
        return 0.5 * (1.0 + std::cos(PI * (i - (head - fade)) / (double)fade));
    };
    tail_left->assign(N, 0.0); tail_right->assign(N, 0.0);
    for (int i = 0; i < N; ++i) {
        const double w = window(i);
        (*tail_left)[i] = baseL[i] * s * (1.0 - w); (*tail_right)[i] = baseR[i] * s * (1.0 - w);
        for (int r = 0; r < K; ++r) { left[r][i] *= w; right[r][i] *= w; }
    }
    for (int r = 0; r < K; ++r) { left[r].resize(head); right[r].resize(head); }
}

inline void make_soundboard_ir(std::vector<double>& irL, std::vector<double>& irR,
                               uint64_t seed, const SoundboardVoicing& board = {}) {
    const double dur = 1.4 * board.decay_scale;
    const int N = (int)(dur * FS);
    irL.assign(N, 0.0);
    irR.assign(N, 0.0);
    std::mt19937_64 rng{seed};
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::uniform_real_distribution<double> um(-1.0, 1.0);

    double f = 48.0;
    while (f < 16500.0) {
        // Modal decay: about 0.35 s at 100 Hz, 0.2 s at 200 Hz, 0.1 s at 1 kHz.
        // The earlier 0.75 s low cap made the board a bass reverb after release.
        double T60 = 0.04 + 0.30 * std::pow(100.0 / f, 0.7);
        T60 = std::min(T60, 0.45) * board.decay_scale;
        double sig = std::log(1000.0) / T60;

        double amp = board_radiation(f, board) * std::pow(10.0, um(rng) * 3.0 / 20.0);

        double phL = u01(rng) * 2 * PI;
        double coher = std::clamp(f / 700.0, 0.0, 1.0);   // 0 = identical phase
        double phR = phL + coher * um(rng) * PI;
        double ampR = amp * std::pow(10.0, coher * um(rng) * 3.0 / 20.0);

        std::complex<double> zL = amp  * std::exp(std::complex<double>(0, phL));
        std::complex<double> zR = ampR * std::exp(std::complex<double>(0, phR));
        std::complex<double> st = std::exp(std::complex<double>(-sig / FS, 2 * PI * f / FS));
        int len = std::min(N, (int)(T60 * 1.15 * FS));
        for (int i = 0; i < len; ++i) {
            irL[i] += zL.imag();
            irR[i] += zR.imag();
            zL *= st;
            zR *= st;
        }
        // modal spacing: dense at low f, opening up with frequency; kept
        // below the modal bandwidth so partials don't fall into notches
        // Adjustable low-mode spacing; accepted default restores the dense layout.
        f += board_mode_spacing(f, board.low_mode_density) * (0.75 + 0.5 * u01(rng));
    }

    // Control both the broad envelope and excessive narrow modal variation,
    // using a causal correction so truncation does not discard pre-response.
    for (auto* ir : {&irL, &irR}) flatten_ir(*ir, board);

    // normalize IR energy
    double e = 0.0;
    for (int i = 0; i < N; ++i) e += irL[i] * irL[i] + irR[i] * irR[i];
    double s = 1.0 / std::sqrt(e / 2.0);
    for (int i = 0; i < N; ++i) { irL[i] *= s; irR[i] *= s; }
}

// ---------------------------------------------------------------------------
// FFT convolution

inline void fft(std::vector<std::complex<double>>& a, bool inverse) {
    int n = (int)a.size();
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * PI / len * (inverse ? 1 : -1);
        std::complex<double> wl(std::cos(ang), std::sin(ang));
        for (int i = 0; i < n; i += len) {
            std::complex<double> w(1.0);
            for (int j = 0; j < len / 2; ++j) {
                auto x = a[i + j], y = a[i + j + len / 2] * w;
                a[i + j] = x + y;
                a[i + j + len / 2] = x - y;
                w *= wl;
            }
        }
    }
    if (inverse) for (auto& x : a) x /= n;
}

// Overlap-add stereo convolution: bounded FFT size so a six-minute piece
// doesn't need a quarter-gigabyte transform, and the input block spectrum
// is shared between the two channel IRs.
inline void convolve_stereo_unchecked(const std::vector<double>& x,
                            const std::vector<double>& hL,
                            const std::vector<double>& hR,
                            std::vector<double>& outL,
                            std::vector<double>& outR) {
    size_t hn = std::max(hL.size(), hR.size());
    size_t outn = x.size() + hn - 1;
    size_t n = 1;
    while (n < outn && n < (size_t)1 << 20) n <<= 1;
    while (n < hn * 2) n <<= 1;
    size_t seg = n - hn + 1;                  // input samples per block

    std::vector<std::complex<double>> HL(n), HR(n);
    for (size_t i = 0; i < hL.size(); ++i) HL[i] = hL[i];
    for (size_t i = 0; i < hR.size(); ++i) HR[i] = hR[i];
    fft(HL, false); fft(HR, false);

    outL.assign(outn, 0.0);
    outR.assign(outn, 0.0);
    std::vector<std::complex<double>> A, B;
    for (size_t start = 0; start < x.size(); start += seg) {
        size_t m = std::min(seg, x.size() - start);
        A.assign(n, 0.0);
        for (size_t i = 0; i < m; ++i) A[i] = x[start + i];
        fft(A, false);
        B.resize(n);
        for (size_t i = 0; i < n; ++i) B[i] = A[i] * HR[i];
        for (size_t i = 0; i < n; ++i) A[i] *= HL[i];
        fft(A, true); fft(B, true);
        size_t lim = std::min(outn - start, m + hn - 1);
        for (size_t i = 0; i < lim; ++i) {
            outL[start + i] += A[i].real();
            outR[start + i] += B[i].real();
        }
    }
}

inline constexpr size_t MAX_IR_SAMPLES = (size_t)1 << 28;
inline constexpr size_t MAX_RENDER_SAMPLES = (size_t)std::numeric_limits<int>::max() - FS;

inline void validate_options(const Options& o) {
    if (!std::isfinite(o.damper_scale) || o.damper_scale <= 0.0
        || !std::isfinite(o.knock_gain) || o.knock_gain < 0.0)
        throw std::invalid_argument("piano: damper_scale must be positive; knock_gain nonnegative; both finite");
}
inline void validate_velocity(double velocity) {
    if (!std::isfinite(velocity))
        throw std::invalid_argument("piano: velocity must be finite");
}
inline void validate_ir(const std::vector<double>& l, const std::vector<double>& r) {
    if (l.empty() && r.empty())
        throw std::invalid_argument("piano: at least one impulse response must be nonempty");
    if (std::max(l.size(), r.size()) > MAX_IR_SAMPLES)
        throw std::length_error("piano: impulse response is too long");
    for (const auto* channel : {&l, &r})
        for (double sample : *channel)
            if (!std::isfinite(sample))
                throw std::invalid_argument("piano: impulse response must contain finite samples");
}
inline void validate_buffer(const double* data, size_t size) {
    if (!data && size != 0)
        throw std::invalid_argument("piano: nonempty buffer has a null pointer");
    if (size > MAX_RENDER_SAMPLES)
        throw std::length_error("piano: buffer is too large");
}
inline bool overlaps(AudioBuffer a, AudioBuffer b) {
    if (!a.size || !b.size) return false;
    std::less<const double*> before;
    return before(a.data, b.data + b.size) && before(b.data, a.data + a.size);
}

} // namespace detail

// Immutable physical parameters produced only from a valid piano note number.
// Fractional note numbers in [21,108] are supported. parameters() is a read-only
// calibration view; detail::KeyParams is not part of the stable interface.
namespace detail { struct KeyAccess; }
class Key {
    detail::KeyParams params_;
    std::shared_ptr<const std::vector<detail::StringModes>> strings_;
    explicit Key(detail::KeyParams params)
        : params_(params), strings_(std::make_shared<const std::vector<detail::StringModes>>(
                                      detail::prepare_key_strings(params))) {}
    friend Key make_key(double, const Voicing&);
    friend struct detail::KeyAccess;
public:
    // Copies (including rvalues) share immutable coefficients. Keeping ordinary
    // copy semantics also leaves source keys valid after std::move.
    Key(const Key&) = default;
    Key& operator=(const Key&) = default;
    const detail::KeyParams& parameters() const noexcept { return params_; }
    double frequency_hz() const noexcept { return params_.f0; }
    bool has_damper() const noexcept { return params_.has_damper; }
};

namespace detail {
struct KeyAccess {
    static const std::vector<StringModes>& strings(const Key& key) noexcept { return *key.strings_; }
};
} // namespace detail

inline Key make_key(double note_number, const Voicing& voicing = {}) {
    validate_voicing(voicing);
    if (!std::isfinite(note_number) || note_number < 21.0 || note_number > 108.0)
        throw std::invalid_argument("piano: note number must be finite and in [21,108]");
    return Key(detail::make_key_params(note_number, voicing));
}

inline double note_tail_seconds(const Key& key) noexcept {
    return key.has_damper() ? 5.0 : 2.3 * key.parameters().T60_1 + 1.0;
}

// Add one note's bridge force. Negative hold/offset and offsets past the buffer
// are errors. Empty buffers and offset==size are valid no-output renders.
// Finite velocities retain the model's existing clamp to [0.02,1].
inline NoteDebug render_bridge_force(const Key& key, double velocity,
                                     std::int64_t hold_samples, std::vector<double>& buffer,
                                     std::ptrdiff_t offset, uint64_t seed,
                                     const Options& options = {}) {
    detail::validate_options(options);
    detail::validate_velocity(velocity);
    if (hold_samples < 0 || offset < 0 || (size_t)offset > buffer.size())
        throw std::invalid_argument("piano: hold/offset must be nonnegative and offset within the buffer");
    detail::validate_buffer(buffer.data(), buffer.size());
    int hold = (int)std::min(hold_samples, (std::int64_t)buffer.size());
    return detail::render_note(key.parameters(), velocity, hold, buffer, (int)offset, seed, options,
                               &detail::KeyAccess::strings(key));
}

inline void make_soundboard_ir(std::vector<double>& left, std::vector<double>& right,
                               uint64_t seed, const Voicing& voicing = {}) {
    validate_voicing(voicing);
    if (&left == &right)
        throw std::invalid_argument("piano: soundboard output channels must be distinct");
    detail::make_soundboard_ir(left, right, seed, voicing.soundboard);
}
// SOUNDBOARD_REGIONS stereo responses along the bridge (see detail::make_soundboard_regions).
// Supply both tail outputs or neither. All outputs must be distinct; tails
// must not refer to existing elements of either region vector. Invalid output
// combinations throw invalid_argument before any output is modified.
inline void make_soundboard_regions(std::vector<std::vector<double>>& left, std::vector<std::vector<double>>& right,
                                    uint64_t seed, const Voicing& voicing = {},
                                    std::vector<double>* tail_left = nullptr, std::vector<double>* tail_right = nullptr) {
    validate_voicing(voicing);
    if (&left == &right || (tail_left && tail_left == tail_right))
        throw std::invalid_argument("piano: soundboard output channels must be distinct");
    if ((tail_left == nullptr) != (tail_right == nullptr))
        throw std::invalid_argument("piano: soundboard tail outputs must be supplied together");
    if (tail_left) {
        for (const auto* regions : {&left, &right})
            for (const auto& region : *regions)
                if (tail_left == &region || tail_right == &region)
                    throw std::invalid_argument("piano: soundboard tails must not alias region outputs");
    }
    detail::make_soundboard_regions(left, right, seed, voicing.soundboard, tail_left, tail_right);
}

// Empty input produces empty output. At least one IR must be nonempty; an empty
// IR channel is silent. Output vectors must be distinct from all inputs and each
// other. This operation allocates and includes the complete convolution tail.
inline void convolve_stereo(const std::vector<double>& input,
                            const std::vector<double>& ir_left,
                            const std::vector<double>& ir_right,
                            std::vector<double>& left, std::vector<double>& right) {
    if (&left == &right || &left == &input || &right == &input
        || &left == &ir_left || &left == &ir_right || &right == &ir_left || &right == &ir_right)
        throw std::invalid_argument("piano: convolution outputs must not alias inputs or each other");
    detail::validate_ir(ir_left, ir_right);
    detail::validate_buffer(input.data(), input.size());
    if (input.empty()) { left.clear(); right.clear(); return; }
    detail::convolve_stereo_unchecked(input, ir_left, ir_right, left, right);
}

struct VoiceDiagnostics {
    std::int64_t samples_rendered;
    size_t active_modes;
    bool release_requested;
    bool damped;
    bool active;
};

// Storage requirements are explicit so a host can reserve its entire voice
// pool during setup. Reserving can allocate; starting a prepared note cannot.
struct VoiceCapacity {
    size_t phasors = 0, attack_samples = 0, damper_samples = 0;
    void include(VoiceCapacity other) noexcept {
        phasors = std::max(phasors, other.phasors);
        attack_samples = std::max(attack_samples, other.attack_samples);
        damper_samples = std::max(damper_samples, other.damper_samples);
    }
};

// Conservative capacity for any velocity/seed/options at this key. This is a
// setup operation; it builds string coefficients but does not solve contact.
inline VoiceCapacity voice_capacity(const Key& key) {
    const auto& k = key.parameters();
    // Lowest possible unison frequency bounds the mode and parent counts.
    auto string = detail::build_string(k, -1.15 * k.det_scale * 1.2);
    size_t parents = 0;
    // Cover the taper for fractional pitches between 200 Hz and G#3.
    if (k.f0 < detail::midi_to_freq(56.0))
        for (double w : string.w) if (w / (2 * detail::PI) < 3500.0) ++parents;
    return {(2 * string.w.size() + 8 * parents) * (size_t)k.n_strings,
            (size_t)((k.f0 > detail::midi_to_freq(68.0) ? .65 : .030) * FS), key.has_damper() ? (size_t)(.030 * FS) : 0};
}

class PreparedNote;
inline PreparedNote prepare_strike(const Key&, double, uint64_t, const Options& = {});

// Immutable, reusable initial strike state. Preparing/copying/destroying this
// object belongs off the audio thread (StrikeWorkspace supplies a reusable
// preparation alternative). Voice::start copies into reserved storage
// and retains no reference, so the prepared object can then be reused or freed.
class PreparedNote {
    detail::NoteVoiceData data_;
    bool has_damper_;
    PreparedNote(detail::NoteVoiceData data, bool damper)
        : data_(std::move(data)), has_damper_(damper) {}
    friend PreparedNote prepare_strike(const Key&, double, uint64_t, const Options&);
    friend class Voice;
    friend class StrikeWorkspace;
public:
    VoiceCapacity capacity() const noexcept {
        return {data_.ph.size(), data_.attack.size(), data_.damp_noise.size()};
    }
    bool valid() const noexcept { return !data_.ph.empty() && !data_.attack.empty(); }
    const NoteDebug& note_diagnostics() const noexcept { return data_.dbg; }
};

inline PreparedNote prepare_strike(const Key& key, double velocity, uint64_t seed,
                                   const Options& options) {
    detail::validate_options(options);
    detail::validate_velocity(velocity);
    return PreparedNote(detail::prepare_note_checked(key.parameters(), velocity, seed, options,
                                                      &detail::KeyAccess::strings(key)),
                        key.has_damper());
}

// Seed-independent hammer/string motion at one exact normalized velocity.
// Construct/copy/destroy off the audio thread. Key copies share coefficients;
// the response itself is immutable and may be read concurrently by workspaces.
class ContactResponse {
    Key key_;
    double velocity_;
    detail::ContactResult contact_;
    std::array<std::vector<double>,3> force_, q_, qd_;
    friend class StrikeWorkspace;
public:
    ContactResponse(const Key& key, double velocity) : key_(key), velocity_(velocity) {
        detail::validate_velocity(velocity);
        velocity_ = std::clamp(velocity, .02, 1.0);
        auto strings = detail::KeyAccess::strings(key_);
        std::vector<double> force;
        double hammer = key_.parameters().hammer.maximum_velocity_mps
            * std::pow(velocity_, key_.parameters().hammer.velocity_exponent);
        contact_ = detail::run_contact(key_.parameters(), strings, hammer, force, &force_);
        bool finite = std::isfinite(contact_.duration_ms) && std::isfinite(contact_.peak_force);
        for (size_t i = 0; i < strings.size(); ++i) {
            q_[i] = std::move(strings[i].q); qd_[i] = std::move(strings[i].qd);
            for (const auto* values : {&q_[i], &qd_[i], &force_[i]})
                for (double value : *values) finite &= std::isfinite(value);
        }
        if (!finite) throw std::invalid_argument("piano: unrepresentable contact response");
    }
    double velocity() const noexcept { return velocity_; }
    ContactStatus contact_status() const noexcept { return contact_.status; }
    // Allocated vector payload, excluding this object, shared key data and
    // allocator overhead. Capacity counts unused growth storage too.
    size_t storage_bytes() const noexcept {
        size_t bytes = 0;
        for (size_t i = 0; i < 3; ++i)
            bytes += (force_[i].capacity()+q_[i].capacity()+qd_[i].capacity())*sizeof(double);
        return bytes;
    }
};

// One reusable workspace per Key/options combination. Construction allocates
// and touches capacity; prepare() performs no allocation or contact integration.
// Debug builds assert capacity before vector growth to catch reservation drift.
// A single owning thread uses each workspace. The borrowed note is valid until
// the next successful prepare; Voice::start copies it into reserved voice memory.
class StrikeWorkspace {
    Key key_;
    Options options_;
    PreparedNote note_;
    std::vector<detail::StringModes> strings_;
    std::vector<double> force_;
    std::vector<detail::Parent> parents_v_, parents_h_;
    struct CachedContact {
        const ContactResponse& response;
        std::array<double,3> gains{};
        CachedContact(const ContactResponse& r, uint64_t seed) : response(r) {
            std::mt19937_64 rng{seed}; std::uniform_real_distribution<double> u(-1,1);
            for (size_t i = 0; i < detail::KeyAccess::strings(r.key_).size(); ++i)
                gains[i] = (1+r.key_.parameters().strings.horizontal_ratio)*(1+detail::unison_asymmetry(detail::KeyAccess::strings(r.key_).size(), i)+.13*u(rng));
        }
        detail::ContactResult operator()(const detail::KeyParams&,
            std::vector<detail::StringModes>& strings, double, std::vector<double>& force) const {
            assert(response.force_[0].size() <= force.capacity());
            force.assign(response.force_[0].size(), 0.0);
            for (size_t s = 0; s < strings.size(); ++s) {
                assert(response.q_[s].size() == strings[s].q.size());
                assert(response.qd_[s].size() == strings[s].qd.size());
                assert(response.force_[s].size() == force.size());
                std::copy(response.q_[s].begin(), response.q_[s].end(), strings[s].q.begin());
                std::copy(response.qd_[s].begin(), response.qd_[s].end(), strings[s].qd.begin());
                for (size_t i = 0; i < force.size(); ++i)
                    force[i] += gains[s]*response.force_[s][i];
            }
            return response.contact_;
        }
    };
public:
    explicit StrikeWorkspace(const Key& key, const Options& options = {})
        : key_(key), options_(options), note_({}, key.has_damper()),
          strings_(detail::KeyAccess::strings(key)) {
        detail::validate_options(options);
        auto capacity = voice_capacity(key);
        if (options.knock && key.parameters().f0 > detail::midi_to_freq(68.0))
            (void)detail::structural_knock_waveform();
        auto touch = [](auto& vector, size_t size) { vector.resize(size); vector.clear(); };
        touch(note_.data_.ph, capacity.phasors);
        touch(note_.data_.attack, capacity.attack_samples);
        touch(note_.data_.damp_noise, capacity.damper_samples);
        touch(force_, capacity.attack_samples);
        size_t modes = 0;
        for (const auto& string : strings_) modes = std::max(modes, string.w.size());
        touch(parents_v_, modes); touch(parents_h_, modes);
    }
    StrikeWorkspace(const StrikeWorkspace&) = delete;
    StrikeWorkspace& operator=(const StrikeWorkspace&) = delete;
    StrikeWorkspace(StrikeWorkspace&&) = delete;
    StrikeWorkspace& operator=(StrikeWorkspace&&) = delete;
    // Response must originate from this Key (or a copy of it). A mismatched key
    // leaves the previous note intact. Nonfinite synthesis returns false and
    // invalidates the result; neither failure throws on the audio thread.
    bool prepare(const ContactResponse& response, uint64_t seed) noexcept {
        if (response.force_[0].empty()
            || &detail::KeyAccess::strings(key_) != &detail::KeyAccess::strings(response.key_))
            return false;
        detail::prep_note_into(key_.parameters(), response.velocity_, seed, note_.data_,
            strings_, force_, parents_v_, parents_h_, options_, detail::NoPreparationTrace{},
            &detail::KeyAccess::strings(key_), CachedContact(response, seed), true);
        if (!detail::note_data_finite(note_.data_)) {
            note_.data_.ph.clear(); note_.data_.attack.clear(); note_.data_.damp_noise.clear();
            return false;
        }
        return true;
    }
    const PreparedNote& note() const noexcept { return note_; }
};

class Voice;
inline Voice prepare_note(const Key&, double, uint64_t, const Options& = {});

// Owns mutable playback state. Copies intentionally clone a voice's current
// state; diagnostics expose values, never writable references to that state.
class Voice {
    std::vector<detail::Phasor> ph;
    std::vector<double> attack, damp_noise;
    size_t att_pos = 0, dn_pos = 0;
    std::int64_t age = 0;
    int coff = 0;
    bool has_damper = true;
    // Damper state in samples of the 40 ms transition: damper_position is the
    // current mix (0 open .. DAMPER_TRANSITION_SAMPLES fully seated) and moves
    // one sample per sample toward damper_target. While they differ the modes
    // blend their open and damped multipliers per sample; once they agree the
    // blend is committed to step_now. A new target reverses or redirects any
    // transition from its current mix, so a pedal catch keeps the remaining
    // amplitude and phase, and a partial pedal position damps partially.
    int damper_position = 0, damper_target = 0;
    double kill_abs = 0.0;
    int until_cull = detail::CULL_INTERVAL_SAMPLES;
    NoteDebug debug_{};

    Voice(detail::NoteVoiceData data, bool damper)
        : ph(std::move(data.ph)), attack(std::move(data.attack)),
          damp_noise(std::move(data.damp_noise)), dn_pos(damp_noise.size()),
          coff(data.coff), has_damper(damper), kill_abs(data.peak_amp * detail::SILENCE_RATIO),
          debug_(data.dbg) {}
    friend Voice prepare_note(const Key&, double, uint64_t, const Options&);

    // Ramp blends the open and damped multipliers by release_mix per sample;
    // otherwise the committed step_now for the settled damper position is used.
    template<size_t N, bool Ramp, bool Rising>
    static void render_mode_group_scalar(detail::Phasor* modes, double* values,
                                  const double* onset, int length, const double* release_mix) noexcept {
        std::array<double,N> re, im, sr, si, rr, ri, weight;
        std::array<bool,N> longitudinal;
        for (size_t m = 0; m < N; ++m) {
            re[m] = modes[m].z.real(); im[m] = modes[m].z.imag();
            weight[m] = modes[m].w_out;
            if constexpr (Rising) longitudinal[m] = modes[m].longitudinal;
            const auto& settled = Ramp ? modes[m].step : modes[m].step_now;
            sr[m] = settled.real(); si[m] = settled.imag();
            if constexpr (Ramp) { rr[m] = modes[m].step_r.real(); ri[m] = modes[m].step_r.imag(); }
        }
        for (int i = 0; i < length; ++i) {
            for (size_t m = 0; m < N; ++m) {
                double value = weight[m]*re[m];
                if constexpr (Rising) value *= longitudinal[m] ? onset[i] : 1.0;
                values[i] += value;
            }
            // Independent component arrays allow vector updates without changing
            // the ordered sample sum. Finite, decaying prepared states need no
            // generic complex NaN/Inf recovery branch. Keep products separate
            // to retain std::complex rounding instead of contracting an FMA.
            double t = Ramp ? release_mix[i] : 0;
            for (size_t m = 0; m < N; ++m) {
                double mr = sr[m], mi = si[m];
                if constexpr (Ramp) {
                    double hr = sr[m]*(1.0-t), hi = si[m]*(1.0-t);
                    double dr = rr[m]*t, di = ri[m]*t;
                    mr = hr+dr; mi = hi+di;
                }
                double ac = re[m]*mr;
                double bd = im[m]*mi;
                double ad = re[m]*mi;
                double bc = im[m]*mr;
                re[m] = ac-bd; im[m] = bc+ad;
            }
        }
        for (size_t m = 0; m < N; ++m) modes[m].z = {re[m],im[m]};
    }
#if (defined(__clang__) || defined(__GNUC__)) && !defined(PIANO_DISABLE_VECTOR_RENDER)
    // Compiler vector arithmetic maps to native SIMD where available, with no
    // platform headers or framework dependency. Other compilers use the scalar
    // implementation. Two independent modes occupy the two double lanes;
    // output additions remain scalar and ordered, preserving the original sum.
    using ModePair = double __attribute__((vector_size(16)));
    template<size_t N, bool Ramp, bool Rising>
    static void render_mode_pairs(detail::Phasor* modes, double* values,
                                  const double* onset, int length, const double* release_mix) noexcept {
        static_assert(N%2 == 0, "mode pairs need an even group");
        std::array<ModePair,N/2> re, im, sr, si, rr, ri, weight;
        std::array<std::array<bool,2>,N/2> longitudinal;
        for (size_t p = 0; p < N/2; ++p) {
            const auto& a = modes[2*p]; const auto& b = modes[2*p+1];
            re[p] = ModePair{a.z.real(),b.z.real()}; im[p] = ModePair{a.z.imag(),b.z.imag()};
            const auto& sa = Ramp ? a.step : a.step_now;
            const auto& sb = Ramp ? b.step : b.step_now;
            sr[p] = ModePair{sa.real(),sb.real()}; si[p] = ModePair{sa.imag(),sb.imag()};
            weight[p] = ModePair{a.w_out,b.w_out};
            if constexpr (Ramp) {
                rr[p] = ModePair{a.step_r.real(),b.step_r.real()};
                ri[p] = ModePair{a.step_r.imag(),b.step_r.imag()};
            }
            if constexpr (Rising) longitudinal[p] = {a.longitudinal,b.longitudinal};
        }
        for (int i = 0; i < length; ++i) {
            double value = values[i];
            for (size_t p = 0; p < N/2; ++p) {
                ModePair sample = weight[p]*re[p];
                if constexpr (Rising) {
                    ModePair gain{longitudinal[p][0] ? onset[i] : 1.0,
                                  longitudinal[p][1] ? onset[i] : 1.0};
                    sample *= gain;
                }
                value += sample[0]; value += sample[1];
            }
            values[i] = value;
            for (size_t p = 0; p < N/2; ++p) {
                ModePair mr = sr[p], mi = si[p];
                if constexpr (Ramp) {
                    ModePair t{release_mix[i],release_mix[i]};
                    ModePair hold{1.0-release_mix[i],1.0-release_mix[i]};
                    ModePair hr = sr[p]*hold, hi = si[p]*hold;
                    ModePair dr = rr[p]*t, di = ri[p]*t;
                    mr = hr+dr; mi = hi+di;
                }
                // Separate products preserve the unfused complex arithmetic.
                ModePair ac = re[p]*mr, bd = im[p]*mi;
                ModePair ad = re[p]*mi, bc = im[p]*mr;
                re[p] = ac-bd; im[p] = bc+ad;
            }
        }
        for (size_t p = 0; p < N/2; ++p) {
            modes[2*p].z = {re[p][0],im[p][0]};
            modes[2*p+1].z = {re[p][1],im[p][1]};
        }
    }
#endif
    template<size_t N, bool Ramp, bool Rising>
    static void render_mode_group(detail::Phasor* modes, double* values,
                                  const double* onset, int length, const double* release_mix) noexcept {
#if (defined(__clang__) || defined(__GNUC__)) && !defined(PIANO_DISABLE_VECTOR_RENDER)
        if constexpr (N%2 == 0) {
            render_mode_pairs<N,Ramp,Rising>(modes,values,onset,length,release_mix);
            return;
        }
#endif
        render_mode_group_scalar<N,Ramp,Rising>(modes,values,onset,length,release_mix);
    }
    template<bool Ramp, bool Rising>
    void render_modes(double* values, const double* onset, int length, const double* release_mix) noexcept {
        constexpr size_t group = Ramp ? 4 : 8;
        size_t mode = 0;
        for (; mode+group <= ph.size(); mode += group)
            render_mode_group<group,Ramp,Rising>(ph.data()+mode,values,onset,length,release_mix);
        for (; mode+2 <= ph.size(); mode += 2)
            render_mode_group<2,Ramp,Rising>(ph.data()+mode,values,onset,length,release_mix);
        for (; mode < ph.size(); ++mode)
            render_mode_group<1,Ramp,Rising>(ph.data()+mode,values,onset,length,release_mix);
    }
    // Commit the blend for the settled damper position, with the same
    // arithmetic the ramp uses per sample (exact at both ends).
    void commit_damper() noexcept {
        const double t = (double)damper_position / detail::DAMPER_TRANSITION_SAMPLES;
        for (auto& mode : ph) mode.step_now = mode.step * (1.0 - t) + mode.step_r * t;
    }

    void render_samples(double* force, int count) noexcept {
        // Visit each mode for a short span, keeping its state in registers.
        // Values still accumulate in the original mode order for every sample.
        // Bound spans at lifecycle transitions and leave culling to the caller.
        constexpr int tile_size = 64;
        int position = 0;
        while (position < count) {
            // The damper moves toward its target once the hammer has separated.
            // The damper noise plays whenever contact begins from fully open.
            const bool moving = age >= detail::release_sample(0, coff) && damper_position != damper_target;
            const int direction = damper_target > damper_position ? 1 : -1;
            if (moving && damper_position == 0) dn_pos = 0;
            int length = std::min(count-position, tile_size);
            bool free = age >= coff;
            bool rising = free && age-coff < detail::LONGITUDINAL_RISE_SAMPLES;
            if (!free) length = (int)std::min<std::int64_t>(length, coff-age);
            else if (rising) length = (int)std::min<std::int64_t>(length,
                detail::LONGITUDINAL_RISE_SAMPLES-(age-coff));
            if (moving) length = std::min(length, std::abs(damper_target-damper_position));

            double values[tile_size]{}, onset[tile_size], release_mix[tile_size];
            for (int i = 0; i < length; ++i) {
                if (att_pos < attack.size()) values[i] += attack[att_pos++];
                if (rising) onset[i] = detail::longitudinal_onset(age-coff+i);
                if (moving) release_mix[i] = (double)(damper_position+direction*i)/detail::DAMPER_TRANSITION_SAMPLES;
            }
            if (free) {
                if (moving) {
                    if (rising) render_modes<true,true>(values,onset,length,release_mix);
                    else render_modes<true,false>(values,onset,length,release_mix);
                    damper_position += direction*length;
                    if (damper_position == damper_target) commit_damper();
                } else {
                    if (rising) render_modes<false,true>(values,onset,length,release_mix);
                    else render_modes<false,false>(values,onset,length,release_mix);
                }
            }
            for (int i = 0; i < length; ++i) {
                if (dn_pos < damp_noise.size()) values[i] += damp_noise[dn_pos++];
                force[position+i] += values[i];
            }
            age += length; position += length;
        }
    }

public:
    Voice() = default; // empty, inactive, with no reserved storage
    // Setup only. Existing playback is preserved, including on partial failure.
    void reserve(VoiceCapacity capacity) {
        auto reserve_and_touch = [](auto& storage, size_t count) {
            size_t size = storage.size();
            storage.reserve(count);
            // Initialize the reserved region during setup, not on the first
            // note-on. Keep existing playback elements and size unchanged.
            storage.resize(std::max(size,count));
            storage.resize(size);
        };
        reserve_and_touch(ph, capacity.phasors);
        reserve_and_touch(attack, capacity.attack_samples);
        reserve_and_touch(damp_noise, capacity.damper_samples);
    }
    bool can_start(const PreparedNote& note) const noexcept {
        auto required = note.capacity();
        return note.valid() && ph.capacity() >= required.phasors
            && attack.capacity() >= required.attack_samples
            && damp_noise.capacity() >= required.damper_samples;
    }
    // False leaves this voice unchanged (invalid/moved-from note or insufficient
    // capacity). Success performs bounded copies only: no allocation or solver.
    bool start(const PreparedNote& note) noexcept {
        if (!can_start(note)) return false;
        const auto& data = note.data_;
        ph.resize(data.ph.size());
        attack.resize(data.attack.size());
        damp_noise.resize(data.damp_noise.size());
        std::copy(data.ph.begin(), data.ph.end(), ph.begin());
        std::copy(data.attack.begin(), data.attack.end(), attack.begin());
        std::copy(data.damp_noise.begin(), data.damp_noise.end(), damp_noise.begin());
        att_pos = 0; dn_pos = damp_noise.size(); age = 0;
        coff = data.coff; has_damper = note.has_damper_;
        damper_position = damper_target = 0;
        kill_abs = data.peak_amp * detail::SILENCE_RATIO;
        until_cull = detail::CULL_INTERVAL_SAMPLES;
        debug_ = data.dbg;
        return true;
    }
    // Immediate silence for reset/transport handling; retains all capacity.
    void stop() noexcept {
        ph.clear(); attack.clear(); damp_noise.clear();
        att_pos = dn_pos = 0; age = 0; coff = 0;
        damper_position = damper_target = 0;
        kill_abs = 0; until_cull = detail::CULL_INTERVAL_SAMPLES; debug_ = {};
    }
    const NoteDebug& note_diagnostics() const noexcept { return debug_; }
    int contact_samples() const noexcept { return coff; }
    bool active() const noexcept {
        return !(ph.empty() && att_pos >= attack.size() && dn_pos >= damp_noise.size());
    }
    VoiceDiagnostics diagnostics() const noexcept {
        return {age, ph.size(), damper_target == detail::DAMPER_TRANSITION_SAMPLES,
                damper_position == detail::DAMPER_TRANSITION_SAMPLES && damper_target == damper_position, active()};
    }
    // Requested damper contact, 0 (strings open) to 1 (damper fully seated);
    // partial values model a half-pedal. The damper moves there over up to
    // 40 ms from wherever it is, so a catch (back toward 0) keeps the remaining
    // amplitude and phase and restores no lost energy, and nothing restarts.
    // A request during hammer contact takes effect at separation. Repeated
    // requests are harmless; keys without dampers ignore them.
    void damper(double contact) noexcept {
        if (has_damper)
            damper_target = (int)std::lround(std::clamp(contact, 0.0, 1.0) * detail::DAMPER_TRANSITION_SAMPLES);
    }
    void release() noexcept { damper(1.0); }
    void undamp() noexcept { damper(0.0); }

    // Add to a borrowed buffer. An empty view is a strict no-op. Culling occurs
    // every 8192 elapsed samples, including inside a large caller buffer.
    bool render_bridge_force(AudioBuffer buffer) {
        detail::validate_buffer(buffer.data, buffer.size);
        if (buffer.size > (uint64_t)(std::numeric_limits<std::int64_t>::max() - age))
            throw std::length_error("piano: voice sample clock overflow");
        size_t position = 0;
        while (position < buffer.size) {
            int count = (int)std::min(buffer.size - position, (size_t)until_cull);
            render_samples(buffer.data + position, count);
            position += count;
            until_cull -= count;
            if (until_cull == 0) {
                ph.erase(std::remove_if(ph.begin(), ph.end(), [&](const detail::Phasor& mode) {
                    return detail::below_threshold(mode, kill_abs);
                }), ph.end());
                until_cull = detail::CULL_INTERVAL_SAMPLES;
            }
        }
        return active();
    }
};

inline Voice prepare_note(const Key& key, double velocity, uint64_t seed,
                           const Options& options) {
    detail::validate_options(options);
    detail::validate_velocity(velocity);
    return Voice(detail::prepare_note_checked(key.parameters(), velocity, seed, options,
                                                      &detail::KeyAccess::strings(key)), key.has_damper());
}

// A feed-forward approximation of the unstruck string register. Construct once
// per instrument, mix all voices first, then process before the soundboard.
// This does not change hammer contact, implement restrikes, or feed energy back
// into the primary voices. The initial voicing is physical-parameter-derived,
// not fitted to measured pedal impulse responses.
class SympatheticResonance {
    struct Mode {
        double re = 0.0, im = 0.0;
        double cosine, sine, open_radius, closed_radius, drive;
    };
    struct StringGroup {
        size_t begin = 0, end = 0;
        bool held = false, undamped = false;
        double openness = 0.0;
    };
    std::vector<Mode> modes_;
    std::array<StringGroup, 88> strings_{};
    double gain_;
    double ramp_;
    double own_drive_ = 1.0;
    double pedal_lift_ = 0.0; // 0 dampers seated .. 1 fully lifted

    static size_t own_extent(size_t samples, size_t stride) {
        if (!samples) return 0;
        if (stride < samples)
            throw std::invalid_argument("piano: own-key stride must cover the input block");
        constexpr size_t maximum = (size_t)std::numeric_limits<std::ptrdiff_t>::max() / sizeof(double);
        if (samples > maximum || stride > (maximum - samples) / 87)
            throw std::length_error("piano: own-key buffer span is too large");
        return 87 * stride + samples;
    }

public:
    explicit SympatheticResonance(const ResonanceOptions& options = {})
        : SympatheticResonance([&] { Voicing v; v.resonance = options; return v; }()) {}
    explicit SympatheticResonance(const Voicing& voicing)
        : gain_(voicing.resonance.gain), ramp_(0.0), own_drive_(voicing.resonance.own_drive) {
        validate_voicing(voicing);
        const auto& options = voicing.resonance;
        ramp_ = 1.0 / (options.transition_seconds * FS);
        for (size_t key = 0; key < strings_.size(); ++key) {
            auto k = detail::make_key_params(21 + key, voicing);
            auto& group = strings_[key];
            group.begin = modes_.size();
            group.undamped = !k.has_damper;
            group.openness = group.undamped ? 1.0 : 0.0;
            // A reduced set of transverse modes, with the same stretch and
            // stiffness as the primary strings. Higher modes contribute less.
            for (int partial = 1; partial <= 16; ++partial) {
                double f = partial * k.f0 * std::sqrt(1.0 + k.B * partial * partial);
                if (f > 7000.0) break;
                double sigma = std::log(1000.0) / std::min(k.T60_1, options.maximum_decay_seconds)
                             + k.c2 * (f * f - k.f0 * k.f0);
                double open = std::exp(-sigma / FS);
                double closed = std::exp(-std::max(sigma, std::log(1000.0) / options.damper_decay_seconds) / FS);
                double angle = 2.0 * detail::PI * f / FS;
                // Normalise each resonance's near-centre sinusoidal gain;
                // otherwise long bass decays also imply huge gain peaks.
                double weight = 1.0 / (partial * (1.0 + std::pow(f / 3500.0, 2.0)));
                modes_.push_back({0.0, 0.0, std::cos(angle), std::sin(angle),
                                  open, closed, 2.0 * (1.0 - open) * weight});
            }
            group.end = modes_.size();
        }
    }

    SympatheticResonance(const SympatheticResonance&) = default;
    SympatheticResonance& operator=(const SympatheticResonance&) = default;
    // Copies clone playback state. Moves leave a valid, empty bypass instance.
    SympatheticResonance(SympatheticResonance&& other) noexcept
        : modes_(std::move(other.modes_)), strings_(other.strings_),
          gain_(other.gain_), ramp_(other.ramp_), own_drive_(other.own_drive_), pedal_lift_(other.pedal_lift_) {
        other.modes_.clear(); other.strings_ = {}; other.gain_ = 0.0; other.pedal_lift_ = 0.0;
    }
    SympatheticResonance& operator=(SympatheticResonance&& other) noexcept {
        if (this != &other) {
            modes_ = std::move(other.modes_); strings_ = other.strings_;
            gain_ = other.gain_; ramp_ = other.ramp_; own_drive_ = other.own_drive_; pedal_lift_ = other.pedal_lift_;
            other.modes_.clear(); other.strings_ = {}; other.gain_ = 0.0; other.pedal_lift_ = 0.0;
        }
        return *this;
    }

    size_t mode_count() const noexcept { return modes_.size(); }
    // Nominal 80 dB free-decay allowance after the last input sample. Supply
    // the bank's options when using a custom cap (default T60=12 s).
    static double tail_seconds(const ResonanceOptions& options = {}) {
        Voicing v; v.resonance = options; validate_voicing(v);
        return options.maximum_decay_seconds * (4.0 / 3.0);
    }
    void set_pedal(bool down) noexcept { pedal_lift_ = down ? 1.0 : 0.0; }
    // Continuous pedal: how far the dampers are lifted, 0..1 (see damper_contact).
    void set_pedal_lift(double lift) noexcept { pedal_lift_ = std::isfinite(lift) ? std::clamp(lift, 0.0, 1.0) : 0.0; }
    // The host aggregates overlapping notes/channels into a held-key state.
    void set_key_held(int midi, bool held) {
        if (midi < 21 || midi > 108)
            throw std::invalid_argument("piano: resonance key must be in [21,108]");
        strings_[(size_t)(midi - 21)].held = held;
    }
    // Reset vibration AND controls to pedal up / no keys held, without allocation.
    void reset() noexcept {
        pedal_lift_ = 0.0;
        for (auto& group : strings_) {
            group.held = false;
            group.openness = group.undamped ? 1.0 : 0.0;
        }
        for (auto& mode : modes_) mode.re = mode.im = 0.0;
    }
    // In-place dry + wet bridge force; arbitrary buffer divisions are equivalent.
    // Finite audio is a caller precondition, as for the other realtime APIs.
    // own_by_key optionally gives each physical key's own contribution to the
    // force as 88 rows of own_stride samples (row k for MIDI key 21+k); with
    // own_drive below 1 that share is withheld from the key's own resonators.
    // A supplied stride must cover force.size; its address range must be
    // representable. Raw pointers still require caller-owned storage through
    // sample 87*own_stride + force.size-1. Use the sized overload to check it.
    // Invalid layouts are rejected before output or resonance state changes.
    // Layout checks apply even when gain or own-key withholding is disabled.
    void process(AudioBuffer force, const double* own_by_key = nullptr, size_t own_stride = 0) {
        detail::validate_buffer(force.data, force.size);
        if (own_by_key) (void)own_extent(force.size, own_stride);
        if (gain_ == 0.0) return;
        const bool withhold = own_by_key && own_drive_ < 1.0;
        const double withheld = 1.0 - own_drive_;
        for (size_t i = 0; i < force.size; ++i) {
            const double input = force.data[i]; // wet output never feeds the bank
            double wet = 0.0;
            for (size_t g = 0; g < strings_.size(); ++g) {
                auto& group = strings_[g];
                const double drive_input = withhold ? input - withheld * own_by_key[g * own_stride + i] : input;
                const double target = group.held || group.undamped ? 1.0 : pedal_lift_;
                if (target >= 1.0) group.openness = std::min(1.0, group.openness + ramp_);
                else if (target <= 0.0) group.openness = std::max(0.0, group.openness - ramp_);
                else if (group.openness < target) group.openness = std::min(target, group.openness + ramp_);
                else group.openness = std::max(target, group.openness - ramp_);
                for (size_t m = group.begin; m < group.end; ++m) {
                    auto& mode = modes_[m];
                    double radius = mode.closed_radius
                                  + group.openness * (mode.open_radius - mode.closed_radius);
                    double re = radius * (mode.cosine * mode.re - mode.sine * mode.im)
                              + mode.drive * group.openness * drive_input;
                    mode.im = radius * (mode.sine * mode.re + mode.cosine * mode.im);
                    mode.re = re;
                    if (std::abs(mode.re) + std::abs(mode.im) < 1e-30)
                        mode.re = mode.im = 0.0;
                    wet += mode.re;
                }
            }
            force.data[i] = input + gain_ * wet;
        }
    }

    // Checked storage extent, including sub-block pointers into padded rows.
    // The rows must remain stable for the call and must not overlap force.
    void process(AudioBuffer force, ConstAudioBuffer own_by_key, size_t own_stride) {
        detail::validate_buffer(force.data, force.size);
        if (!force.size) return;
        detail::validate_buffer(own_by_key.data, own_by_key.size);
        if (own_by_key.data && own_by_key.size < own_extent(force.size, own_stride))
            throw std::invalid_argument("piano: own-key buffer is shorter than its row layout");
        process(force, own_by_key.data, own_stride);
    }
};

class PartitionedConvolver {
    int P = 0;
    std::vector<std::vector<std::complex<double>>> H[2], X;
    std::vector<std::complex<double>> Y[2], A;
    std::vector<double> prev;
    int head = 0;

    void swap(PartitionedConvolver& other) noexcept {
        std::swap(P, other.P); std::swap(head, other.head);
        std::swap(H, other.H); X.swap(other.X);
        std::swap(Y, other.Y); A.swap(other.A); prev.swap(other.prev);
    }

    void initialize_unchecked(const std::vector<double>& irL, const std::vector<double>& irR) {
        size_t n = std::max(irL.size(), irR.size());
        P = (int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        const std::vector<double>* irs[2] = {&irL, &irR};
        for (int c = 0; c < 2; ++c) {
            H[c].assign(P, {});
            for (int p = 0; p < P; ++p) {
                std::vector<std::complex<double>> h(2 * BLOCK_SIZE, 0.0);
                for (int i = 0; i < BLOCK_SIZE; ++i) {
                    size_t s = (size_t)p * BLOCK_SIZE + i;
                    if (s < irs[c]->size()) h[i] = (*irs[c])[s];
                }
                detail::fft(h, false);
                H[c][p] = std::move(h);
            }
            Y[c].assign(2 * BLOCK_SIZE, 0.0);
        }
        X.assign(P, std::vector<std::complex<double>>(2 * BLOCK_SIZE, 0.0));
        A.assign(2 * BLOCK_SIZE, 0.0);
        prev.assign(BLOCK_SIZE, 0.0);
    }

public:
    PartitionedConvolver() = default;
    PartitionedConvolver(const PartitionedConvolver&) = delete;
    PartitionedConvolver& operator=(const PartitionedConvolver&) = delete;
    PartitionedConvolver(PartitionedConvolver&& other) noexcept { swap(other); }
    PartitionedConvolver& operator=(PartitionedConvolver&& other) noexcept {
        if (this != &other) {
            PartitionedConvolver replacement(std::move(other));
            swap(replacement);
        }
        return *this;
    }

    bool ready() const noexcept { return P != 0; }
    size_t partition_count() const noexcept { return (size_t)P; }

    // Build separately so failed validation/allocation preserves the old setup.
    void init(const std::vector<double>& left, const std::vector<double>& right) {
        detail::validate_ir(left, right);
        PartitionedConvolver replacement;
        replacement.initialize_unchecked(left, right);
        swap(replacement);
    }

    // Clear the complete playback history, retaining the IR and all allocation.
    void reset() noexcept {
        head = 0;
        for (auto& block : X) std::fill(block.begin(), block.end(), std::complex<double>{});
        for (auto& channel : Y) std::fill(channel.begin(), channel.end(), std::complex<double>{});
        std::fill(A.begin(), A.end(), std::complex<double>{});
        std::fill(prev.begin(), prev.end(), 0.0);
    }

    // Exactly BLOCK_SIZE samples in each view. Input may alias either output;
    // the two output views must not overlap. Checks do not scan audio samples.
    void process(ConstAudioBuffer input, AudioBuffer left, AudioBuffer right) {
        if (!ready()) throw std::logic_error("piano: convolver has not been initialized");
        if (input.size != BLOCK_SIZE || left.size != BLOCK_SIZE || right.size != BLOCK_SIZE)
            throw std::invalid_argument("piano: convolution requires BLOCK_SIZE samples per channel");
        detail::validate_buffer(input.data, input.size);
        detail::validate_buffer(left.data, left.size);
        detail::validate_buffer(right.data, right.size);
        if (detail::overlaps(left, right))
            throw std::invalid_argument("piano: convolution output buffers must not overlap");
        const double* in = input.data;
        double* outL = left.data;
        double* outR = right.data;
        for (int i = 0; i < BLOCK_SIZE; ++i) { A[i] = prev[i]; A[BLOCK_SIZE + i] = in[i]; }
        std::copy(in, in + BLOCK_SIZE, prev.begin());
        detail::fft(A, false);
        X[head] = A;
        std::fill(Y[0].begin(), Y[0].end(), std::complex<double>(0.0));
        std::fill(Y[1].begin(), Y[1].end(), std::complex<double>(0.0));
        for (int p = 0; p < P; ++p) {
            const auto& x = X[(head - p + P) % P];
            const auto& hL = H[0][p];
            const auto& hR = H[1][p];
            for (int i = 0; i < 2 * BLOCK_SIZE; ++i) {
                Y[0][i] += x[i] * hL[i];
                Y[1][i] += x[i] * hR[i];
            }
        }
        detail::fft(Y[0], true); detail::fft(Y[1], true);
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            outL[i] = Y[0][BLOCK_SIZE + i].real();
            outR[i] = Y[1][BLOCK_SIZE + i].real();
        }
        head = (head + 1) % P;
    }
};

// Shared DC filter state for whole-buffer and incremental host output.
class DcBlocker {
    double previous_input_ = 0.0, previous_output_ = 0.0;
public:
    void reset() noexcept { previous_input_ = previous_output_ = 0.0; }
    double process(double input) noexcept {
        constexpr double coefficient = 1.0 - 2 * detail::PI * 20.0 / FS;
        double output = input - previous_input_ + coefficient * previous_output_;
        previous_input_ = input;
        previous_output_ = output;
        return output;
    }
};

inline void dc_block(std::vector<double>& samples) noexcept {
    DcBlocker filter;
    for (double& sample : samples) sample = filter.process(sample);
}

inline void fade_tail(std::vector<double>& samples, double seconds) {
    if (!std::isfinite(seconds) || seconds < 0.0)
        throw std::invalid_argument("piano: fade duration must be finite and nonnegative");
    size_t count = seconds >= (double)samples.size() / FS
        ? samples.size() : (size_t)(seconds * FS);
    size_t start = samples.size() - count;
    for (size_t i = 0; i < count; ++i)
        samples[start + i] *= 0.5 * (1.0 + std::cos(detail::PI * i / (double)count));
}

inline double limit_output(double x) {
    // Preserve normal playing exactly; only round off exceptional peaks.
    // An always-on tanh colored hard chords even below digital full scale.
    constexpr double knee = 0.8;
    double a = std::abs(x);
    if (a <= knee) return x;
    return std::copysign(knee + (1.0 - knee) * std::tanh((a - knee) / (1.0 - knee)), x);
}

} // namespace piano

#endif // SYNTHPIANO_PIANO_HPP
