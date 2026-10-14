// Host-side preset I/O. The synthesis library remains independent of files.
#ifndef SYNTHPIANO_PRESET_HPP
#define SYNTHPIANO_PRESET_HPP
#include "piano.hpp"
#include "reverb.hpp"
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>

namespace preset {
// The host splits instrument settings from room effects. Only voicing crosses
// the piano.hpp API boundary; room settings never enter the synthesis library.
struct Preset {
    piano::Voicing voicing;
    room::Options reverb;
};
inline void validate(const Preset& p) {
    piano::validate_voicing(p.voicing);
    room::validate_options(p.reverb);
}
struct Setting {
    const char* section;
    const char* name;
    double* value;
    const char* description;
};

inline std::vector<Setting> settings(piano::Voicing& v) {
    return {
        {"hammer", "stiffness_scale", &v.hammer.stiffness_scale, "Felt stiffness curve multiplier [0.25,4]."},
        {"hammer", "mass_scale", &v.hammer.mass_scale, "Hammer mass curve multiplier [0.5,2]."},
        {"hammer", "felt_width_scale", &v.hammer.felt_width_scale, "Contact width curve multiplier [0.5,2]."},
        {"hammer", "maximum_velocity_mps", &v.hammer.maximum_velocity_mps, "Hammer speed at velocity 1, metres/second [2,8]."},
        {"hammer", "velocity_exponent", &v.hammer.velocity_exponent, "Speed = maximum_velocity_mps * velocity^exponent [0.75,2.5]."},
        {"strings", "decay_scale", &v.strings.decay_scale, "Late fundamental T60 curve multiplier [0.25,2]."},
        {"strings", "treble_decay_scale", &v.strings.treble_decay_scale, "Natural treble decay multiplier [0.1,2]; blends from C6 to F#6, full above."},
        {"strings", "treble_unison_detune", &v.strings.treble_unison_detune, "Minimum unison detune [0,3]; blends from G5 to C6, full above; 0.875 puts the outer strings about 1.6 cents apart."},
        {"strings", "partial_loss_scale", &v.strings.partial_loss_scale, "Quadratic frequency-dependent loss multiplier [0.25,4]."},
        {"strings", "inharmonicity_scale", &v.strings.inharmonicity_scale, "String stiffness/inharmonicity curve multiplier [0,3]."},
        {"strings", "unison_detune_scale", &v.strings.unison_detune_scale, "Unison detuning curve multiplier [0,3]."},
        {"strings", "fast_decay_scale", &v.strings.fast_decay_scale, "Fast/slow decay-ratio curve multiplier [0.75,2], resulting ratio >=1."},
        {"strings", "horizontal_ratio", &v.strings.horizontal_ratio, "Slow/fast polarization amplitude ratio [0,1]."},
        {"strings", "longitudinal_gain", &v.strings.longitudinal_gain, "Quadratic longitudinal excitation gain, per newton [0,0.5]."},
        {"strings", "bridge_coupling", &v.strings.bridge_coupling, "String-soundboard admittance coupling: per-partial decay and detuning [0,3]."},
        {"tone", "fundamental_boost_db", &v.tone.fundamental_boost_db, "Output boost in dB [-6,12], strongest A2-A3 and tapered to zero A1/A4."},
        {"tone", "lower_overtone_cut_db", &v.tone.lower_overtone_cut_db, "Output cut in dB [0,12], same register taper, overtones around 220-1000 Hz."},
        {"tone", "bridge_curve_scale", &v.tone.bridge_curve_scale, "Existing bridge EQ depth multiplier [0,2]; register taper retained."},
        {"tone", "mid_gain_db", &v.tone.mid_gain_db, "Additional bell-shaped bridge EQ in dB [-9,9], across all keys."},
        {"tone", "mid_frequency_hz", &v.tone.mid_frequency_hz, "Bell centre in Hz [200,5000]."},
        {"tone", "mid_width_octaves", &v.tone.mid_width_octaves, "Gaussian bell standard deviation in octaves [0.2,2]."},
        {"soundboard", "bass_corner_hz", &v.soundboard.bass_corner_hz, "Bass radiation corner in Hz [30,250]; lower retains more bass."},
        {"soundboard", "treble_corner_hz", &v.soundboard.treble_corner_hz, "Treble radiation corner in Hz [1500,10000]."},
        {"soundboard", "treble_slope", &v.soundboard.treble_slope, "Treble rolloff exponent [0.5,3]."},
        {"soundboard", "decay_scale", &v.soundboard.decay_scale, "Board modal decay and impulse length multiplier [0.25,2]."},
        {"soundboard", "low_mode_density", &v.soundboard.low_mode_density, "Low board mode spacing [0,1]: 0 sparse, 1 dense; intermediate values rebuild the modal layout."},
        {"soundboard", "ripple_fraction", &v.soundboard.ripple_fraction, "Fraction of narrow modal ripple retained [0,1]."},
        {"soundboard", "spread", &v.soundboard.spread, "Stereo spread of bridge regions [0,1]; 0 centres everything."},
        {"soundboard", "position_depth", &v.soundboard.position_depth, "How much a region's bridge position colours the board [0,1]; 0 keeps one shared response."},
        {"resonance", "gain", &v.resonance.gain, "Sympathetic coupling gain [0,1]; zero bypasses the bank."},
        {"resonance", "maximum_decay_seconds", &v.resonance.maximum_decay_seconds, "Open sympathetic T60 cap in seconds [1,30]."},
        {"resonance", "damper_decay_seconds", &v.resonance.damper_decay_seconds, "Closed sympathetic T60 in seconds [0.03,2]."},
        {"resonance", "transition_seconds", &v.resonance.transition_seconds, "Sympathetic damper transition in seconds [0.001,0.2]."},
        {"resonance", "own_drive", &v.resonance.own_drive, "Share of a key's own force driving its own resonators [0,1]; 0 avoids a nominal-pitch copy beating against the unisons."},
    };
}

inline std::vector<Setting> settings(Preset& p) {
    auto fields = settings(p.voicing);
    fields.insert(fields.end(), {
        {"reverb", "mix", &p.reverb.mix, "Wet level relative to unchanged dry audio [0,1]; zero is exact bypass."},
        {"reverb", "decay_seconds", &p.reverb.decay_seconds, "Nominal low-frequency room T60 in seconds [0.2,4]."},
        {"reverb", "predelay_ms", &p.reverb.predelay_ms, "Delay before the room response, milliseconds [0,80]."},
        {"reverb", "damping_hz", &p.reverb.damping_hz, "Feedback lowpass cutoff in Hz [500,12000]; lower gives darker tails."},
    });
    return fields;
}

inline std::string trim(const std::string& s) {
    auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    return s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}
inline double number(const std::string& s) {
    size_t used = 0;
    double value = std::stod(s, &used);
    if (used != s.size() || !std::isfinite(value))
        throw std::invalid_argument("expected a finite number, got '" + s + "'");
    return value;
}

inline Preset read(std::istream& input, const std::string& source = "preset") {
    Preset result;
    auto fields = settings(result);
    std::set<std::string> seen;
    std::string line, section;
    size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        try {
            line = trim(line.substr(0, line.find_first_of("#;")));
            if (line.empty()) continue;
            if (line.front() == '[') {
                if (line.back() != ']') throw std::invalid_argument("expected [section]");
                section = trim(line.substr(1, line.size() - 2));
                bool known = false;
                for (const auto& field : fields) known |= section == field.section;
                if (!known) throw std::invalid_argument("unknown section '" + section + "'");
                continue;
            }
            auto equals = line.find('=');
            if (equals == std::string::npos) throw std::invalid_argument("expected name = value");
            std::string name = trim(line.substr(0, equals));
            std::string full = section + "." + name;
            if (!seen.insert(full).second) throw std::invalid_argument("duplicate setting '" + full + "'");
            Setting* match = nullptr;
            for (auto& field : fields)
                if (section == field.section && name == field.name) match = &field;
            if (!match) throw std::invalid_argument("unknown setting '" + full + "'");
            *match->value = number(trim(line.substr(equals + 1)));
            validate(result);
        } catch (const std::exception& e) {
            throw std::invalid_argument(source + ":" + std::to_string(line_number) + ": " + e.what());
        }
    }
    if (input.bad()) throw std::runtime_error("could not read " + source);
    return result;
}
inline Preset load(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open preset '" + path + "'");
    return read(input, path);
}
inline void write_fields(std::ostream& output, const std::vector<Setting>& fields) {
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    std::string section;
    for (const auto& field : fields) {
        if (section != field.section) {
            section = field.section;
            output << "\n[" << section << "]\n";
        }
        output << "# " << field.description << '\n' << field.name << " = " << *field.value << '\n';
    }
    if (!output) throw std::runtime_error("could not write preset");
}
inline void write(std::ostream& output, piano::Voicing voicing) {
    piano::validate_voicing(voicing);
    write_fields(output, settings(voicing));
}
inline void write(std::ostream& output, Preset p) {
    validate(p);
    write_fields(output, settings(p));
}
} // namespace preset
#endif
