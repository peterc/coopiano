# coopiano - physically modelled piano synthesizer

https://github.com/user-attachments/assets/aa7c3835-e664-40b2-91cd-c30ee46f2247

A physically modelled piano synthesizer in C++17. It renders notes and MIDI
files to stereo WAV and supports live MIDI input on macOS. Sound is synthesized
from a hammer, string, soundboard and sympathetic-resonance model, with a room
reverb effect. No piano samples, downloads or third-party libraries are required.

> [!NOTE]
> It is VERY EARLY days for releasing this code. The piano is fine, but I have a prototyped AU and VST plugin, etc. which are not part of this open release so far.

## Build

You need a C++17 compiler and Make. On macOS, install the Xcode Command Line
Tools (`xcode-select --install`) if they are not already available, then run:

```sh
make
./coopiano --help
```

macOS builds link the system CoreMIDI, AudioToolbox, CoreFoundation and CoreAudio
frameworks. The source also supports offline rendering on other platforms with
a C++17 compiler and threads; live MIDI/audio is macOS-only. This extraction was
built and tested on macOS; other platforms have not been verified here.

Compiler settings can be overridden, for example `make CXX=clang++ CXXFLAGS=-O3`.
Run `make clean` before changing compiler settings. `make clean` removes only the
executable, preserving any renders and presets.

## Run

```sh
# Note, velocity, duration, optional key-hold time (seconds)
./coopiano A4 0.5 8
# Writes A4_gp.wav; quote sharp note names, e.g. 'G#3'.

# Built-in musical demo; writes demo_piano.wav
./coopiano demo

# Render your own MIDI file; no MIDI files are bundled
./coopiano performance.mid 1.0 performance.wav

# Play from connected MIDI sources or the virtual coopiano MIDI destination
./coopiano live
# macOS only; Ctrl-C to stop
```

Running `./coopiano` without arguments renders A4 at velocity 0.5 for eight
seconds. Offline commands write 44.1 kHz, 16-bit stereo WAV files; they do not
play them automatically. On macOS, use `afplay A4_gp.wav` to listen.
Each offline render is peak-normalized independently.

## Presets

Built-in defaults work without any configuration file. `default.ini` documents
the available voicing and reverb controls, including their ranges. Copy it,
edit the copy, then select it explicitly:

```sh
cp default.ini my-piano.ini
./coopiano --preset my-piano.ini live
./coopiano --preset my-piano.ini audition --output audition.wav
```

The audition command plays a chord at increasing velocities. Use `--help` for
its chord, velocity and timing options. Editing `default.ini` alone does not
change the built-in sound; pass `--preset` and restart playback to load changes.

```sh
GP_NO_REVERB=1 ./coopiano A4 0.5 8
GP_SAVE_SETTINGS=1 ./coopiano --preset my-piano.ini A4 0.5 8
```

`GP_SAVE_SETTINGS=1` writes a resolved preset beside the WAV as `A4_gp.wav.ini`.
Environment overrides take precedence over preset settings.

## Source layout

- `coopiano.cpp`: CLI, MIDI file parser, WAV writer and macOS live host.
- `piano.hpp`: header-only synthesis library in namespace `piano`.
- `instrument.hpp`: voice pool, note/pedal scheduling and instrument effects.
- `streaming_soundboard.hpp`: streaming soundboard convolution.
- `preset.hpp` and `reverb.hpp`: INI configuration and stereo room effect.
- `piano_host.hpp` and `piano_resampler.hpp`: host-rate adaptation and resampling.
- `live_diagnostics.hpp` and `live_bench.hpp`: support compiled into the macOS host.

The C++ source/header files originate from the development project. Historical
comments may refer to documentation in that larger project.
The original test and benchmark collections are not bundled here.

## License

MIT. See [LICENSE](LICENSE).
