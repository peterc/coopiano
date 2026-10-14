CXX ?= c++
CXXFLAGS ?= -O2

ifeq ($(shell uname -s),Darwin)
HOST_FLAGS = -Wno-deprecated-declarations
HOST_LIBS = -framework CoreMIDI -framework AudioToolbox -framework CoreFoundation -framework CoreAudio
endif

HEADERS = instrument.hpp streaming_soundboard.hpp piano.hpp preset.hpp reverb.hpp \
          live_diagnostics.hpp live_bench.hpp piano_host.hpp piano_resampler.hpp

.PHONY: all clean
all: coopiano

coopiano: coopiano.cpp $(HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -std=c++17 -pthread $(HOST_FLAGS) coopiano.cpp -o $@ $(LDFLAGS) $(LDLIBS) $(HOST_LIBS)

clean:
	rm -f coopiano
