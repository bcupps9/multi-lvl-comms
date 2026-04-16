# Provided for convenience; delegates immediately to cmake.
# - `make` builds all targets in the `build` directory.
# - `make BUILD=build-san SAN=1` builds with sanitizers in `build-dir`.
# - `make targetname` builds a single target.

# Set build directory
BUILD ?= build

# SAN, ASAN, UBSAN, TSAN: enable different sanitizers
cmake_bool = $(if $(filter 1 on,$(1)),ON,$(if $(filter 0 off,$(1)),OFF,$(1)))
cmake_build := -DSAN=$(call cmake_bool,$(SAN)) \
	-DASAN=$(call cmake_bool,$(ASAN)) \
	-DUBSAN=$(call cmake_bool,$(UBSAN)) \
	-DTSAN=$(call cmake_bool,$(TSAN))

ifeq ($(V),1)
cmake_verbose := --verbose
endif

targets = pt-single pt-backup pt-paxos

all:
	cmake -B $(BUILD) $(cmake_build)
	cmake --build $(BUILD) $(cmake_verbose)

clean:
	rm -rf $(BUILD) .cache

$(targets):
	cmake -B $(BUILD) $(cmake_build)
	cmake --build $(BUILD) --target $@ $(cmake_verbose)

$(targets:%=$(BUILD)/%): $(BUILD)/%:
	cmake -B $(BUILD) $(cmake_build)
	cmake --build $(BUILD) --target $* $(cmake_verbose)

.PHONY: all clean check test $(targets) $(targets:%=$(BUILD)/%)
