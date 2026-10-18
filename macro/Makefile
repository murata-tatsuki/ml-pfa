# Run from this directory:  make
# Configures CMake in ./build (if needed) and builds the default target.

BUILD_DIR   := build
BUILD_TYPE  ?= Release
ROOT_CONFIG ?= root-config
# Match the active ROOT environment, even if build/ cached another ROOT.
ROOT_DIR    ?= $(shell $(ROOT_CONFIG) --prefix)/cmake
CMAKE_FLAGS ?= -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

.PHONY: all clean reconfigure efficiency_purity_contribution_pid

all:
	@cmake -S . -B $(BUILD_DIR) -DROOT_DIR="$(ROOT_DIR)" $(CMAKE_FLAGS)
	@cmake --build $(BUILD_DIR) --parallel

clean:
	$(RM) -r $(BUILD_DIR)

# Force re-run CMake (e.g. after editing CMakeLists.txt)
reconfigure:
	$(RM) -r $(BUILD_DIR)
	@$(MAKE) all

# Build only the PID analysis executable.
efficiency_purity_contribution_pid:
	@cmake -S . -B $(BUILD_DIR) -DROOT_DIR="$(ROOT_DIR)" $(CMAKE_FLAGS)
	@cmake --build $(BUILD_DIR) --target $@ --parallel
