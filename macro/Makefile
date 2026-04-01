# Run from this directory:  make
# Configures CMake in ./build (if needed) and builds the default target.

BUILD_DIR   := build
BUILD_TYPE  ?= Release
CMAKE_FLAGS ?= -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

.PHONY: all clean reconfigure

all:
	@cmake -S . -B $(BUILD_DIR) $(CMAKE_FLAGS)
	@cmake --build $(BUILD_DIR) --parallel

clean:
	$(RM) -r $(BUILD_DIR)

# Force re-run CMake (e.g. after editing CMakeLists.txt)
reconfigure:
	$(RM) -r $(BUILD_DIR)
	@$(MAKE) all
