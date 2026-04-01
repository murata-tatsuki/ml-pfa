# Build ROOT analysis macros as native executables (faster than CINT/cling for heavy jobs).
#
# Typical workflow:
#   cd /path/to/master/macro
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j$(nproc)
#   ./build/efficiency_purity_contribution
# Entry points in mains/, macros in src/
#
cmake_minimum_required(VERSION 3.16)
project(macro_root_tools LANGUAGES CXX)

# Match ROOT's ABI if possible (avoids RConfigure "standard does not match" warnings).
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ROOT CONFIG REQUIRED COMPONENTS Core Hist Tree Graf Gpad MathCore RIO Gui)

# Helper: register another macro executable — copy this block and change names/sources.
function(macro_root_executable target_name)
    cmake_parse_arguments(ARG "" "" "SOURCES;EXTRA_LIBS" ${ARGN})
    add_executable(${target_name} ${ARG_SOURCES})
    target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${target_name}
        PRIVATE
        ROOT::Core
        ROOT::Hist
        ROOT::Tree
        ROOT::Graf
        ROOT::Gpad
        ROOT::MathCore
        ROOT::RIO
        ROOT::Gui
        ${ARG_EXTRA_LIBS}
    )
endfunction()

macro_root_executable(efficiency_purity_contribution
    SOURCES
        mains/main_efficiency_purity_contribution.cxx
        src/efficiency_purity_check_reco_effpur_contiribution.cxx
)
