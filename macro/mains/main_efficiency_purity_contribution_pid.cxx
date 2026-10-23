// Native batch entry point for the PID efficiency/purity analysis only.
#include "TApplication.h"
#include "TROOT.h"

#include <exception>
#include <unistd.h>
#include <iostream>
#include <string>
#include <stdexcept>

void efficiency_purity_check_reco_effpur_contiribution_pid(
    const char* input_file, bool pandora_mode, const char* output_directory);
void efficiency_purity_check_pid_energy_readout(
    const char* readout_cache, const char* energy_readout, const char* output_directory,
    const char* image_directory);

namespace {

void usage(std::ostream& out, const char* program) {
    out << "Usage: " << program << " [INPUT_ROOT [PANDORA [OUTPUT_DIRECTORY]]]\n"
        << "  INPUT_ROOT: omitted or empty uses the configured input list.\n"
        << "  PANDORA: true/false/1/0; default is false.\n"
        << "  OUTPUT_DIRECTORY: omitted or empty uses the configured output.\n"
        << "  Internal paired pass: --energy-readout raw_head|track_replaced\n"
        << "    --readout-cache CACHE_ROOT --output-directory DIRECTORY\n"
        << "    [--image-directory IMAGE_BASE] (defaults to DIRECTORY)\n"
        << "  --help, -h: show this help without running the analysis.\n";
}

bool parsePandora(const std::string& value, bool& result) {
    if (value == "true" || value == "1") {
        result = true;
        return true;
    }
    if (value == "false" || value == "0") {
        result = false;
        return true;
    }
    return false;
}

} // namespace

int main(int argc, char** argv) {
    const char* program = argc > 0 && argv[0] ? argv[0] : "efficiency_purity_contribution_pid";
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string argument(argv[i]);
            if (argument == "--help" || argument == "-h") {
                usage(std::cout, program);
                return 0;
            }
        }
        bool energy_pass = false;
        std::string readout, cache, paired_output, paired_images;
        for (int i=1; i<argc; ++i) {
            const std::string argument(argv[i]);
            if (argument == "--energy-readout" || argument == "--readout-cache" ||
                argument == "--output-directory" || argument == "--image-directory")
                energy_pass = true;
        }
        if (energy_pass) {
            for (int i=1; i<argc; i+=2) {
                const std::string argument(argv[i]);
                if (i + 1 >= argc) throw std::runtime_error("Missing internal paired-pass option value");
                std::string* destination = argument == "--energy-readout" ? &readout :
                    argument == "--readout-cache" ? &cache :
                    argument == "--output-directory" ? &paired_output :
                    argument == "--image-directory" ? &paired_images : nullptr;
                if (!destination || !destination->empty())
                    throw std::runtime_error("Invalid or repeated internal paired-pass option: " + argument);
                *destination = argv[i+1];
            }
            if ((readout != "raw_head" && readout != "track_replaced") || cache.empty() || paired_output.empty())
                throw std::runtime_error("Paired pass requires readout, cache and output directory");
        }
        if (!energy_pass && argc > 4) {
            std::cerr << "Too many arguments.\n";
            usage(std::cerr, program);
            return 2;
        }
        bool pandora = false;
        if (!energy_pass && argc > 2 && !parsePandora(argv[2], pandora)) {
            std::cerr << "Invalid PANDORA value: " << argv[2]
                      << "; expected true, false, 1, or 0.\n";
            usage(std::cerr, program);
            return 2;
        }

        // Keep these strings alive until both the analysis and TApplication finish.
        const std::string input = !energy_pass && argc > 1 ? argv[1] : "";
        const std::string output = energy_pass ? paired_output : (argc > 3 ? argv[3] : "");
        if (!input.empty()) {
            if (::access(input.c_str(), R_OK) != 0) {
                std::cerr << "Cannot read input ROOT file: " << input << '\n';
                return 1;
            }
        }

        // ROOT receives only its own writable argv[0], never the analysis CLI.
        int root_argc = 1;
        char root_program[] = "efficiency_purity_contribution_pid";
        char* root_argv[] = {root_program, nullptr};
        gROOT->SetBatch(kTRUE);
        TApplication app("effpur_contribution_pid", &root_argc, root_argv);
        if (energy_pass)
            efficiency_purity_check_pid_energy_readout(cache.c_str(), readout.c_str(), output.c_str(),
                paired_images.empty() ? nullptr : paired_images.c_str());
        else
            efficiency_purity_check_reco_effpur_contiribution_pid(
                input.empty() ? nullptr : input.c_str(), pandora,
                output.empty() ? nullptr : output.c_str());
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "PID analysis failed: " << error.what() << '\n';
        return 1;
    }
}
