// Native batch entry point for the PID efficiency/purity analysis only.
#include "TApplication.h"
#include "TROOT.h"

#include <exception>
#include <fstream>
#include <iostream>
#include <string>

void efficiency_purity_check_reco_effpur_contiribution_pid(
    const char* input_file, bool pandora_mode, const char* output_directory);

namespace {

void usage(std::ostream& out, const char* program) {
    out << "Usage: " << program << " [INPUT_ROOT [PANDORA [OUTPUT_DIRECTORY]]]\n"
        << "  INPUT_ROOT: omitted or empty uses the configured input list.\n"
        << "  PANDORA: true/false/1/0; default is false.\n"
        << "  OUTPUT_DIRECTORY: omitted or empty uses the configured output.\n"
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
        if (argc > 4) {
            std::cerr << "Too many arguments.\n";
            usage(std::cerr, program);
            return 2;
        }
        bool pandora = false;
        if (argc > 2 && !parsePandora(argv[2], pandora)) {
            std::cerr << "Invalid PANDORA value: " << argv[2]
                      << "; expected true, false, 1, or 0.\n";
            usage(std::cerr, program);
            return 2;
        }

        // Keep these strings alive until both the analysis and TApplication finish.
        const std::string input = argc > 1 ? argv[1] : "";
        const std::string output = argc > 3 ? argv[3] : "";
        if (!input.empty()) {
            std::ifstream check(input.c_str(), std::ios::binary);
            if (!check) {
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
        efficiency_purity_check_reco_effpur_contiribution_pid(
            input.empty() ? nullptr : input.c_str(), pandora,
            output.empty() ? nullptr : output.c_str());
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "PID analysis failed: " << error.what() << '\n';
        return 1;
    }
}
