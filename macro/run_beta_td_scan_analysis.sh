#!/usr/bin/env bash

# Create a compact beta/distance scan summary and its plots without hadd.
# ROOT macros are compiled with ACLiC and cached below the result directory.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./run_beta_td_scan_analysis.sh SCAN_DIRECTORY ENERGY_GEV [RESULT_DIRECTORY] [CATEGORY]

Arguments:
  SCAN_DIRECTORY    Directory containing tbetaXXXtdXXX directories.
  ENERGY_GEV        Center-of-mass energy, for example 40, 91, 200, 350, or 500.
  RESULT_DIRECTORY  Optional. Default: SCAN_DIRECTORY
  CATEGORY          Optional. Default: all
                    all, inclusive, electron, pion, photon, neutron, K0, or muon

Optional environment variables:
  CEDRIC_N_MIN      Minimum number of hits in a Cedric cluster. Default: 2
  CEDRIC_E_MIN      Minimum Cedric cluster energy in GeV. Default: 0.0

Example:
  ./run_beta_td_scan_analysis.sh ../output/energy_regression_1to1/skimmed/tc_nnqq_2M/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/91GeV/multi-head 91
USAGE
}

if [[ $# -lt 2 || $# -gt 4 ]]; then
  usage >&2
  exit 2
fi

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
scan_directory_input=$1
energy_gev=$2
category=${4:-all}
cedric_n_min=${CEDRIC_N_MIN:-2}
cedric_e_min=${CEDRIC_E_MIN:-0.0}

if [[ ! -d "$scan_directory_input" ]]; then
  echo "ERROR: scan directory does not exist: $scan_directory_input" >&2
  exit 1
fi
if [[ ! "$energy_gev" =~ ^[0-9]+$ ]]; then
  echo "ERROR: ENERGY_GEV must be a non-negative integer: $energy_gev" >&2
  exit 1
fi
if [[ ! "$cedric_n_min" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: CEDRIC_N_MIN must be a positive integer: $cedric_n_min" >&2
  exit 1
fi
if [[ ! "$cedric_e_min" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "ERROR: CEDRIC_E_MIN must be a non-negative number: $cedric_e_min" >&2
  exit 1
fi
case "$category" in
  all|inclusive|electron|pion|photon|neutron|K0|muon) ;;
  *)
    echo "ERROR: unsupported category: $category" >&2
    usage >&2
    exit 1
    ;;
esac
if ! command -v root >/dev/null 2>&1; then
  echo "ERROR: ROOT command was not found. Source thisroot.sh first." >&2
  exit 1
fi

scan_directory=$(realpath "$scan_directory_input")
result_directory_input=${3:-"$scan_directory"}
mkdir -p "$result_directory_input"
result_directory=$(realpath "$result_directory_input")
plot_directory="$result_directory/beta_td_scan_plots_${energy_gev}GeV"
aclic_directory="$result_directory/.beta_td_scan_aclic"
summary_file="$result_directory/beta_td_scan_summary_${energy_gev}GeV.root"
mkdir -p "$plot_directory" "$aclic_directory"

shopt -s nullglob
scan_point_candidates=("$scan_directory"/tbeta???td???)
shopt -u nullglob
scan_point_count=0
for scan_point in "${scan_point_candidates[@]}"; do
  scan_point_name=${scan_point##*/}
  if [[ -d "$scan_point" && "$scan_point_name" =~ ^tbeta[0-9]{3}td[0-9]{3}$ ]]; then
    scan_point_count=$((scan_point_count + 1))
  fi
done
if [[ $scan_point_count -eq 0 ]]; then
  echo "ERROR: no tbetaXXXtdXXX directories found under $scan_directory" >&2
  exit 1
fi
if [[ $scan_point_count -ne 81 ]]; then
  echo "WARNING: found $scan_point_count scan points; 81 were expected." >&2
else
  echo "Found all 81 scan points."
fi

# Escape paths before embedding them in ROOT/C++ string literals.
cpp_escape() {
  local value=$1
  value=${value//\\/\\\\}
  value=${value//\"/\\\"}
  printf '%s' "$value"
}

scan_directory_cpp=$(cpp_escape "$scan_directory")
summary_file_cpp=$(cpp_escape "$summary_file")
plot_directory_cpp=$(cpp_escape "$plot_directory")
aclic_directory_cpp=$(cpp_escape "$aclic_directory")
make_macro_cpp=$(cpp_escape "$script_directory/src/make_beta_td_scan_summary.cxx")
plot_macro_cpp=$(cpp_escape "$script_directory/src/plot_beta_td_scan_summary.cxx")

echo "Creating scan summary: $summary_file"
root -l -b -q \
  -e "gSystem->SetBuildDir(\"$aclic_directory_cpp\", kTRUE);" \
  "$make_macro_cpp+(\"$scan_directory_cpp\",\"$summary_file_cpp\",$energy_gev,$cedric_n_min,$cedric_e_min)"

echo "Creating scan plots: $plot_directory"
root -l -b -q \
  -e "gSystem->SetBuildDir(\"$aclic_directory_cpp\", kTRUE);" \
  "$plot_macro_cpp+(\"$summary_file_cpp\",\"$plot_directory_cpp\",\"$category\")"

echo "Analysis completed."
echo "  Summary ROOT: $summary_file"
echo "  Plot directory: $plot_directory"
echo "  Plot ROOT: $plot_directory/beta_td_scan_plots.root"
