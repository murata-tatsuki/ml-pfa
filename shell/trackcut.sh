#!/usr/bin/env bash
# Apply explicit track-quality cuts to *.h5 directly inside one directory.
set -euo pipefail

if [ "$#" -lt 2 ]; then
    printf 'Usage: bash %s INPUT_DIR OUTPUT_DIR --max-r R --invalid-r-policy keep|drop|error [options]\n' "$0" >&2
    exit 2
fi
input_dir=$1
output_dir=$2
shift 2
if [ ! -d "$input_dir" ]; then
    printf 'Input directory does not exist: %s\n' "$input_dir" >&2
    exit 2
fi
mkdir -p -- "$output_dir"
input_dir=$(cd -- "$input_dir" && pwd -P)
output_dir=$(cd -- "$output_dir" && pwd -P)
if [ "$input_dir" = "$output_dir" ]; then
    printf 'Input and output directories must differ.\n' >&2
    exit 2
fi
script_dir=$(cd -- "$(dirname -- "$0")/.." && pwd -P)
found=0
for input in "$input_dir"/*.h5; do
    [ -f "$input" ] || continue
    found=1
    name=$(basename -- "$input")
    python -B "$script_dir/trackcut.py" -i "$input" -o "$output_dir/$name" "$@"
done
if [ "$found" -eq 0 ]; then
    printf 'No *.h5 files directly inside: %s\n' "$input_dir" >&2
    exit 2
fi
