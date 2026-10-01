#!/bin/bash
# Usage: bash shell/timingcut_single_particle.sh inputdir outputdir [cut options]
set -euo pipefail

if (( $# < 2 )); then
    echo "Usage: $0 inputdir outputdir [cut options]" >&2
    exit 2
fi

scriptdir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
inputdir=$1
outputdir=$2
shift 2
shopt -s nullglob
files=("$inputdir"/*.h5)
if (( ${#files[@]} == 0 )); then
    echo "No .h5 files found in $inputdir" >&2
    exit 1
fi
mkdir -p -- "$outputdir"
for file in "${files[@]}"; do
    echo "Processing $file ..."
    python "$scriptdir/timingcut_single_particle.py" \
        -i "$file" -o "$outputdir/$(basename -- "$file")" "$@"
done
