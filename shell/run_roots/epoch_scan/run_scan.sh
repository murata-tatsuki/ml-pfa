#!/usr/bin/env bash
# 共通処理。通常はsetting.shを編集して実行します。
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${PYTHON_BIN:-python3}" -B "$script_dir/run_setting.py" "$@"
