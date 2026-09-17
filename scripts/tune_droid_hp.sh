#!/usr/bin/env bash
set -Eeuo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$script_dir/.."
command=("${DROID_PYTHON:-.venv-droid/bin/python}" -u scripts/tune_droid_hp.py)
exec "${command[@]}" "$@"
