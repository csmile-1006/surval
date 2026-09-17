#!/usr/bin/env bash
set -Eeuo pipefail
task_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_repo_root"
task_python="${PYTHON:-$task_repo_root/.venv-droid/bin/python}"
if [[ "${SKIP_ARTIFACT_CHECK:-0}" == "1" ]]; then
  export DRY_RUN=1
fi
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
exec "$task_python" -u scripts/tune_droid_policy_reference.py --reference-epoch 25 "$@"
