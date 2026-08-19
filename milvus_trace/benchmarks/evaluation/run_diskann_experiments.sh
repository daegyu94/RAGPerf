#!/usr/bin/env bash
# Activate the project environment and run the DiskANN experiment matrix.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/../../.." && pwd -P)"
venv="$project_root/.venv"

if [[ ! -x "$venv/bin/python" ]]; then
  echo "Project virtual environment is missing: $venv" >&2
  exit 1
fi

cd "$project_root"
source "$venv/bin/activate"
python -m milvus_trace.benchmarks.evaluation.run_diskann_experiments "$@"
