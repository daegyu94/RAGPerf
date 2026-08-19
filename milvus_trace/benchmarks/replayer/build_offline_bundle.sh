#!/usr/bin/env bash
# Build the Python wheelhouse and optional Milvus image archive used by an
# isolated staged replay node.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/../../.." && pwd -P)"
project_python="$project_root/.venv/bin/python"
requirements="$project_root/milvus_trace/docker/requirements-replay.lock"
output_dir=""
python_version=""
target_platform=""
target_implementation="cp"
target_abi=""
include_milvus_images=false

milvus_images=(
  "quay.io/coreos/etcd:v3.5.5"
  "minio/minio:RELEASE.2023-03-20T20-16-18Z"
  "milvusdb/milvus:v2.4.15"
)

usage() {
  cat <<'EOF'
Usage:
  bash milvus_trace/benchmarks/replayer/build_offline_bundle.sh \
    --output-dir PATH [OPTIONS]

Options:
  --output-dir PATH       New or empty output directory.
  --requirements PATH     Replay lock file.
  --python-version X.Y    Target CPython version for cross-platform downloads.
  --platform TAG          Target wheel platform, for example manylinux_2_28_x86_64.
  --implementation TAG    Target Python implementation (default: cp).
  --abi TAG               Target Python ABI, for example cp312.
  --include-milvus-images Save the three already-cached standalone Milvus images.
  -h, --help              Show this help.

The script always uses this project's .venv. It downloads wheels on the
controller; the replay node later installs them with pip --no-index.
EOF
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

require_value() {
  (($# >= 2)) || die "$1 requires a value"
}

while (($#)); do
  case "$1" in
    --output-dir)
      require_value "$@"
      output_dir="$2"
      shift 2
      ;;
    --requirements)
      require_value "$@"
      requirements="$2"
      shift 2
      ;;
    --python-version)
      require_value "$@"
      python_version="$2"
      shift 2
      ;;
    --platform)
      require_value "$@"
      target_platform="$2"
      shift 2
      ;;
    --implementation)
      require_value "$@"
      target_implementation="$2"
      shift 2
      ;;
    --abi)
      require_value "$@"
      target_abi="$2"
      shift 2
      ;;
    --include-milvus-images)
      include_milvus_images=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unknown option: $1"
      ;;
  esac
done

[[ -n "$output_dir" ]] || die "--output-dir is required"
[[ "$output_dir" == /* ]] || die "--output-dir must be absolute"
[[ -x "$project_python" ]] || die "project virtual environment is missing: $project_python"
[[ -f "$requirements" ]] || die "requirements file not found: $requirements"

if [[ -e "$output_dir" ]]; then
  [[ -d "$output_dir" ]] || die "output exists and is not a directory: $output_dir"
  [[ -z "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] || \
    die "output directory is not empty: $output_dir"
else
  mkdir -p -- "$output_dir"
fi
mkdir -p -- "$output_dir/wheelhouse"

download_args=(
  -m pip download
  --only-binary=:all:
  --dest "$output_dir/wheelhouse"
  --requirement "$requirements"
)
if [[ -n "$python_version" || -n "$target_platform" || -n "$target_abi" ]]; then
  [[ -n "$python_version" && -n "$target_platform" && -n "$target_abi" ]] || \
    die "--python-version, --platform, and --abi must be supplied together"
  download_args+=(
    --python-version "$python_version"
    --platform "$target_platform"
    --implementation "$target_implementation"
    --abi "$target_abi"
  )
fi

"$project_python" "${download_args[@]}"
cp -- "$requirements" "$output_dir/requirements-replay.lock"

if [[ "$include_milvus_images" == true ]]; then
  command -v docker >/dev/null 2>&1 || die "docker is required to save Milvus images"
  for image in "${milvus_images[@]}"; do
    docker image inspect "$image" >/dev/null 2>&1 || \
      die "container image is not cached on the controller: $image"
  done
  docker save --output "$output_dir/milvus-images.tar" "${milvus_images[@]}"
fi

(
  cd "$output_dir"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)

echo "Offline replay bundle: $output_dir"
echo "Wheel files: $(find "$output_dir/wheelhouse" -maxdepth 1 -type f | wc -l)"
if [[ "$include_milvus_images" == true ]]; then
  echo "Milvus images: $output_dir/milvus-images.tar"
fi
