#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="${RAGPERF_VENV_DIR:-$REPO_ROOT/.venv}"
BUILD_DIR="${RAGPERF_BUILD_DIR:-$REPO_ROOT/build}"
SKIP_MONITORING=0
BUILD_JOBS="${RAGPERF_BUILD_JOBS:-2}"

usage() {
    cat <<EOF_USAGE
Usage: $(basename "$0") [options]

Create or reuse a Python venv, install RAGPerf dependencies, and build MSys.

Options:
  --venv-dir PATH    venv location (default: $REPO_ROOT/.venv)
  --build-dir PATH   CMake build location (default: $REPO_ROOT/build)
  --skip-monitoring  Do not build libmsys_pymod
  -h, --help         Show this help

This script does not install Docker, CUDA, system packages, or a C++ compiler.
EOF_USAGE
}

die() {
    echo "setup_venv: $*" >&2
    exit 1
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

resolve_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "$REPO_ROOT" "$1" ;;
    esac
}

while (($# > 0)); do
    case "$1" in
        --venv-dir)
            (($# >= 2)) || die "--venv-dir requires a path"
            VENV_DIR="$(resolve_path "$2")"
            shift 2
            ;;
        --build-dir)
            (($# >= 2)) || die "--build-dir requires a path"
            BUILD_DIR="$(resolve_path "$2")"
            shift 2
            ;;
        --skip-monitoring)
            SKIP_MONITORING=1
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

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    if [[ -n "${RAGPERF_PYTHON:-}" ]]; then
        HOST_PYTHON="$RAGPERF_PYTHON"
    elif command_exists python; then
        HOST_PYTHON=python
    elif command_exists python3; then
        HOST_PYTHON=python3
    else
        die "Python 3.10 or newer is required to create $VENV_DIR"
    fi
    echo "Creating Python venv: $VENV_DIR"
    "$HOST_PYTHON" -m venv "$VENV_DIR" || die "failed to create $VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
"$VENV_PYTHON" - <<'EOF_VERSION'
import sys

if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10 or newer is required")
EOF_VERSION

command_exists cmake || die "cmake is required; install it before running this script"
if ! command_exists c++ && ! command_exists g++; then
    die "a C++ compiler is required; install a C++20-compatible compiler first"
fi

TMP_ROOT="${TMPDIR:-/tmp}"
[[ "$TMP_ROOT" = /* ]] || die "TMPDIR must be an absolute path"
SETUP_TMP_DIR="$(mktemp -d "$TMP_ROOT/ragperf-trace-setup.XXXXXX")"
case "$SETUP_TMP_DIR" in
    "$TMP_ROOT"/ragperf-trace-setup.*) ;;
    *) die "mktemp created an unexpected temporary directory: $SETUP_TMP_DIR" ;;
esac

cleanup() {
    if [[ -n "${SETUP_TMP_DIR:-}" && -d "$SETUP_TMP_DIR" ]]; then
        case "$SETUP_TMP_DIR" in
            "$TMP_ROOT"/ragperf-trace-setup.*) rm -rf -- "$SETUP_TMP_DIR" ;;
        esac
    fi
}
trap cleanup EXIT

prepare_local_requirements() {
    local root_requirements="$REPO_ROOT/resource/requirements.in"
    local cmake_extra_requirements="$REPO_ROOT/resource/generated/extra.in"
    local local_requirements="$SETUP_TMP_DIR/requirements.in"
    [[ -f "$root_requirements" ]] || die "missing root requirements file: $root_requirements"
    [[ -f "$cmake_extra_requirements" ]] || die "CMake did not generate $cmake_extra_requirements"
    # Keep dependency ownership in the root project, while applying trace-local
    # compatibility overrides without modifying any root-tracked file.
    awk '!/^[[:space:]]*vllm([[:space:]]|=|<|>)/' "$root_requirements" > "$local_requirements"
    printf '\n# Milvus trace compatibility overrides\nvllm==0.8.5.post1\nmarshmallow<4\nsetuptools>=74.1.1,<81\nprotobuf==6.32.0\n' >> "$local_requirements"
    sed -E 's/^clang==([0-9]+\.[0-9]+)(.*)$/clang~=\1\2/' \
        "$cmake_extra_requirements" >> "$local_requirements"
    printf '%s\n' "$local_requirements"
}

mkdir -p "$BUILD_DIR"
echo "Using Python: $VENV_PYTHON"
echo "Using CMake build directory: $BUILD_DIR"
export PATH="$VENV_DIR/bin:$PATH"
"$VENV_PYTHON" -m pip install --upgrade "pip==25.3" "pip-tools==7.5.2"
cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
LOCAL_REQUIREMENTS_IN="$(prepare_local_requirements)"
LOCAL_REQUIREMENTS_LOCK="$SETUP_TMP_DIR/requirements.txt"
"$VENV_DIR/bin/pip-compile" "$LOCAL_REQUIREMENTS_IN" \
    --output-file "$LOCAL_REQUIREMENTS_LOCK" --strip-extras
"$VENV_PYTHON" -m pip install -r "$LOCAL_REQUIREMENTS_LOCK"

if (( ! SKIP_MONITORING )); then
    cmake --build "$BUILD_DIR" --target libmsys_pymod --parallel "$BUILD_JOBS"
fi

echo
echo "RAGPerf Python environment is ready."
echo "Activate it with: source $VENV_DIR/bin/activate"
if (( SKIP_MONITORING )); then
    echo "Monitoring build skipped; run without --skip-monitoring to build libmsys_pymod."
fi
