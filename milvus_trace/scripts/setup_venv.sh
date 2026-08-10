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

mkdir -p "$BUILD_DIR"
echo "Using Python: $VENV_PYTHON"
echo "Using CMake build directory: $BUILD_DIR"
"$VENV_PYTHON" -m pip install --upgrade "pip==25.3" "pip-tools==7.5.2"
cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --target generate_py3_requirements
[[ -f "$REPO_ROOT/requirement.txt" ]] || die "CMake did not generate requirement.txt"
"$VENV_PYTHON" -m pip install -r "$REPO_ROOT/requirement.txt"

if (( ! SKIP_MONITORING )); then
    cmake --build "$BUILD_DIR" --target libmsys_pymod --parallel "$BUILD_JOBS"
fi

echo
echo "RAGPerf Python environment is ready."
echo "Activate it with: source $VENV_DIR/bin/activate"
if (( SKIP_MONITORING )); then
    echo "Monitoring build skipped; run without --skip-monitoring to build libmsys_pymod."
fi
