#!/usr/bin/env bash
# Run one Milvus DISKANN replay with all Milvus data on a verified filesystem.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/../../.." && pwd -P)"
compose_file="$project_root/milvus_trace/docker/milvus-diskann-compose.yml"

artifact_dir=""
backend=""
expected_fstype=""
diskann_root=""
meta_root=""
output_dir=""
run_name=""
uri="http://127.0.0.1:19530"
token="root:Milvus"
collection=""
time_scale="1"
timing="original"
max_in_flight="1024"
bootstrap_batch_size="1024"
warmup="0"
milvus_version="v2.4.15"
host_port="19530"
webui_port="9091"
wait_seconds="180"
keep_server=false

usage() {
  cat <<'EOF'
Usage:
  bash milvus_trace/benchmarks/replayer/run_diskann_replay.sh [OPTIONS]

Required:
  --artifact-dir PATH       Extracted RAGPerf trace artifact.
  --backend NAME            Backend label: xfs, 3fs, or pnfs.
  --expected-fstype LIST    Comma-separated findmnt FSTYPE values.
  --diskann-root PATH       Data directory on the filesystem under test.
  --meta-root PATH          Local directory used only for etcd metadata.
  --output-dir PATH         New per-run result directory.
  --run-name NAME           Unique run label.

Replay options:
  --uri URI                 Milvus URI (default: http://127.0.0.1:19530).
  --token TOKEN             Milvus token (default: root:Milvus).
  --collection NAME         Target collection (default derived from run name).
  --timing original|none
  --time-scale NUMBER
  --max-in-flight NUMBER
  --bootstrap-batch-size NUMBER
  --warmup NUMBER
  --milvus-version TAG      Preloaded Milvus image tag.
  --host-port PORT
  --webui-port PORT
  --wait-seconds SECONDS
  --keep-server             Leave the Compose project running after replay.
  -h, --help                Show this help.

The command uses docker compose --pull never. It never downloads an image or a
Python package on the replay node.
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
    --artifact-dir|--backend|--expected-fstype|--diskann-root|--meta-root|--output-dir|--run-name|--uri|--token|--collection|--timing|--time-scale|--max-in-flight|--bootstrap-batch-size|--warmup|--milvus-version|--host-port|--webui-port|--wait-seconds)
      require_value "$@"
      option="${1#--}"
      option="${option//-/_}"
      printf -v "$option" '%s' "$2"
      shift 2
      ;;
    --keep-server)
      keep_server=true
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

for required in artifact_dir backend expected_fstype diskann_root meta_root output_dir run_name; do
  [[ -n "${!required}" ]] || die "--${required//_/-} is required"
done
[[ "$run_name" =~ ^[A-Za-z0-9._-]+$ ]] || die "run name contains unsupported characters"
[[ "$backend" =~ ^[A-Za-z0-9._-]+$ ]] || die "backend contains unsupported characters"
[[ "$timing" == original || "$timing" == none ]] || die "--timing must be original or none"
for path_value in "$artifact_dir" "$diskann_root" "$meta_root" "$output_dir"; do
  [[ "$path_value" == /* ]] || die "all data paths must be absolute: $path_value"
  [[ "$path_value" != / ]] || die "data paths must not be /"
done
[[ -d "$artifact_dir" ]] || die "artifact directory not found: $artifact_dir"
[[ -d "$diskann_root" ]] || die "DiskANN data directory not found: $diskann_root"
[[ ! -L "$diskann_root" ]] || die "DiskANN data directory must not be a symlink"
[[ -w "$diskann_root" ]] || die "DiskANN data directory is not writable: $diskann_root"
[[ -f "$compose_file" ]] || die "Compose file not found: $compose_file"
[[ -x "$project_root/.venv/bin/python" ]] || die "replay venv is missing: $project_root/.venv"
[[ ! -e "$output_dir" ]] || die "output directory already exists: $output_dir"

command -v findmnt >/dev/null 2>&1 || die "findmnt is required"
mount_record="$(findmnt -n -T "$diskann_root" -o TARGET,FSTYPE,SOURCE || true)"
[[ -n "$mount_record" ]] || die "DiskANN root does not resolve to a mounted filesystem: $diskann_root"
read -r mount_target actual_fstype mount_source <<< "$mount_record"
fstype_matches=false
IFS=',' read -r -a accepted_fstypes <<< "$expected_fstype"
for accepted in "${accepted_fstypes[@]}"; do
  if [[ "$actual_fstype" == "$accepted" ]]; then
    fstype_matches=true
    break
  fi
done
[[ "$fstype_matches" == true ]] || \
  die "mount FSTYPE $actual_fstype does not match $expected_fstype for $backend"

command -v docker >/dev/null 2>&1 || die "docker is required"
docker compose version >/dev/null 2>&1 || die "Docker Compose v2 is required"
docker info >/dev/null 2>&1 || die "Docker daemon is unavailable"
for image in \
  quay.io/coreos/etcd:v3.5.5 \
  minio/minio:RELEASE.2023-03-20T20-16-18Z \
  "milvusdb/milvus:$milvus_version"; do
  docker image inspect "$image" >/dev/null 2>&1 || die "preloaded image is missing: $image"
done

diskann_run_root="$diskann_root/$run_name"
meta_run_root="$meta_root/$run_name"
[[ ! -e "$diskann_run_root" ]] || die "DiskANN run directory already exists: $diskann_run_root"
[[ ! -e "$meta_run_root" ]] || die "metadata run directory already exists: $meta_run_root"
mkdir -p -- "$diskann_run_root" "$meta_run_root" "$output_dir"

compose_project="ragperf-${backend}-${run_name}"
compose_project="${compose_project//./-}"
export MILVUS_DISKANN_DIR="$diskann_run_root"
export MILVUS_META_DIR="$meta_run_root"
export MILVUS_VERSION="$milvus_version"
export MILVUS_HOST_PORT="$host_port"
export MILVUS_WEBUI_PORT="$webui_port"

compose() {
  docker compose --project-name "$compose_project" --file "$compose_file" "$@"
}

cleanup() {
  local status=$?
  if [[ "$keep_server" != true ]]; then
    compose down >/dev/null 2>&1 || true
  fi
  return "$status"
}
trap cleanup EXIT

{
  echo "run_name=$run_name"
  echo "backend=$backend"
  echo "mount_target=$mount_target"
  echo "mount_source=$mount_source"
  echo "mount_fstype=$actual_fstype"
  echo "diskann_run_root=$diskann_run_root"
  echo "meta_run_root=$meta_run_root"
  echo "artifact_dir=$artifact_dir"
  echo "milvus_version=$milvus_version"
  echo "timing=$timing"
  echo "time_scale=$time_scale"
  echo "started_at=$(date --iso-8601=seconds)"
  findmnt -T "$diskann_root"
  df -hT "$diskann_root"
} > "$output_dir/run-metadata.txt"

compose up -d --pull never
deadline=$((SECONDS + wait_seconds))
while ((SECONDS < deadline)); do
  if (echo >/dev/tcp/127.0.0.1/"$host_port") >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
if ! (echo >/dev/tcp/127.0.0.1/"$host_port") >/dev/null 2>&1; then
  compose ps >> "$output_dir/run-metadata.txt" 2>&1 || true
  compose logs --tail=200 standalone > "$output_dir/milvus-failure.log" 2>&1 || true
  die "Milvus did not open port $host_port within ${wait_seconds}s"
fi

source "$project_root/.venv/bin/activate"
milvus_ready=false
while ((SECONDS < deadline)); do
  if MILVUS_REPLAY_URI="$uri" MILVUS_REPLAY_TOKEN="$token" python -c 'import os; from pymilvus import MilvusClient; client = MilvusClient(uri=os.environ["MILVUS_REPLAY_URI"], token=os.environ["MILVUS_REPLAY_TOKEN"], timeout=2); client.list_collections(); client.close()' >/dev/null 2>&1; then
    milvus_ready=true
    break
  fi
  sleep 2
done
if [[ "$milvus_ready" != true ]]; then
  compose ps >> "$output_dir/run-metadata.txt" 2>&1 || true
  compose logs --tail=200 standalone > "$output_dir/milvus-failure.log" 2>&1 || true
  die "Milvus Proxy was not ready within ${wait_seconds}s"
fi
echo "milvus_ready_at=$(date --iso-8601=seconds)" >> "$output_dir/run-metadata.txt"

if [[ -z "$collection" ]]; then
  collection="ragperf_${run_name//[^A-Za-z0-9_]/_}"
fi
python -m milvus_trace.replay \
  --artifact-dir "$artifact_dir" \
  --uri "$uri" \
  --token "$token" \
  --collection "$collection" \
  --timing "$timing" \
  --time-scale "$time_scale" \
  --max-in-flight "$max_in_flight" \
  --bootstrap-batch-size "$bootstrap_batch_size" \
  --warmup "$warmup" \
  --result-file "$output_dir/replay-result.json"

{
  echo "finished_at=$(date --iso-8601=seconds)"
  du -sb "$diskann_run_root"
  compose ps
} >> "$output_dir/run-metadata.txt"
