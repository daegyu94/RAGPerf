#!/usr/bin/env bash
# Stage a RAGPerf trace, offline Python runtime, and replay command to an
# isolated storage/replay node. The replay node never contacts GitHub, Hugging Face, or PyPI.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "$script_dir/../../.." && pwd -P)"

topology_file=""
phase=""
dry_run=false
overwrite_output=false
run_name=""
assets=()
reset_targets=()
command_args=()
declare -A topology_values=()

usage() {
  cat <<'EOF'
Usage:
  bash milvus_trace/benchmarks/replayer/staged_remote_replay.sh PHASE \
    --topology PATH [OPTIONS]

Phases:
  check-prerequisites  Check apt-provided replay-node tools, Python, and mounts.
  prepare-trace        Transfer and extract one or more local .tar.zst artifacts.
  prepare-replay       Stage milvus_trace, offline wheels/images, and create .venv.
  replay               Run a supplied command and retrieve its output.
  all                  Run prepare-trace, prepare-replay, and replay.
  reset                Reset selected replay-node paths; never touches controller data.

Options:
  --topology PATH      Required flat-scalar topology YAML.
  --asset PATH         Path relative to controller_trace_root. Repeatable.
  --run-name NAME      Required for replay/all.
  --target NAME        reset target: repo, runtime, trace, output, diskann, or all.
  --overwrite-output   Replace only this run-name's remote/controller output.
  --dry-run            Print commands without executing SSH or transfers.
  -h, --help           Show this help.

Command placeholders:
  @REPO_ROOT@ @VENV_ROOT@ @TRACE_ROOT@ @OUTPUT_ROOT@ @DISKANN_ROOT@
  @META_ROOT@ @RUN_NAME@ @STORAGE_BACKEND@ @EXPECTED_FSTYPE@ @MILVUS_URI@

Example:
  bash milvus_trace/benchmarks/replayer/staged_remote_replay.sh all \
    --topology milvus_trace/configs/replayer/staged-remote/xfs.yaml \
    --asset vector/0.5tb.tar.zst \
    --run-name vector-0.5tb-xfs-r1 -- \
    bash @REPO_ROOT@/milvus_trace/benchmarks/replayer/run_diskann_replay.sh \
      --artifact-dir @TRACE_ROOT@/vector/0.5tb \
      --backend @STORAGE_BACKEND@ \
      --expected-fstype @EXPECTED_FSTYPE@ \
      --diskann-root @DISKANN_ROOT@ \
      --meta-root @META_ROOT@ \
      --output-dir @OUTPUT_ROOT@ \
      --run-name @RUN_NAME@ \
      --uri @MILVUS_URI@
EOF
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

warn() {
  echo "[WARN] $*" >&2
}

info() {
  echo "[INFO] $*"
}

require_value() {
  (($# >= 2)) || die "$1 requires a value"
}

while (($#)); do
  case "$1" in
    check-prerequisites|prepare-trace|prepare-replay|replay|all|reset)
      [[ -z "$phase" ]] || die "only one phase may be specified"
      phase="$1"
      shift
      ;;
    --topology)
      require_value "$@"
      topology_file="$2"
      shift 2
      ;;
    --asset)
      require_value "$@"
      assets+=("$2")
      shift 2
      ;;
    --target)
      require_value "$@"
      reset_targets+=("$2")
      shift 2
      ;;
    --run-name)
      require_value "$@"
      run_name="$2"
      shift 2
      ;;
    --overwrite-output)
      overwrite_output=true
      shift
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    --)
      shift
      command_args=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unknown argument: $1"
      ;;
  esac
done

[[ -n "$phase" ]] || { usage >&2; exit 2; }
[[ -n "$topology_file" ]] || die "--topology is required"
[[ -f "$topology_file" ]] || die "topology file not found: $topology_file"

load_topology() {
  local raw line key value
  while IFS= read -r raw || [[ -n "$raw" ]]; do
    raw="${raw%$'\r'}"
    line="${raw%%#*}"
    if [[ "$line" =~ ^[[:space:]]*([A-Za-z0-9_.-]+)[[:space:]]*:[[:space:]]*(.*)[[:space:]]*$ ]]; then
      key="${BASH_REMATCH[1]}"
      value="${BASH_REMATCH[2]}"
      case "$value" in
        \"*\") value="${value:1:${#value}-2}" ;;
        \'*\') value="${value:1:${#value}-2}" ;;
      esac
      topology_values["$key"]="$value"
    fi
  done < "$topology_file"
}

topology_get() {
  local key="$1"
  [[ -v "topology_values[$key]" ]] || return 1
  printf '%s\n' "${topology_values[$key]}"
}

require_topology_key() {
  local value
  value="$(topology_get "$1" || true)"
  [[ -n "$value" ]] || die "topology key is required and has no default: $1"
}

load_topology
for key in \
  controller_repo_root controller_trace_root controller_runtime_root \
  controller_output_root replay_host replay_user replay_port replay_repo_root \
  replay_venv_root replay_runtime_root replay_trace_root replay_output_root \
  replay_diskann_root replay_meta_root replay_python replay_runtime_requirements \
  replay_milvus_uri storage_backend expected_fstype transfer_method; do
  require_topology_key "$key"
done

controller_repo_root="$(topology_get controller_repo_root)"
controller_trace_root="$(topology_get controller_trace_root)"
controller_runtime_root="$(topology_get controller_runtime_root)"
controller_output_root="$(topology_get controller_output_root)"
replay_host="$(topology_get replay_host)"
replay_user="$(topology_get replay_user)"
replay_jump_user="$(topology_get replay_jump_user || true)"
replay_port="$(topology_get replay_port)"
replay_repo_root="$(topology_get replay_repo_root)"
replay_venv_root="$(topology_get replay_venv_root)"
replay_runtime_root="$(topology_get replay_runtime_root)"
replay_trace_root="$(topology_get replay_trace_root)"
replay_output_root="$(topology_get replay_output_root)"
replay_diskann_root="$(topology_get replay_diskann_root)"
replay_meta_root="$(topology_get replay_meta_root)"
replay_python="$(topology_get replay_python)"
replay_runtime_requirements="$(topology_get replay_runtime_requirements)"
replay_require_docker="$(topology_get replay_require_docker || true)"
replay_container_images="$(topology_get replay_container_images || true)"
replay_milvus_uri="$(topology_get replay_milvus_uri)"
storage_backend="$(topology_get storage_backend)"
expected_fstype="$(topology_get expected_fstype)"
transfer_method="$(topology_get transfer_method)"

[[ "$replay_port" =~ ^[0-9]+$ ]] || die "replay_port must be an integer"
case "$transfer_method" in rsync|scp) ;; *) die "transfer_method must be rsync or scp" ;; esac
case "$replay_require_docker" in ""|false|true) ;; *) die "replay_require_docker must be true or false" ;; esac
if [[ "$replay_require_docker" == true ]]; then
  [[ -n "$replay_container_images" ]] || die "replay_container_images is required when replay_require_docker=true"
fi
case "$replay_container_images" in
  ""|/*|../*|*/../*|*/..) [[ -z "$replay_container_images" ]] || die "replay_container_images must be relative" ;;
esac

path_values=(
  "$controller_repo_root" "$controller_trace_root" "$controller_runtime_root"
  "$controller_output_root" "$replay_repo_root" "$replay_venv_root"
  "$replay_runtime_root" "$replay_trace_root" "$replay_output_root"
  "$replay_diskann_root" "$replay_meta_root"
)
for path_value in "${path_values[@]}"; do
  [[ "$path_value" == /* ]] || die "topology paths must be absolute: $path_value"
  [[ "$path_value" != / ]] || die "topology paths must not be /"
  [[ "$path_value" != *$'\n'* && "$path_value" != *'|'* && "$path_value" != *'&'* ]] || \
    die "topology path contains unsupported shell characters: $path_value"
done
[[ "$replay_venv_root" == "$replay_repo_root/.venv" ]] || \
  die "replay_venv_root must equal replay_repo_root/.venv"
[[ "$storage_backend" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid storage_backend"

ssh_target="${replay_user}@${replay_host}"
if [[ -n "$replay_jump_user" ]]; then
  ssh_target="${replay_jump_user}@${replay_host}"
  sudo_prefix="sudo -n -u $replay_user"
  rsync_remote_path="$sudo_prefix rsync"
else
  sudo_prefix=""
  rsync_remote_path="rsync"
fi

shell_quote() {
  printf '%q' "$1"
}

remote_exec() {
  local command="$1" quoted_command ssh_command
  printf -v quoted_command '%q' "$command"
  if [[ -n "$sudo_prefix" ]]; then
    ssh_command="$sudo_prefix bash -lc $quoted_command"
  else
    ssh_command="bash -lc $quoted_command"
  fi
  if [[ "$dry_run" == true ]]; then
    printf '[DRY-RUN] ssh -o BatchMode=yes -o RemoteCommand=none -o RequestTTY=no -p %s %s %s\n' \
      "$replay_port" "$ssh_target" "$ssh_command"
    return 0
  fi
  ssh -o BatchMode=yes -o RemoteCommand=none -o RequestTTY=no \
    -p "$replay_port" "$ssh_target" "$ssh_command"
}

remote_path_exists() {
  local path="$1" remote_command ssh_command quoted_remote
  remote_command="test -e $(shell_quote "$path") || test -L $(shell_quote "$path")"
  if [[ -n "$sudo_prefix" ]]; then
    printf -v quoted_remote '%q' "$remote_command"
    ssh_command="$sudo_prefix bash -lc $quoted_remote"
  else
    ssh_command="$remote_command"
  fi
  if [[ "$dry_run" == true ]]; then
    return 1
  fi
  ssh -o BatchMode=yes -o RemoteCommand=none -o RequestTTY=no \
    -p "$replay_port" "$ssh_target" "$ssh_command"
}

require_transport_command() {
  [[ "$dry_run" == true ]] && return
  command -v "$1" >/dev/null 2>&1 || die "$1 is required on the controller"
}

rsync_ssh_command="ssh -o BatchMode=yes -o RemoteCommand=none -o RequestTTY=no -p $replay_port"

transfer_file() {
  local local_path="$1" remote_path="$2"
  [[ -f "$local_path" ]] || die "controller file not found: $local_path"
  if remote_path_exists "$remote_path"; then
    warn "remote file already exists; not overwriting: $remote_path"
    return
  fi
  remote_exec "mkdir -p $(shell_quote "$(dirname -- "$remote_path")")"
  if [[ "$dry_run" == true ]]; then
    printf '[DRY-RUN] %s %s -> %s:%s\n' "$transfer_method" "$local_path" "$ssh_target" "$remote_path"
    return
  fi
  require_transport_command "$transfer_method"
  if [[ "$transfer_method" == rsync ]]; then
    rsync -a --ignore-existing -e "$rsync_ssh_command" --rsync-path "$rsync_remote_path" \
      -- "$local_path" "$ssh_target:$remote_path"
  else
    scp -q -P "$replay_port" -o BatchMode=yes -- "$local_path" "$ssh_target:$remote_path"
  fi
}

transfer_directory() {
  local local_path="$1" remote_path="$2"
  [[ -d "$local_path" ]] || die "controller directory not found: $local_path"
  remote_exec "mkdir -p $(shell_quote "$remote_path")"
  if [[ "$dry_run" == true ]]; then
    printf '[DRY-RUN] %s directory %s -> %s:%s\n' "$transfer_method" "$local_path" "$ssh_target" "$remote_path"
    return
  fi
  require_transport_command "$transfer_method"
  if [[ "$transfer_method" == rsync ]]; then
    rsync -a --delete --exclude='.venv' --exclude='__pycache__' \
      -e "$rsync_ssh_command" --rsync-path "$rsync_remote_path" \
      -- "$local_path/" "$ssh_target:$remote_path/"
  else
    scp -q -r -P "$replay_port" -o BatchMode=yes -- "$local_path/." "$ssh_target:$remote_path/"
  fi
}

retrieve_directory() {
  local remote_path="$1" local_path="$2"
  if [[ "$dry_run" == true ]]; then
    printf '[DRY-RUN] retrieve %s:%s -> %s\n' "$ssh_target" "$remote_path" "$local_path"
    return
  fi
  if ! remote_path_exists "$remote_path"; then
    warn "remote result directory does not exist: $remote_path"
    return 1
  fi
  mkdir -p -- "$(dirname -- "$local_path")"
  mkdir -- "$local_path"
  if [[ "$transfer_method" == rsync ]]; then
    rsync -a --ignore-existing -e "$rsync_ssh_command" --rsync-path "$rsync_remote_path" \
      -- "$ssh_target:$remote_path/" "$local_path/"
  else
    scp -q -r -P "$replay_port" -o BatchMode=yes -- "$ssh_target:$remote_path/." "$local_path/"
  fi
}

validate_asset() {
  local asset="$1"
  [[ "$asset" == *.tar.zst ]] || die "asset must end in .tar.zst: $asset"
  case "$asset" in
    ""|/*|../*|*/../*|*/..|*' '*|*$'\n') die "asset must be a safe relative path: $asset" ;;
  esac
}

prepare_trace_asset() {
  local asset="$1" local_archive local_checksum remote_archive remote_checksum
  validate_asset "$asset"
  local_archive="$controller_trace_root/$asset"
  local_checksum="$local_archive.sha256"
  remote_archive="$replay_trace_root/$asset"
  remote_checksum="$remote_archive.sha256"
  [[ -f "$local_archive" ]] || die "trace archive not found: $local_archive"
  [[ -f "$local_checksum" ]] || die "trace checksum not found: $local_checksum"

  transfer_file "$local_archive" "$remote_archive"
  transfer_file "$local_checksum" "$remote_checksum"

  local category archive_name="${asset##*/}" trace_name
  if [[ "$asset" == */* ]]; then
    category="${asset%/*}"
  else
    category="."
  fi
  trace_name="${archive_name%.tar.zst}"
  local remote_category="$replay_trace_root/$category"
  local remote_trace_dir="$remote_category/$trace_name"
  if remote_path_exists "$remote_trace_dir"; then
    warn "remote extracted trace already exists; not overwriting: $remote_trace_dir"
  else
    remote_exec "set -euo pipefail; cd $(shell_quote "$(dirname -- "$remote_archive")"); sha256sum -c $(shell_quote "${remote_checksum##*/}"); mkdir -p $(shell_quote "$remote_category"); tar --zstd --keep-old-files -xf $(shell_quote "$remote_archive") -C $(shell_quote "$remote_category")"
  fi
  info "Trace staged: $asset"
}

check_remote_prerequisites() {
  local command
  command="set -euo pipefail"$'\n'
  command+='required=(bash tar zstd find sed ln mkdir dirname pwd sha256sum findmnt)'$'\n'
  if [[ "$transfer_method" == rsync ]]; then
    command+='required+=(rsync)'$'\n'
  fi
  if [[ "$replay_require_docker" == true ]]; then
    command+='required+=(docker)'$'\n'
  fi
  command+='missing=(); for name in "${required[@]}"; do command -v "$name" >/dev/null 2>&1 || missing+=("$name"); done'$'\n'
  command+='if ((${#missing[@]})); then echo "Missing replay-node commands: ${missing[*]}" >&2; exit 1; fi'$'\n'
  command+="base_python=$(shell_quote "$replay_python")"$'\n'
  command+='[[ -x "$base_python" ]] || { echo "Replay Python not found: $base_python" >&2; exit 1; }'$'\n'
  command+='"$base_python" -c '\''import sys; assert sys.version_info >= (3, 10), sys.version; import ensurepip, venv'\'''$'\n'
  if [[ "$replay_require_docker" == true ]]; then
    command+='docker compose version >/dev/null'$'\n'
    command+='docker info >/dev/null'$'\n'
  fi
  command+='echo "Replay-node prerequisites: OK"'$'\n'
  remote_exec "$command"
}

verify_remote_runtime() {
  local command
  command="set -euo pipefail"$'\n'
  command+="venv=$(shell_quote "$replay_venv_root")"$'\n'
  command+="repo=$(shell_quote "$replay_repo_root")"$'\n'
  command+='[[ -x "$venv/bin/python" ]] || { echo "Replay venv missing: $venv" >&2; exit 1; }'$'\n'
  command+='cd "$repo"'$'\n'
  command+='source "$venv/bin/activate"'$'\n'
  command+='python -c '\''import milvus_trace, pyarrow, pymilvus, yaml'\'''$'\n'
  command+='python -m pip check'$'\n'
  remote_exec "$command"
}

prepare_remote_runtime() {
  transfer_directory "$controller_repo_root/milvus_trace" "$replay_repo_root/milvus_trace"
  transfer_directory "$controller_runtime_root" "$replay_runtime_root"
  check_remote_prerequisites
  remote_exec "set -euo pipefail; cd $(shell_quote "$replay_runtime_root"); sha256sum -c SHA256SUMS"

  if remote_path_exists "$replay_venv_root"; then
    warn "remote venv already exists; verifying without overwriting: $replay_venv_root"
    verify_remote_runtime
  else
    local command
    command="set -euo pipefail"$'\n'
    command+="python=$(shell_quote "$replay_python")"$'\n'
    command+="venv=$(shell_quote "$replay_venv_root")"$'\n'
    command+="runtime=$(shell_quote "$replay_runtime_root")"$'\n'
    command+="requirements=$(shell_quote "$replay_runtime_requirements")"$'\n'
    command+='cd "$runtime"; sha256sum -c SHA256SUMS'$'\n'
    command+='"$python" -m venv "$venv"'$'\n'
    command+='source "$venv/bin/activate"'$'\n'
    command+='PIP_NO_INDEX=1 python -m pip install --no-index --find-links "$runtime/wheelhouse" -r "$runtime/$requirements"'$'\n'
    remote_exec "$command"
    verify_remote_runtime
  fi

  if [[ "$replay_require_docker" == true ]]; then
    remote_exec "set -euo pipefail; docker load --input $(shell_quote "$replay_runtime_root/$replay_container_images")"
  fi
  info "Offline replay runtime prepared on $ssh_target"
}

validate_run_name() {
  [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]] || die "run-name contains unsupported characters: $1"
}

expand_placeholder() {
  local value="$1"
  value="${value//@REPO_ROOT@/$replay_repo_root}"
  value="${value//@VENV_ROOT@/$replay_venv_root}"
  value="${value//@TRACE_ROOT@/$replay_trace_root}"
  value="${value//@OUTPUT_ROOT@/$remote_output_root}"
  value="${value//@DISKANN_ROOT@/$replay_diskann_root}"
  value="${value//@META_ROOT@/$replay_meta_root}"
  value="${value//@RUN_NAME@/$run_name}"
  value="${value//@STORAGE_BACKEND@/$storage_backend}"
  value="${value//@EXPECTED_FSTYPE@/$expected_fstype}"
  value="${value//@MILVUS_URI@/$replay_milvus_uri}"
  printf '%s\n' "$value"
}

remove_run_output() {
  local remote_path="$1" local_path="$2"
  if remote_path_exists "$remote_path"; then
    remote_exec "set -euo pipefail; path=$(shell_quote "$remote_path"); root=$(shell_quote "$replay_output_root"); [[ \"\$path\" == \"\$root\"/* && \"\$path\" != \"\$root\" ]] || exit 1; [[ ! -L \"\$path\" ]] || exit 1; rm -rf --one-file-system -- \"\$path\""
  fi
  if [[ -e "$local_path" || -L "$local_path" ]]; then
    [[ ! -L "$local_path" ]] || die "controller output is a symlink: $local_path"
    [[ "$local_path" == "$controller_output_root"/* && "$local_path" != "$controller_output_root" ]] || \
      die "controller run output is outside output root: $local_path"
    rm -rf --one-file-system -- "$local_path"
  fi
}

replay_run() {
  [[ -n "$run_name" ]] || die "--run-name is required for replay"
  validate_run_name "$run_name"
  ((${#command_args[@]} > 0)) || die "replay requires a command after --"
  remote_output_root="$replay_output_root/$run_name"
  local local_output_root="$controller_output_root/$run_name"

  if [[ "$dry_run" == false ]]; then
    if [[ "$overwrite_output" == true ]]; then
      remove_run_output "$remote_output_root" "$local_output_root"
    else
      ! remote_path_exists "$remote_output_root" || die "remote output exists; choose a new run-name or use --overwrite-output"
      [[ ! -e "$local_output_root" && ! -L "$local_output_root" ]] || \
        die "controller output exists; choose a new run-name or use --overwrite-output"
    fi
  fi

  local expanded=() argument
  for argument in "${command_args[@]}"; do
    expanded+=("$(expand_placeholder "$argument")")
  done
  local command="set -euo pipefail; cd $(shell_quote "$replay_repo_root");"
  for argument in "${expanded[@]}"; do
    command+=" $(shell_quote "$argument")"
  done

  info "Starting staged remote replay: $run_name ($storage_backend)"
  local replay_status=0 retrieve_status=0
  set +e
  remote_exec "$command"
  replay_status=$?
  set -e
  retrieve_directory "$remote_output_root" "$local_output_root" || retrieve_status=$?
  if [[ "$dry_run" == true ]]; then
    return "$replay_status"
  fi
  mkdir -p -- "$local_output_root"
  printf '%s\n' "$replay_status" > "$local_output_root/remote_exit_code"
  if ((replay_status != 0)); then
    warn "remote replay exited with status $replay_status; partial output was retrieved"
    return "$replay_status"
  fi
  info "Replay results retrieved: $local_output_root"
  return "$retrieve_status"
}

reset_path() {
  local remote_path="$1" label="$2" mode="$3" command
  info "Resetting $label on $ssh_target: $remote_path"
  command="set -euo pipefail"$'\n'
  command+="path=$(shell_quote "$remote_path")"$'\n'
  command+='[[ "$path" != / && ! -L "$path" ]] || { echo "unsafe reset path: $path" >&2; exit 1; }'$'\n'
  if [[ "$mode" == clear ]]; then
    command+='mkdir -p -- "$path"; find "$path" -xdev -mindepth 1 -delete'$'\n'
  else
    command+='if [[ -e "$path" ]]; then rm -rf --one-file-system -- "$path"; fi; mkdir -p -- "$path"'$'\n'
  fi
  remote_exec "$command"
}

reset_replay_node() {
  ((${#reset_targets[@]} > 0)) || die "reset requires at least one --target"
  local expanded=() ordered=() target seen item
  for target in "${reset_targets[@]}"; do
    if [[ "$target" == all ]]; then
      expanded+=(repo runtime trace output diskann)
    else
      expanded+=("$target")
    fi
  done
  for target in "${expanded[@]}"; do
    case "$target" in repo|runtime|trace|output|diskann) ;; *) die "unknown reset target: $target" ;; esac
    seen=false
    for item in ${ordered[@]+"${ordered[@]}"}; do [[ "$item" == "$target" ]] && seen=true; done
    [[ "$seen" == true ]] || ordered+=("$target")
  done
  for target in "${ordered[@]}"; do
    case "$target" in
      repo) reset_path "$replay_repo_root" "replay repository" wipe ;;
      runtime) reset_path "$replay_runtime_root" "offline runtime" wipe ;;
      trace) reset_path "$replay_trace_root" "replay trace root" wipe ;;
      output) reset_path "$replay_output_root" "replay output root" wipe ;;
      diskann) reset_path "$replay_diskann_root" "DiskANN data directory contents" clear ;;
    esac
  done
}

case "$phase" in
  check-prerequisites) check_remote_prerequisites ;;
  prepare-trace)
    ((${#assets[@]} > 0)) || die "prepare-trace requires at least one --asset"
    for asset in "${assets[@]}"; do prepare_trace_asset "$asset"; done
    ;;
  prepare-replay) prepare_remote_runtime ;;
  replay) replay_run ;;
  all)
    ((${#assets[@]} > 0)) || die "all requires at least one --asset"
    [[ -n "$run_name" ]] || die "all requires --run-name"
    for asset in "${assets[@]}"; do prepare_trace_asset "$asset"; done
    prepare_remote_runtime
    replay_run
    ;;
  reset) reset_replay_node ;;
esac
