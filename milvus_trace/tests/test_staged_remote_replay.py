import hashlib
import os
import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "replayer"
    / "staged_remote_replay.sh"
)


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _write_fake_transport(fake_bin: Path) -> None:
    fake_bin.mkdir(parents=True)
    _write_executable(
        fake_bin / "ssh",
        """#!/usr/bin/env bash
set -euo pipefail
while (($#)); do
  case "$1" in
    -o|-p) shift 2 ;;
    *) target="$1"; shift; break ;;
  esac
done
exec bash -lc "$*"
""",
    )
    _write_executable(
        fake_bin / "scp",
        """#!/usr/bin/env bash
set -euo pipefail
recursive=false
args=()
while (($#)); do
  case "$1" in
    -q|--) shift ;;
    -r) recursive=true; shift ;;
    -P|-o) shift 2 ;;
    *) args+=("$1"); shift ;;
  esac
done
src="${args[${#args[@]}-2]}"
dst="${args[${#args[@]}-1]}"
map_path() { case "$1" in *@*:*) printf '%s\n' "${1#*:}" ;; *) printf '%s\n' "$1" ;; esac; }
src_path="$(map_path "$src")"
dst_path="$(map_path "$dst")"
if [[ "$recursive" == true ]]; then
  mkdir -p -- "$dst_path"
  cp -a -- "$src_path" "$dst_path"
else
  mkdir -p -- "$(dirname -- "$dst_path")"
  cp -a -- "$src_path" "$dst_path"
fi
""",
    )


def _write_fake_python(path: Path) -> None:
    _write_executable(
        path,
        """#!/usr/bin/env bash
if [[ "${1:-}" == "-c" && "${2:-}" == *sys.version_info* ]]; then
  printf '3.12\n'
fi
exit 0
""",
    )


def _write_checksum(root: Path, relative_paths: list[str]) -> None:
    lines = []
    for relative in relative_paths:
        digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        lines.append(f"{digest}  {relative}\n")
    (root / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


def _write_topology(
    tmp_path: Path,
    controller_repo: Path,
    controller_traces: Path,
    controller_runtime: Path,
    remote: Path,
) -> Path:
    topology = tmp_path / "topology.yaml"
    topology.write_text(
        f"""controller_repo_root: {controller_repo}
controller_trace_root: {controller_traces}
controller_runtime_root: {controller_runtime}
controller_output_root: {tmp_path / "controller/outputs"}
replay_host: fake-replay
replay_user: fake
replay_port: 22
transfer_method: scp
replay_repo_root: {remote / "ragperf"}
replay_venv_root: {remote / "ragperf/.venv"}
replay_runtime_root: {remote / "runtime"}
replay_trace_root: {remote / "traces"}
replay_output_root: {remote / "outputs"}
replay_meta_root: {remote / "meta"}
storage_backend: xfs
replay_diskann_root: {remote / "diskann"}
expected_fstype: xfs
replay_python: {tmp_path / "fake-bin/python3.12"}
replay_runtime_requirements: requirements-replay.lock
replay_require_docker: false
replay_milvus_uri: http://127.0.0.1:19530
""",
        encoding="utf-8",
    )
    return topology


def test_all_stages_local_assets_and_retrieves_failed_run(tmp_path: Path) -> None:
    controller_repo = tmp_path / "controller/RAGPerf"
    controller_module = controller_repo / "milvus_trace"
    controller_module.mkdir(parents=True)
    (controller_module / "__init__.py").write_text("", encoding="utf-8")

    controller_runtime = tmp_path / "controller/runtime"
    (controller_runtime / "wheelhouse").mkdir(parents=True)
    (controller_runtime / "requirements-replay.lock").write_text("", encoding="utf-8")
    _write_checksum(controller_runtime, ["requirements-replay.lock"])

    controller_traces = tmp_path / "controller/traces"
    archive_dir = controller_traces / "vector"
    archive_dir.mkdir(parents=True)
    payload = tmp_path / "payload/gpu-smoke"
    payload.mkdir(parents=True)
    (payload / "workload-manifest.yaml").write_text(
        "format: ragperf-milvus-trace\n", encoding="utf-8"
    )
    archive = archive_dir / "gpu-smoke.tar.zst"
    subprocess.run(
        [
            "tar",
            "--zstd",
            "-cf",
            str(archive),
            "-C",
            str(payload.parent),
            payload.name,
        ],
        check=True,
    )
    archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_name(f"{archive.name}.sha256").write_text(
        f"{archive_digest}  {archive.name}\n", encoding="utf-8"
    )

    fake_bin = tmp_path / "fake-bin"
    _write_fake_transport(fake_bin)
    _write_fake_python(fake_bin / "python3.12")
    remote = tmp_path / "remote"
    remote_venv = remote / "ragperf/.venv/bin"
    remote_venv.mkdir(parents=True)
    (remote_venv / "python").symlink_to(fake_bin / "python3.12")
    (remote_venv / "activate").write_text(
        f'export PATH="{remote_venv}:{os.environ["PATH"]}"\n',
        encoding="utf-8",
    )
    (remote / "diskann").mkdir(parents=True)
    topology = _write_topology(
        tmp_path, controller_repo, controller_traces, controller_runtime, remote
    )

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "all",
            "--topology",
            str(topology),
            "--asset",
            "vector/gpu-smoke.tar.zst",
            "--run-name",
            "gpu-smoke-xfs-r1",
            "--",
            "bash",
            "-c",
            "mkdir -p @OUTPUT_ROOT@; printf result > @OUTPUT_ROOT@/result.txt; exit 7",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 7, result.stderr
    assert "Replay-node prerequisites: OK" in result.stdout
    assert (remote / "ragperf/milvus_trace/__init__.py").is_file()
    assert (remote / "runtime/requirements-replay.lock").is_file()
    assert (remote / "traces/vector/gpu-smoke/workload-manifest.yaml").is_file()
    assert (remote / "outputs/gpu-smoke-xfs-r1/result.txt").read_text() == "result"
    controller_result = tmp_path / "controller/outputs/gpu-smoke-xfs-r1"
    assert (controller_result / "result.txt").read_text() == "result"
    assert (controller_result / "remote_exit_code").read_text().strip() == "7"


def test_replay_dry_run_expands_diskann_placeholders(tmp_path: Path) -> None:
    controller_repo = tmp_path / "controller/RAGPerf"
    (controller_repo / "milvus_trace").mkdir(parents=True)
    controller_traces = tmp_path / "controller/traces"
    controller_traces.mkdir(parents=True)
    controller_runtime = tmp_path / "controller/runtime"
    controller_runtime.mkdir(parents=True)
    remote = tmp_path / "remote"
    topology = _write_topology(
        tmp_path, controller_repo, controller_traces, controller_runtime, remote
    )

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "replay",
            "--topology",
            str(topology),
            "--run-name",
            "dry-run",
            "--dry-run",
            "--",
            "printf",
            "%s %s %s %s",
            "@DISKANN_ROOT@",
            "@STORAGE_BACKEND@",
            "@EXPECTED_FSTYPE@",
            "@OUTPUT_ROOT@",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert str(remote / "diskann") in result.stdout
    assert "xfs" in result.stdout
    assert str(remote / "outputs/dry-run") in result.stdout


def test_reset_diskann_keeps_mountpoint_directory(tmp_path: Path) -> None:
    controller_repo = tmp_path / "controller/RAGPerf"
    (controller_repo / "milvus_trace").mkdir(parents=True)
    controller_traces = tmp_path / "controller/traces"
    controller_traces.mkdir(parents=True)
    controller_runtime = tmp_path / "controller/runtime"
    controller_runtime.mkdir(parents=True)
    remote = tmp_path / "remote"
    diskann = remote / "diskann"
    (diskann / "old/nested").mkdir(parents=True)
    (diskann / "old/nested/data").write_text("old", encoding="utf-8")
    inode_before = diskann.stat().st_ino
    topology = _write_topology(
        tmp_path, controller_repo, controller_traces, controller_runtime, remote
    )
    fake_bin = tmp_path / "fake-bin"
    _write_fake_transport(fake_bin)
    _write_fake_python(fake_bin / "python3.12")
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "reset",
            "--topology",
            str(topology),
            "--target",
            "diskann",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert diskann.stat().st_ino == inode_before
    assert list(diskann.iterdir()) == []


def test_topology_rejects_root_diskann_path(tmp_path: Path) -> None:
    topology = tmp_path / "unsafe.yaml"
    topology.write_text(
        """controller_repo_root: /controller/repo
controller_trace_root: /controller/traces
controller_runtime_root: /controller/runtime
controller_output_root: /controller/output
replay_host: host
replay_user: user
replay_port: 22
transfer_method: scp
replay_repo_root: /replay/repo
replay_venv_root: /replay/repo/.venv
replay_runtime_root: /replay/runtime
replay_trace_root: /replay/traces
replay_output_root: /replay/output
replay_meta_root: /replay/meta
storage_backend: xfs
replay_diskann_root: /
expected_fstype: xfs
replay_python: /usr/bin/python3
replay_runtime_requirements: requirements-replay.lock
replay_milvus_uri: http://127.0.0.1:19530
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", str(SCRIPT), "replay", "--topology", str(topology), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "topology paths must not be /" in result.stderr
