"""Run an xfs/3FS/pNFS DISKANN matrix through staged remote replay."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as datetime_module
import math
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "milvus_trace/configs/evaluation/diskann-experiments.yaml"
)
STAGED_SCRIPT = (
    PROJECT_ROOT / "milvus_trace/benchmarks/replayer/staged_remote_replay.sh"
)
REMOTE_REPLAY_SCRIPT = (
    "@REPO_ROOT@/milvus_trace/benchmarks/replayer/run_diskann_replay.sh"
)
SAFE_LABEL = re.compile(r"[^A-Za-z0-9._-]+")


class MatrixConfigError(ValueError):
    """Invalid matrix input or state."""


@dataclasses.dataclass(frozen=True)
class Workload:
    name: str
    asset: str
    artifact_path: str


@dataclasses.dataclass(frozen=True)
class Case:
    case_id: str
    run_name: str
    workload: str
    backend: str
    topology: str
    asset: str
    artifact_path: str
    time_scale: float
    repeat: int
    warmup: int
    max_in_flight: int
    bootstrap_batch_size: int
    state_file: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def safe_label(value: str) -> str:
    return SAFE_LABEL.sub("-", value).strip("-") or "unnamed"


def parse_csv(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise MatrixConfigError("comma-separated override must not be empty")
    return values


def parse_scales(raw: str | None) -> list[float] | None:
    values = parse_csv(raw)
    if values is None:
        return None
    result = []
    for item in values:
        try:
            scale = float(item)
        except ValueError as error:
            raise MatrixConfigError(f"invalid time scale: {item}") from error
        if not math.isfinite(scale) or scale <= 0:
            raise MatrixConfigError(f"time scale must be finite and positive: {item}")
        result.append(scale)
    return result


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise MatrixConfigError("experiment config must contain a mapping")
    if payload.get("format") != "ragperf-diskann-experiments-v1":
        raise MatrixConfigError("unsupported experiment config format")
    for key in ("backends", "workloads", "presets"):
        if not isinstance(payload.get(key), dict):
            raise MatrixConfigError(f"experiment config needs a {key} mapping")
    return payload


def parse_topology_overrides(raw_values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in raw_values:
        if "=" not in raw:
            raise MatrixConfigError("--topology must use NAME=PATH")
        name, raw_path = raw.split("=", 1)
        if not name or not raw_path:
            raise MatrixConfigError("--topology must use non-empty NAME=PATH")
        result[name] = Path(raw_path).expanduser().resolve()
    return result


def resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def parse_remote_path(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = raw.strip()
    if not value or not value.startswith("/") or value == "/":
        raise MatrixConfigError("--diskann-root must be a non-root absolute path")
    if any(character in value for character in ("\n", "|", "&")):
        raise MatrixConfigError("--diskann-root contains unsupported shell characters")
    return value


def build_cases(
    payload: dict[str, Any],
    preset_name: str,
    *,
    run_tag: str,
    state_root: Path,
    topology_overrides: dict[str, Path] | None = None,
    workloads_override: list[str] | None = None,
    backends_override: list[str] | None = None,
    scales_override: list[float] | None = None,
    repeats_override: int | None = None,
) -> list[Case]:
    preset = payload["presets"].get(preset_name)
    if not isinstance(preset, dict):
        choices = ", ".join(sorted(payload["presets"]))
        raise MatrixConfigError(
            f"unknown preset {preset_name!r}; choose from {choices}"
        )
    workloads = workloads_override or list(preset.get("workloads") or [])
    backends = backends_override or list(preset.get("backends") or [])
    scales = scales_override or [
        float(value) for value in preset.get("time_scales") or []
    ]
    if repeats_override is None:
        raise MatrixConfigError(
            "--repeats is required; choose the repeat count for this run"
        )
    repeats = repeats_override
    if not workloads or not backends or not scales or repeats <= 0:
        raise MatrixConfigError(
            "matrix workloads, backends, scales, and repeats are required"
        )
    warmup = int(preset.get("warmup", 0))
    max_in_flight = int(preset.get("max_in_flight", 0))
    bootstrap_batch_size = int(preset.get("bootstrap_batch_size", 0))
    if warmup < 0 or max_in_flight <= 0 or bootstrap_batch_size <= 0:
        raise MatrixConfigError("invalid replay limits in experiment preset")

    topology_overrides = topology_overrides or {}
    cases: list[Case] = []
    for workload_name in workloads:
        workload_value = payload["workloads"].get(workload_name)
        if not isinstance(workload_value, dict):
            raise MatrixConfigError(f"unknown workload: {workload_name}")
        workload = Workload(
            workload_name,
            str(workload_value.get("asset", "")),
            str(workload_value.get("artifact_path", "")),
        )
        if not workload.asset.endswith(".tar.zst") or not workload.artifact_path:
            raise MatrixConfigError(f"invalid workload asset mapping: {workload_name}")
        for backend in backends:
            backend_value = payload["backends"].get(backend)
            if not isinstance(backend_value, dict):
                raise MatrixConfigError(f"unknown backend: {backend}")
            topology = topology_overrides.get(backend)
            if topology is None:
                topology = resolve_path(str(backend_value.get("topology", "")))
            if not topology.is_file():
                raise MatrixConfigError(
                    f"topology file not found for {backend}: {topology}"
                )
            for scale in scales:
                if not math.isfinite(scale) or scale <= 0:
                    raise MatrixConfigError(f"invalid time scale: {scale}")
                scale_label = format(scale, ".12g")
                for repeat in range(1, repeats + 1):
                    run_name = safe_label(
                        f"diskann-{workload.name}-{backend}-ts{scale_label}-r{repeat}-{run_tag}"
                    )
                    case_id = "/".join(
                        (workload.name, backend, f"ts{scale_label}", f"r{repeat}")
                    )
                    state_file = (
                        state_root
                        / "cases"
                        / safe_label(workload.name)
                        / safe_label(backend)
                        / f"ts{safe_label(scale_label)}"
                        / f"r{repeat}.yaml"
                    )
                    cases.append(
                        Case(
                            case_id=case_id,
                            run_name=run_name,
                            workload=workload.name,
                            backend=backend,
                            topology=str(topology),
                            asset=workload.asset,
                            artifact_path=workload.artifact_path,
                            time_scale=scale,
                            repeat=repeat,
                            warmup=warmup,
                            max_in_flight=max_in_flight,
                            bootstrap_batch_size=bootstrap_batch_size,
                            state_file=str(state_file),
                        )
                    )
    return cases


def write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)
    temporary.replace(path)


def case_command(
    case: Case,
    *,
    diskann_root: str | None,
    overwrite_output: bool,
    dry_run: bool,
) -> list[str]:
    command = [
        "bash",
        str(STAGED_SCRIPT),
        "replay",
        "--topology",
        case.topology,
        "--run-name",
        case.run_name,
    ]
    if overwrite_output:
        command.append("--overwrite-output")
    if dry_run:
        command.append("--dry-run")
    if diskann_root is not None:
        command.extend(("--diskann-root", diskann_root))
    command.extend(
        [
            "--",
            "bash",
            REMOTE_REPLAY_SCRIPT,
            "--artifact-dir",
            f"@TRACE_ROOT@/{case.artifact_path}",
            "--backend",
            "@STORAGE_BACKEND@",
            "--expected-fstype",
            "@EXPECTED_FSTYPE@",
            "--diskann-root",
            "@DISKANN_ROOT@",
            "--meta-root",
            "@META_ROOT@",
            "--output-dir",
            "@OUTPUT_ROOT@",
            "--run-name",
            "@RUN_NAME@",
            "--uri",
            "@MILVUS_URI@",
            "--time-scale",
            format(case.time_scale, ".12g"),
            "--warmup",
            str(case.warmup),
            "--max-in-flight",
            str(case.max_in_flight),
            "--bootstrap-batch-size",
            str(case.bootstrap_batch_size),
        ]
    )
    return command


def prepare_command(
    phase: str, topology: str, *, asset: str | None, dry_run: bool
) -> list[str]:
    command = ["bash", str(STAGED_SCRIPT), phase, "--topology", topology]
    if asset is not None:
        command.extend(("--asset", asset))
    if dry_run:
        command.append("--dry-run")
    return command


def run_matrix(args: argparse.Namespace, cases: list[Case], state_root: Path) -> int:
    plan = {
        "format": "ragperf-diskann-matrix-plan-v1",
        "preset": args.preset,
        "run_tag": args.run_tag,
        "repeats": args.repeats,
        "diskann_root": args.diskann_root,
        "dry_run": args.dry_run,
        "case_count": len(cases),
        "cases": [case.as_dict() for case in cases],
    }
    write_yaml(state_root / "matrix-plan.yaml", plan)

    prepared_runtime: set[tuple[str, str]] = set()
    prepared_assets: set[tuple[str, str, str]] = set()
    failures = 0
    for case in cases:
        marker = Path(case.state_file)
        if marker.is_file() and not args.overwrite_output:
            existing = yaml.safe_load(marker.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and existing.get("status") == "ok":
                print(f"[SKIP] completed: {case.case_id}")
                continue

        if not args.skip_prepare:
            runtime_key = (case.backend, case.topology)
            if runtime_key not in prepared_runtime:
                subprocess.run(
                    prepare_command(
                        "prepare-replay",
                        case.topology,
                        asset=None,
                        dry_run=args.dry_run,
                    ),
                    check=True,
                )
                prepared_runtime.add(runtime_key)
            asset_key = (case.backend, case.topology, case.asset)
            if asset_key not in prepared_assets:
                subprocess.run(
                    prepare_command(
                        "prepare-trace",
                        case.topology,
                        asset=case.asset,
                        dry_run=args.dry_run,
                    ),
                    check=True,
                )
                prepared_assets.add(asset_key)

        started_at = datetime_module.datetime.now(
            datetime_module.timezone.utc
        ).isoformat()
        command = case_command(
            case,
            diskann_root=args.diskann_root,
            overwrite_output=args.overwrite_output,
            dry_run=args.dry_run,
        )
        print(f"[RUN] {case.case_id}: {' '.join(command)}")
        result = subprocess.run(command, check=False)
        status = (
            "dry-run"
            if args.dry_run
            else ("ok" if result.returncode == 0 else "failed")
        )
        write_yaml(
            marker,
            {
                **case.as_dict(),
                "status": status,
                "returncode": result.returncode,
                "started_at": started_at,
                "finished_at": datetime_module.datetime.now(
                    datetime_module.timezone.utc
                ).isoformat(),
                "command": command,
            },
        )
        if result.returncode != 0:
            failures += 1

    write_yaml(
        state_root / "matrix-summary.yaml",
        {
            "format": "ragperf-diskann-matrix-summary-v1",
            "preset": args.preset,
            "run_tag": args.run_tag,
            "repeats": args.repeats,
            "diskann_root": args.diskann_root,
            "case_count": len(cases),
            "failures": failures,
            "status": (
                "dry-run" if args.dry_run else ("ok" if failures == 0 else "failed")
            ),
        },
    )
    return 1 if failures else 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    result.add_argument("--preset", required=True)
    result.add_argument("--run-tag")
    result.add_argument("--state-root", type=Path)
    result.add_argument("--topology", action="append", default=[], metavar="NAME=PATH")
    result.add_argument("--workloads")
    result.add_argument("--backends")
    result.add_argument("--time-scales")
    result.add_argument("--repeats", type=int, required=True)
    result.add_argument(
        "--diskann-root",
        metavar="PATH",
        help="override replay_diskann_root for every topology",
    )
    result.add_argument("--skip-prepare", action="store_true")
    result.add_argument("--overwrite-output", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.repeats <= 0:
        raise MatrixConfigError("--repeats must be positive")
    args.diskann_root = parse_remote_path(args.diskann_root)
    args.run_tag = args.run_tag or datetime_module.datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    if safe_label(args.run_tag) != args.run_tag:
        raise MatrixConfigError(
            "--run-tag may contain only letters, digits, '.', '_', '-'"
        )
    state_root = (
        args.state_root.expanduser().resolve()
        if args.state_root
        else (
            PROJECT_ROOT / "milvus_trace/outputs/diskann-experiments" / args.run_tag
        ).resolve()
    )
    payload = load_config(args.config.expanduser().resolve())
    cases = build_cases(
        payload,
        args.preset,
        run_tag=args.run_tag,
        state_root=state_root,
        topology_overrides=parse_topology_overrides(args.topology),
        workloads_override=parse_csv(args.workloads),
        backends_override=parse_csv(args.backends),
        scales_override=parse_scales(args.time_scales),
        repeats_override=args.repeats,
    )
    return run_matrix(args, cases, state_root)


if __name__ == "__main__":
    raise SystemExit(main())
