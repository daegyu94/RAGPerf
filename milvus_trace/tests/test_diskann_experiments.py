from pathlib import Path

import pytest
import yaml

from milvus_trace.benchmarks.evaluation.run_diskann_experiments import (
    DEFAULT_CONFIG,
    MatrixConfigError,
    build_cases,
    case_command,
    load_config,
)


def test_smoke_matrix_covers_every_backend(tmp_path: Path) -> None:
    payload = load_config(DEFAULT_CONFIG)
    cases = build_cases(
        payload,
        "smoke",
        run_tag="unit",
        state_root=tmp_path,
    )

    assert len(cases) == 3
    assert {case.backend for case in cases} == {"xfs", "3fs", "pnfs"}
    assert {case.workload for case in cases} == {"gpu-smoke"}
    assert all(case.warmup == 1 for case in cases)


def test_capacity_override_builds_stable_case_order(tmp_path: Path) -> None:
    payload = load_config(DEFAULT_CONFIG)
    cases = build_cases(
        payload,
        "capacity",
        run_tag="unit",
        state_root=tmp_path,
        workloads_override=["0.5tb", "1tb"],
        backends_override=["xfs", "pnfs"],
        scales_override=[1, 2],
        repeats_override=2,
    )

    assert len(cases) == 16
    assert cases[0].case_id == "0.5tb/xfs/ts1/r1"
    assert cases[-1].case_id == "1tb/pnfs/ts2/r2"
    assert len({case.run_name for case in cases}) == len(cases)


def test_remote_command_uses_only_topology_placeholders(tmp_path: Path) -> None:
    case = build_cases(
        load_config(DEFAULT_CONFIG),
        "smoke",
        run_tag="unit",
        state_root=tmp_path,
        backends_override=["xfs"],
    )[0]
    command = case_command(case, overwrite_output=False, dry_run=True)

    assert "@DISKANN_ROOT@" in command
    assert "@EXPECTED_FSTYPE@" in command
    assert "@TRACE_ROOT@/vector/gpu-smoke" in command
    assert "@OUTPUT_ROOT@" in command
    assert "--dry-run" in command


def test_unknown_workload_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(MatrixConfigError, match="unknown workload"):
        build_cases(
            load_config(DEFAULT_CONFIG),
            "smoke",
            run_tag="unit",
            state_root=tmp_path,
            workloads_override=["missing"],
        )


def test_topologies_share_diskann_data_path() -> None:
    topology_dir = DEFAULT_CONFIG.parents[1] / "replayer" / "staged-remote"

    for backend in ("xfs", "3fs", "pnfs"):
        topology = yaml.safe_load(
            (topology_dir / f"{backend}.yaml").read_text(encoding="utf-8")
        )
        assert topology["replay_diskann_root"] == "/mnt/nvme/milvus-data"
