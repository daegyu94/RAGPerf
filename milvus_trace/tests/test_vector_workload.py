import json
from pathlib import Path

import pytest

from milvus_trace.artifact import iter_manifest_rows, verify_artifact
from milvus_trace.recorder import TraceConfig, TraceRecorder
from milvus_trace.benchmarks.recorder.record_vector_workload import (
    DEFAULT_CONFIG,
    load_plan,
    record_artifact,
)


def test_decimal_tb_plan_never_exceeds_target() -> None:
    plan = load_plan(DEFAULT_CONFIG, "0.5tb")

    assert plan.rows == 162_760_416
    assert plan.vector_bytes_per_row == 3072
    assert plan.logical_vector_bytes <= 500_000_000_000
    assert 500_000_000_000 - plan.logical_vector_bytes < plan.vector_bytes_per_row


def test_storage_overhead_factor_reduces_logical_rows() -> None:
    plan = load_plan(DEFAULT_CONFIG, "1tb", storage_overhead_factor=1.25)

    assert plan.logical_vector_bytes <= 800_000_000_000
    assert plan.estimated_disk_bytes <= 1_000_000_000_000


def test_recorder_rejects_existing_trace_artifact(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "existing-trace"
    artifact_dir.mkdir()
    (artifact_dir / "old-shard.parquet").write_bytes(b"old")

    with pytest.raises(
        FileExistsError, match="trace artifact output directory is not empty"
    ):
        TraceRecorder(TraceConfig(enabled=True, output_dir=str(artifact_dir)))


def test_small_cpu_workload_records_diskann_corpus_and_timed_queries(
    tmp_path: Path,
) -> None:
    plan = load_plan(
        DEFAULT_CONFIG,
        "gpu-smoke",
        overrides={
            "device": "cpu",
            "rows": 12,
            "dimension": 8,
            "query_count": 3,
            "batch_size": 4,
            "query_batch_size": 2,
            "rows_per_shard": 5,
        },
    )
    artifact_dir = tmp_path / "cpu-smoke"

    result = record_artifact(
        plan,
        artifact_dir,
        collection="cpu_smoke",
        session_id="cpu-smoke-test",
    )

    manifest = verify_artifact(artifact_dir)
    assert result["generator_device"] == "cpu"
    assert manifest["collection"]["workload_preset"] == "gpu-smoke"
    assert manifest["index"]["index_type"] == "DISKANN"
    assert manifest["event_count"] == 3
    corpus = list(iter_manifest_rows(artifact_dir, manifest, "corpus"))
    assert len(corpus) == 12
    assert all(len(row["vector"]) == 8 for row in corpus)
    events = list(iter_manifest_rows(artifact_dir, manifest, "events"))
    assert [row["relative_timestamp_ns"] for row in events] == [
        0,
        50_000_000,
        100_000_000,
    ]
    searches = list(iter_manifest_rows(artifact_dir, manifest, "searches"))
    assert len(searches) == 3
    params = json.loads(events[0]["params_json"])
    assert params["search_params"]["params"]["search_list"] == 32
