"""Generate a bounded-memory, GPU-backed synthetic vector trace artifact."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Iterator

import yaml

from milvus_trace.artifact import verify_artifact
from milvus_trace.recorder import TraceConfig, TraceRecorder


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "milvus_trace/configs/recorder/vector-workloads.yaml"


class WorkloadConfigError(ValueError):
    """Invalid workload preset or override."""


@dataclasses.dataclass(frozen=True)
class WorkloadPlan:
    preset: str
    description: str
    rows: int
    dimension: int
    dtype_bytes: int
    target_vector_bytes: int | None
    logical_vector_bytes: int
    storage_overhead_factor: float
    estimated_disk_bytes: int
    query_count: int
    query_qps: float
    batch_size: int
    query_batch_size: int
    rows_per_shard: int
    max_queue_bytes: int
    producer_delay_ms: float
    metric_type: str
    index_type: str
    index_params: dict[str, Any]
    search_list: int
    top_k: int
    seed: int
    query_seed: int
    device: str

    @property
    def vector_bytes_per_row(self) -> int:
        return self.dimension * self.dtype_bytes

    def as_dict(self) -> dict[str, Any]:
        return {
            **dataclasses.asdict(self),
            "vector_bytes_per_row": self.vector_bytes_per_row,
        }


@dataclasses.dataclass(frozen=True)
class GeneratorRuntime:
    device: str
    device_name: str
    framework: str
    framework_version: str
    device_index: int | None = None


def _positive_int(value: Any, name: str) -> int:
    try:
        converted = int(value)
    except (TypeError, ValueError) as error:
        raise WorkloadConfigError(f"{name} must be an integer") from error
    if converted <= 0:
        raise WorkloadConfigError(f"{name} must be positive")
    return converted


def _positive_float(value: Any, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise WorkloadConfigError(f"{name} must be a number") from error
    if not math.isfinite(converted) or converted <= 0:
        raise WorkloadConfigError(f"{name} must be finite and positive")
    return converted


def _nonnegative_float(value: Any, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise WorkloadConfigError(f"{name} must be a number") from error
    if not math.isfinite(converted) or converted < 0:
        raise WorkloadConfigError(f"{name} must be finite and non-negative")
    return converted


def load_plan(
    config_path: Path,
    preset: str,
    *,
    storage_overhead_factor: float = 1.0,
    overrides: dict[str, Any] | None = None,
) -> WorkloadPlan:
    with config_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise WorkloadConfigError("workload config must contain a mapping")
    if payload.get("format") != "ragperf-vector-workload-presets-v1":
        raise WorkloadConfigError("unsupported workload config format")
    defaults = payload.get("defaults")
    presets = payload.get("presets")
    if not isinstance(defaults, dict) or not isinstance(presets, dict):
        raise WorkloadConfigError("workload config needs defaults and presets mappings")
    selected = presets.get(preset)
    if not isinstance(selected, dict):
        choices = ", ".join(sorted(str(name) for name in presets))
        raise WorkloadConfigError(f"unknown preset {preset!r}; choose from {choices}")

    values = {**defaults, **selected}
    for key, value in (overrides or {}).items():
        if value is not None:
            values[key] = value
            if key == "rows":
                values.pop("target_vector_bytes", None)

    dimension = _positive_int(values.get("dimension"), "dimension")
    dtype_bytes = _positive_int(values.get("dtype_bytes"), "dtype_bytes")
    overhead = _positive_float(storage_overhead_factor, "storage_overhead_factor")
    target = values.get("target_vector_bytes")
    rows = values.get("rows")
    if target is not None and rows is not None:
        raise WorkloadConfigError(
            "preset must specify rows or target_vector_bytes, not both"
        )
    if target is not None:
        target = _positive_int(target, "target_vector_bytes")
        raw_budget = math.floor(target / overhead)
        rows = raw_budget // (dimension * dtype_bytes)
        if rows <= 0:
            raise WorkloadConfigError("target is smaller than one vector row")
    else:
        rows = _positive_int(rows, "rows")

    logical_bytes = rows * dimension * dtype_bytes
    return WorkloadPlan(
        preset=preset,
        description=str(values.get("description", "")),
        rows=rows,
        dimension=dimension,
        dtype_bytes=dtype_bytes,
        target_vector_bytes=target,
        logical_vector_bytes=logical_bytes,
        storage_overhead_factor=overhead,
        estimated_disk_bytes=math.ceil(logical_bytes * overhead),
        query_count=_positive_int(values.get("query_count"), "query_count"),
        query_qps=_positive_float(values.get("query_qps"), "query_qps"),
        batch_size=_positive_int(values.get("batch_size"), "batch_size"),
        query_batch_size=_positive_int(
            values.get("query_batch_size"), "query_batch_size"
        ),
        rows_per_shard=_positive_int(values.get("rows_per_shard"), "rows_per_shard"),
        max_queue_bytes=_positive_int(values.get("max_queue_bytes"), "max_queue_bytes"),
        producer_delay_ms=_nonnegative_float(
            values.get("producer_delay_ms", 0), "producer_delay_ms"
        ),
        metric_type=str(values.get("metric_type", "COSINE")),
        index_type=str(values.get("index_type", "DISKANN")),
        index_params=dict(values.get("index_params") or {}),
        search_list=_positive_int(values.get("search_list"), "search_list"),
        top_k=_positive_int(values.get("top_k"), "top_k"),
        seed=int(values.get("seed")),
        query_seed=int(values.get("query_seed")),
        device=str(values.get("device", "auto")),
    )


def resolve_runtime(requested: str) -> GeneratorRuntime:
    if requested == "auto" or requested.startswith("cuda"):
        try:
            import cupy

            device_index = int(requested.split(":", 1)[1]) if ":" in requested else 0
            with cupy.cuda.Device(device_index):
                cupy.empty(1, dtype=cupy.float32).sum().get()
                properties = cupy.cuda.runtime.getDeviceProperties(device_index)
            raw_name = properties["name"]
            device_name = (
                raw_name.decode("utf-8")
                if isinstance(raw_name, bytes)
                else str(raw_name)
            )
            return GeneratorRuntime(
                device=f"cuda:{device_index}",
                device_name=device_name,
                framework="cupy",
                framework_version=str(cupy.__version__),
                device_index=device_index,
            )
        except Exception as error:
            if requested != "auto":
                raise WorkloadConfigError(
                    f"CuPy could not execute on requested device {requested}: {error}"
                ) from error

    import torch

    device = torch.device("cpu" if requested == "auto" else requested)
    if device.type != "cpu":
        raise WorkloadConfigError(f"unsupported generator device: {requested}")
    return GeneratorRuntime(
        device="cpu",
        device_name="CPU",
        framework="torch",
        framework_version=str(torch.__version__),
    )


def vector_batches(
    *,
    rows: int,
    dimension: int,
    batch_size: int,
    runtime: GeneratorRuntime,
    seed: int,
) -> Iterator[list[list[float]]]:
    if runtime.framework == "cupy":
        import cupy

        assert runtime.device_index is not None
        with cupy.cuda.Device(runtime.device_index):
            generator = cupy.random.default_rng(seed)
            for start in range(0, rows, batch_size):
                count = min(batch_size, rows - start)
                values = generator.standard_normal(
                    (count, dimension), dtype=cupy.float32
                )
                values /= cupy.linalg.norm(values, axis=1, keepdims=True)
                yield cupy.asnumpy(values).tolist()
        return

    import torch
    import torch.nn.functional as functional

    generator = torch.Generator(device=runtime.device)
    generator.manual_seed(seed)
    with torch.inference_mode():
        for start in range(0, rows, batch_size):
            count = min(batch_size, rows - start)
            values = torch.randn(
                (count, dimension),
                dtype=torch.float32,
                device=runtime.device,
                generator=generator,
            )
            values = functional.normalize(values, p=2, dim=1)
            yield values.cpu().tolist()


def record_artifact(
    plan: WorkloadPlan,
    artifact_dir: Path,
    *,
    collection: str,
    session_id: str,
) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    runtime = resolve_runtime(plan.device)
    started = time.monotonic_ns()
    recorder = TraceRecorder(
        TraceConfig(
            enabled=True,
            output_dir=str(artifact_dir),
            max_queue_bytes=plan.max_queue_bytes,
            rows_per_shard=plan.rows_per_shard,
            compression="zstd",
        ),
        session_id=session_id,
    )
    recorder.describe_collection(
        name=collection,
        dimension=plan.dimension,
        auto_id=True,
        consistency_level="Eventually",
        vector_field="vector",
        workload_preset=plan.preset,
        logical_vector_bytes=plan.logical_vector_bytes,
        storage_overhead_factor=plan.storage_overhead_factor,
        generator=f"{runtime.framework}.standard_normal+l2-normalize",
        generator_device=runtime.device,
        generator_device_name=runtime.device_name,
        generator_framework=runtime.framework,
        generator_framework_version=runtime.framework_version,
        corpus_seed=plan.seed,
        query_seed=plan.query_seed,
    )

    for vectors in vector_batches(
        rows=plan.rows,
        dimension=plan.dimension,
        batch_size=plan.batch_size,
        runtime=runtime,
        seed=plan.seed,
    ):
        recorder.record_insert(collection, [{"vector": vector} for vector in vectors])
        if plan.producer_delay_ms > 0:
            time.sleep(plan.producer_delay_ms / 1000)

    recorder.mark_index_created(
        field_name="vector",
        metric_type=plan.metric_type,
        index_type=plan.index_type,
        index_name=f"{plan.index_type}_{plan.metric_type}",
        params=plan.index_params,
    )
    marker_ns = time.monotonic_ns()
    recorder.begin_timed_workload(marker_ns)
    query_index = 0
    for vectors in vector_batches(
        rows=plan.query_count,
        dimension=plan.dimension,
        batch_size=plan.query_batch_size,
        runtime=runtime,
        seed=plan.query_seed,
    ):
        for vector in vectors:
            timestamp_ns = marker_ns + round(
                query_index * 1_000_000_000 / plan.query_qps
            )
            recorder.record_search(
                collection,
                [vector],
                timestamp_ns=timestamp_ns,
                limit=plan.top_k,
                search_params={
                    "metric_type": plan.metric_type,
                    "params": {"search_list": plan.search_list},
                },
                consistency_level="Eventually",
            )
            query_index += 1
    recorder.close()
    manifest = verify_artifact(artifact_dir)
    return {
        "plan": plan.as_dict(),
        "artifact_dir": str(artifact_dir),
        "artifact_bytes": sum(
            path.stat().st_size for path in artifact_dir.iterdir() if path.is_file()
        ),
        "elapsed_seconds": (time.monotonic_ns() - started) / 1_000_000_000,
        "generator_device": runtime.device,
        "generator_device_name": runtime.device_name,
        "generator_framework": runtime.framework,
        "generator_framework_version": runtime.framework_version,
        "manifest": manifest,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    result.add_argument("--preset", required=True)
    result.add_argument("--artifact-dir", type=Path)
    result.add_argument("--collection")
    result.add_argument("--session-id")
    result.add_argument("--storage-overhead-factor", type=float, default=1.0)
    result.add_argument("--device")
    result.add_argument("--rows", type=int)
    result.add_argument("--dimension", type=int)
    result.add_argument("--query-count", type=int)
    result.add_argument("--query-qps", type=float)
    result.add_argument("--batch-size", type=int)
    result.add_argument("--query-batch-size", type=int)
    result.add_argument("--rows-per-shard", type=int)
    result.add_argument("--max-queue-bytes", type=int)
    result.add_argument("--producer-delay-ms", type=float)
    result.add_argument("--plan-only", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    override_names = (
        "device",
        "rows",
        "dimension",
        "query_count",
        "query_qps",
        "batch_size",
        "query_batch_size",
        "rows_per_shard",
        "max_queue_bytes",
        "producer_delay_ms",
    )
    plan = load_plan(
        args.config.resolve(),
        args.preset,
        storage_overhead_factor=args.storage_overhead_factor,
        overrides={name: getattr(args, name) for name in override_names},
    )
    if args.plan_only:
        print(json.dumps(plan.as_dict(), indent=2))
        return 0
    if args.artifact_dir is None:
        raise WorkloadConfigError(
            "--artifact-dir is required unless --plan-only is used"
        )
    preset_label = re.sub(r"[^A-Za-z0-9_]", "_", args.preset)
    collection = args.collection or f"ragperf_vector_{preset_label}"
    session_id = args.session_id or f"vector-{args.preset}-seed-{plan.seed}"
    output = record_artifact(
        plan,
        args.artifact_dir,
        collection=collection,
        session_id=session_id,
    )
    print(json.dumps(output, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
