"""Bounded-memory synthetic artifact record/replay benchmark."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path

from .artifact import verify_artifact
from .recorder import TraceConfig, TraceRecorder
from .replay import MilvusTraceReplayer, ReplayConfig


def peak_rss_kib() -> int:
    """Return peak resident memory in KiB on Linux/macOS."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value / 1024 if platform.system() == "Darwin" else value)


def vector(row: int, dimension: int) -> list[float]:
    return [((row + column) % 97) / 97.0 for column in range(dimension)]


def record(args: argparse.Namespace) -> dict:
    artifact_dir = args.artifact_dir.resolve()
    recorder = TraceRecorder(
        TraceConfig(
            enabled=True,
            output_dir=str(artifact_dir),
            max_queue_bytes=args.queue_bytes,
            rows_per_shard=args.rows_per_shard,
            compression="zstd",
        )
    )
    recorder.describe_collection(
        name=args.collection,
        dimension=args.dimension,
        auto_id=True,
        consistency_level="Eventually",
    )
    for start in range(0, args.rows, args.batch_size):
        end = min(start + args.batch_size, args.rows)
        rows = [
            {"vector": vector(row, args.dimension), "text": f"synthetic-{row}"}
            for row in range(start, end)
        ]
        recorder.record_insert(args.collection, rows)
        if args.producer_delay_ms:
            time.sleep(args.producer_delay_ms / 1000)
    recorder.close()
    manifest = verify_artifact(artifact_dir)
    return {
        "mode": "record",
        "rows": args.rows,
        "dimension": args.dimension,
        "peak_rss_kib": peak_rss_kib(),
        "artifact_dir": str(artifact_dir),
        "artifact_bytes": sum(path.stat().st_size for path in artifact_dir.iterdir()),
        "manifest": manifest,
    }


def replay(args: argparse.Namespace) -> dict:
    result = MilvusTraceReplayer(
        ReplayConfig(
            artifact_dir=args.artifact_dir.resolve(),
            uri=args.uri,
            collection=args.collection,
            timing="none",
            max_in_flight=args.max_in_flight,
            bootstrap_batch_size=args.batch_size,
            result_file=args.result_file,
        )
    ).run()
    result["stress"] = {
        "mode": "replay",
        "peak_rss_kib": peak_rss_kib(),
        "artifact_dir": str(args.artifact_dir.resolve()),
    }
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("record", "replay"), required=True)
    result.add_argument("--artifact-dir", type=Path, required=True)
    result.add_argument("--collection", default="ragperf_stress")
    result.add_argument("--rows", type=int, default=1_000_000)
    result.add_argument("--dimension", type=int, default=128)
    result.add_argument("--batch-size", type=int, default=1024)
    result.add_argument("--rows-per-shard", type=int, default=65_536)
    result.add_argument("--queue-bytes", type=int, default=256 * 1024 * 1024)
    result.add_argument("--producer-delay-ms", type=float, default=0.0)
    result.add_argument("--uri", default="")
    result.add_argument("--max-in-flight", type=int, default=16)
    result.add_argument("--result-file", type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.rows <= 0 or args.dimension <= 0 or args.batch_size <= 0:
        raise ValueError("rows, dimension, and batch-size must be positive")
    if args.mode == "record":
        output = record(args)
    else:
        if not args.uri:
            raise ValueError("--uri is required in replay mode")
        output = replay(args)
    print(json.dumps(output, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
