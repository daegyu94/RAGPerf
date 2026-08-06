#!/usr/bin/env python
"""Generate or replay a count-sized synthetic vector workload."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vector_workload.scripts.common import (  # noqa: E402
    WORKLOAD_DIR,
    add_estimate_arguments,
    add_replay_arguments,
    ensure_empty_run_dir,
    print_artifact_summary,
    print_estimate,
    replay_workload,
    run_command,
)


def record(args: argparse.Namespace) -> None:
    if args.record_count <= 0 or args.query_count <= 0:
        raise ValueError("record-count and query-count must be positive")
    if args.dimension <= 0:
        raise ValueError("dimension must be positive")
    run_dir = ensure_empty_run_dir(args.output_dir)
    print_estimate(
        "synthetic",
        corpus_records=args.record_count,
        corpus_rows=args.record_count,
        query_records=args.query_count,
        query_rows=args.query_count,
        dimension=args.dimension,
        dtype=args.dtype,
        input_bytes=0,
        overhead=args.estimate_overhead,
        assumptions=[
            "corpus-count is exactly the corpus vector row count",
            "synthetic input has no downloaded dataset or embedding model",
        ],
    )
    run_command(
        [
            sys.executable,
            str(WORKLOAD_DIR / "generate_synthetic.py"),
            "--output-dir",
            str(run_dir / "artifact"),
            "--corpus-count",
            str(args.record_count),
            "--query-count",
            str(args.query_count),
            "--dimension",
            str(args.dimension),
            "--dtype",
            args.dtype,
            "--rows-per-shard",
            str(args.rows_per_shard),
            "--query-noise",
            str(args.query_noise),
            "--query-delay-ms",
            str(args.query_delay_ms),
            "--seed",
            str(args.seed),
        ]
    )
    print("Recorded dataset:")
    print_artifact_summary(run_dir / "artifact")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    record_parser = subparsers.add_parser("record", help="generate a synthetic workload")
    record_parser.add_argument("--output-dir", type=Path, required=True)
    record_parser.add_argument("--record-count", type=int, default=100_000)
    record_parser.add_argument("--query-count", type=int, default=2_000)
    record_parser.add_argument("--dimension", type=int, default=384)
    record_parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    record_parser.add_argument("--rows-per-shard", type=int, default=25_000)
    record_parser.add_argument("--query-noise", type=float, default=0.01)
    record_parser.add_argument("--query-delay-ms", type=float, default=0.0)
    record_parser.add_argument("--seed", type=int, default=42)
    add_estimate_arguments(record_parser)
    record_parser.set_defaults(func=record)

    replay_parser = subparsers.add_parser("replay", help="replay a recorded synthetic workload")
    add_replay_arguments(replay_parser)
    replay_parser.set_defaults(func=replay_workload)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
