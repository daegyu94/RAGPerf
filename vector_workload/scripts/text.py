#!/usr/bin/env python
"""Record or replay a count-sized text embedding workload."""

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
    add_option,
    add_replay_arguments,
    add_schedule_arguments,
    ensure_empty_run_dir,
    infer_dimension,
    jsonl_record_count,
    jsonl_text_stats,
    print_artifact_summary,
    print_estimate,
    replay_workload,
    run_command,
)


DEFAULT_CORPUS_COUNT = 100_000
DEFAULT_QUERY_COUNT = 2_000


def record(args: argparse.Namespace) -> None:
    run_dir = ensure_empty_run_dir(args.output_dir)
    if args.record_count is not None and args.record_count <= 0:
        raise ValueError("record-count must be positive")
    if args.query_count is not None and args.query_count <= 0:
        raise ValueError("query-count must be positive")
    if (args.corpus_file is None) != (args.query_file is None):
        raise ValueError("corpus-file and query-file must be provided together")

    if args.corpus_file is None:
        if args.dataset != "wikipedia-nq":
            raise ValueError(f"unsupported dataset: {args.dataset}")
        corpus_count = args.record_count or DEFAULT_CORPUS_COUNT
        query_count = args.query_count or DEFAULT_QUERY_COUNT
        input_dir = run_dir / "input"
        run_command(
            [
                sys.executable,
                str(WORKLOAD_DIR / "prepare_workloads.py"),
                "wikipedia-nq",
                "--output-dir",
                str(input_dir),
                "--corpus-count",
                str(corpus_count),
                "--query-count",
                str(query_count),
            ]
        )
        corpus_file = input_dir / "corpus.jsonl"
        query_file = input_dir / "queries.jsonl"
        max_corpus = None
        max_queries = None
    else:
        corpus_file = args.corpus_file.resolve()
        query_file = args.query_file.resolve()
        max_corpus = args.record_count
        max_queries = args.query_count

    corpus_stats = jsonl_text_stats(
        corpus_file,
        args.chunk_size,
        args.chunk_overlap,
        max_corpus,
    )
    query_count_used = jsonl_record_count(query_file, max_queries)
    dimension = infer_dimension(
        args.model,
        args.estimate_dimension,
        smoke=args.smoke,
        default_dimension=1024,
    )
    print_estimate(
        "text",
        corpus_records=corpus_stats["documents"],
        corpus_rows=corpus_stats["chunks"],
        query_records=query_count_used,
        query_rows=query_count_used,
        dimension=dimension,
        dtype=args.dtype,
        input_bytes=corpus_file.stat().st_size + query_file.stat().st_size,
        overhead=args.estimate_overhead,
        assumptions=[
            f"chunk_size={args.chunk_size}, chunk_overlap={args.chunk_overlap}",
            f"dimension={dimension or 'unknown'}; pass --estimate-dimension for an explicit estimate",
            "artifact estimate is a rough vector-payload multiplier, not a storage guarantee",
        ],
        extra={"corpus_characters": corpus_stats["characters"]},
    )

    export = [
        sys.executable,
        str(WORKLOAD_DIR / "export_vectors.py"),
        "export",
        "--corpus-file",
        str(corpus_file),
        "--query-file",
        str(query_file),
        "--output-dir",
        str(run_dir / "artifact"),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--dtype",
        args.dtype,
        "--chunk-size",
        str(args.chunk_size),
        "--chunk-overlap",
        str(args.chunk_overlap),
        "--rows-per-shard",
        str(args.rows_per_shard),
        "--initial-corpus-ratio",
        str(args.initial_corpus_ratio),
        "--searches-per-insert",
        str(args.searches_per_insert),
        "--insert-event-size",
        str(args.insert_event_size),
        "--seed",
        str(args.seed),
        "--normalize" if args.normalize else "--no-normalize",
    ]
    if args.smoke:
        export.append("--smoke")
    add_option(export, "--model", args.model)
    add_option(export, "--revision", args.revision)
    add_option(export, "--max-corpus", max_corpus)
    add_option(export, "--max-queries", max_queries)
    run_command(export)
    run_command(
        [
            sys.executable,
            str(WORKLOAD_DIR / "export_vectors.py"),
            "verify",
            "--artifact-dir",
            str(run_dir / "artifact"),
        ]
    )
    print("Recorded dataset:")
    print_artifact_summary(run_dir / "artifact")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    record_parser = subparsers.add_parser("record", help="prepare and record a text workload")
    record_parser.add_argument("--output-dir", type=Path, required=True)
    record_parser.add_argument("--dataset", choices=("wikipedia-nq",), default="wikipedia-nq")
    record_parser.add_argument("--record-count", type=int)
    record_parser.add_argument("--query-count", type=int)
    record_parser.add_argument("--corpus-file", type=Path)
    record_parser.add_argument("--query-file", type=Path)
    record_parser.add_argument("--model")
    record_parser.add_argument("--revision")
    record_parser.add_argument("--device", default="cuda:0")
    record_parser.add_argument("--batch-size", type=int, default=32)
    record_parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    record_parser.add_argument("--smoke", action="store_true")
    record_parser.add_argument("--chunk-size", type=int, default=512)
    record_parser.add_argument("--chunk-overlap", type=int, default=0)
    record_parser.add_argument("--rows-per-shard", type=int, default=100_000)
    record_parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    record_parser.add_argument("--seed", type=int, default=42)
    add_schedule_arguments(record_parser)
    add_estimate_arguments(record_parser)
    record_parser.set_defaults(func=record)

    replay_parser = subparsers.add_parser("replay", help="replay a recorded text workload")
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
