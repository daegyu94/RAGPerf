#!/usr/bin/env python
"""Record or replay a count-sized Audio ASR plus text embedding workload."""

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
    directory_bytes,
    ensure_empty_run_dir,
    infer_dimension,
    jsonl_record_count,
    jsonl_text_stats,
    print_artifact_summary,
    print_estimate,
    replay_workload,
    run_command,
)


def record(args: argparse.Namespace) -> None:
    if args.record_count <= 0:
        raise ValueError("record-count must be positive")
    if args.query_count is not None and args.query_count <= 0:
        raise ValueError("query-count must be positive")
    run_dir = ensure_empty_run_dir(args.output_dir)
    audio_dir = args.audio_dir.resolve()
    query_file = args.query_file.resolve()
    input_dir = run_dir / "input"
    prepare = [
        sys.executable,
        str(WORKLOAD_DIR / "prepare_workloads.py"),
        "audio-asr",
        "--audio-dir",
        str(audio_dir),
        "--query-file",
        str(query_file),
        "--output-dir",
        str(input_dir),
        "--model",
        args.asr_model,
        "--device",
        args.asr_device,
        "--dtype",
        args.asr_dtype,
        "--batch-size",
        str(args.asr_batch_size),
        "--max-audio-files",
        str(args.record_count),
        "--chunk-length-seconds",
        str(args.chunk_length_seconds),
    ]
    add_option(prepare, "--revision", args.asr_revision)
    add_option(prepare, "--language", args.language)
    add_option(prepare, "--dataset-name", args.dataset_name)
    if args.audio_extensions:
        prepare.extend(("--audio-extensions", *args.audio_extensions))
    run_command(prepare)

    corpus_file = input_dir / "corpus.jsonl"
    prepared_query_file = input_dir / "queries.jsonl"
    corpus_stats = jsonl_text_stats(
        corpus_file,
        args.chunk_size,
        args.chunk_overlap,
    )
    query_count = jsonl_record_count(prepared_query_file, args.query_count)
    dimension = infer_dimension(
        args.embedding_model,
        args.estimate_dimension,
        default_dimension=1024,
    )
    print_estimate(
        "audio-asr",
        corpus_records=corpus_stats["documents"],
        corpus_rows=corpus_stats["chunks"],
        query_records=query_count,
        query_rows=query_count,
        dimension=dimension,
        dtype=args.dtype,
        input_bytes=directory_bytes(audio_dir)
        + corpus_file.stat().st_size
        + prepared_query_file.stat().st_size,
        overhead=args.estimate_overhead,
        assumptions=[
            f"chunk_size={args.chunk_size}, chunk_overlap={args.chunk_overlap}",
            "ASR window length affects processing time, not directly the final vector row count",
            "raw audio is retained outside the artifact estimate",
        ],
        extra={
            "audio_files": corpus_stats["documents"],
            "audio_input_bytes": directory_bytes(audio_dir),
        },
    )

    export = [
        sys.executable,
        str(WORKLOAD_DIR / "export_vectors.py"),
        "export",
        "--corpus-file",
        str(corpus_file),
        "--query-file",
        str(prepared_query_file),
        "--output-dir",
        str(run_dir / "artifact"),
        "--device",
        args.embedding_device,
        "--batch-size",
        str(args.embedding_batch_size),
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
    ]
    add_option(export, "--model", args.embedding_model)
    add_option(export, "--revision", args.embedding_revision)
    add_option(export, "--max-queries", args.query_count)
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

    record_parser = subparsers.add_parser("record", help="transcribe and record an audio workload")
    record_parser.add_argument("--audio-dir", type=Path, required=True)
    record_parser.add_argument("--query-file", type=Path, required=True)
    record_parser.add_argument("--output-dir", type=Path, required=True)
    record_parser.add_argument("--record-count", type=int, required=True)
    record_parser.add_argument("--query-count", type=int)
    record_parser.add_argument("--asr-model", default="openai/whisper-small")
    record_parser.add_argument("--asr-revision")
    record_parser.add_argument("--asr-device", default="cuda:0")
    record_parser.add_argument("--asr-dtype", choices=("auto", "float16", "float32"), default="auto")
    record_parser.add_argument("--asr-batch-size", type=int, default=8)
    record_parser.add_argument("--language")
    record_parser.add_argument("--chunk-length-seconds", type=float, default=30)
    record_parser.add_argument("--audio-extensions", nargs="+")
    record_parser.add_argument("--dataset-name", default="local-audio-asr")
    record_parser.add_argument("--embedding-model")
    record_parser.add_argument("--embedding-revision")
    record_parser.add_argument("--embedding-device", default="cuda:0")
    record_parser.add_argument("--embedding-batch-size", type=int, default=32)
    record_parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    record_parser.add_argument("--chunk-size", type=int, default=512)
    record_parser.add_argument("--chunk-overlap", type=int, default=0)
    record_parser.add_argument("--rows-per-shard", type=int, default=100_000)
    record_parser.add_argument("--seed", type=int, default=42)
    add_schedule_arguments(record_parser)
    add_estimate_arguments(record_parser)
    record_parser.set_defaults(func=record)

    replay_parser = subparsers.add_parser("replay", help="replay a recorded audio workload")
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
