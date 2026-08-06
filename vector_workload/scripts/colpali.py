#!/usr/bin/env python
"""Record or replay a count-sized ColPali PDF image workload."""

from __future__ import annotations

import argparse
import json
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
    print_artifact_summary,
    print_estimate,
    replay_workload,
    run_command,
)


def page_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                count += 1
    if count == 0:
        raise ValueError(f"no page records found in {path}")
    return count


def record(args: argparse.Namespace) -> None:
    if args.record_count <= 0:
        raise ValueError("record-count must be positive")
    if args.query_count is not None and args.query_count <= 0:
        raise ValueError("query-count must be positive")
    if args.estimated_vectors_per_page <= 0:
        raise ValueError("estimated-vectors-per-page must be positive")
    run_dir = ensure_empty_run_dir(args.output_dir)
    pdf_dir = args.pdf_dir.resolve()
    query_file = args.query_file.resolve()
    input_dir = run_dir / "input"
    prepare = [
        sys.executable,
        str(WORKLOAD_DIR / "prepare_workloads.py"),
        "arxiv-pdf-image",
        "--pdf-dir",
        str(pdf_dir),
        "--query-file",
        str(query_file),
        "--output-dir",
        str(input_dir),
        "--max-pdfs",
        str(args.record_count),
    ]
    run_command(prepare)

    corpus_file = input_dir / "corpus.jsonl"
    prepared_query_file = input_dir / "queries.jsonl"
    pages = page_count(corpus_file)
    query_count = jsonl_record_count(prepared_query_file, args.query_count)
    dimension = infer_dimension(
        args.model,
        args.estimate_dimension,
        default_dimension=128,
    )
    page_image_bytes = directory_bytes(input_dir / "pages")
    pdf_bytes = directory_bytes(pdf_dir)
    estimated_corpus_rows = pages * args.estimated_vectors_per_page
    print_estimate(
        "colpali",
        corpus_records=pages,
        corpus_rows=estimated_corpus_rows,
        query_records=query_count,
        query_rows=query_count * args.estimated_query_vectors,
        dimension=dimension,
        dtype=args.dtype,
        input_bytes=pdf_bytes + page_image_bytes + corpus_file.stat().st_size + prepared_query_file.stat().st_size,
        overhead=args.estimate_overhead,
        assumptions=[
            f"estimated_vectors_per_page={args.estimated_vectors_per_page}",
            f"estimated_query_vectors={args.estimated_query_vectors}",
            "ColPali token row counts are model/revision dependent; manifest values after export are authoritative",
            "PDF and rendered PNG input bytes are separate from the final artifact",
        ],
        extra={
            "pdf_files_requested": args.record_count,
            "pdf_pages": pages,
            "pdf_input_bytes": pdf_bytes,
            "rendered_page_bytes": page_image_bytes,
        },
    )

    export = [
        sys.executable,
        str(WORKLOAD_DIR / "export_colpali.py"),
        "--corpus-file",
        str(corpus_file),
        "--query-file",
        str(prepared_query_file),
        "--output-dir",
        str(run_dir / "artifact"),
        "--model",
        args.model,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--dtype",
        args.dtype,
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
    add_option(export, "--revision", args.revision)
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

    record_parser = subparsers.add_parser("record", help="render and record a ColPali workload")
    record_parser.add_argument("--pdf-dir", type=Path, required=True)
    record_parser.add_argument("--query-file", type=Path, required=True)
    record_parser.add_argument("--output-dir", type=Path, required=True)
    record_parser.add_argument("--record-count", type=int, required=True, help="maximum PDF files")
    record_parser.add_argument("--query-count", type=int)
    record_parser.add_argument("--model", default="vidore/colpali-v1.2")
    record_parser.add_argument("--revision")
    record_parser.add_argument("--device", default="cuda:0")
    record_parser.add_argument("--batch-size", type=int, default=1)
    record_parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    record_parser.add_argument("--rows-per-shard", type=int, default=100_000)
    record_parser.add_argument("--estimated-vectors-per-page", type=int, default=1_000)
    record_parser.add_argument("--estimated-query-vectors", type=int, default=32)
    record_parser.add_argument("--seed", type=int, default=42)
    add_schedule_arguments(record_parser)
    add_estimate_arguments(record_parser)
    record_parser.set_defaults(func=record)

    replay_parser = subparsers.add_parser("replay", help="replay a recorded ColPali workload")
    add_replay_arguments(replay_parser)
    replay_parser.set_defaults(func=replay_workload)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except (OSError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
