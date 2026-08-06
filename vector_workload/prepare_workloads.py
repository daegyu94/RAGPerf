#!/usr/bin/env python
"""Prepare canonical corpus/query inputs for portable RAGPerf workloads."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable


def ensure_empty_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"output directory must be empty: {path}")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    if count == 0:
        raise ValueError(f"no records written to {path}")
    return count


def wikipedia_rows(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for index, row in enumerate(rows):
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        source_id = str(row.get("id", f"wikipedia-{index:08d}"))
        yield {
            "id": source_id,
            "text": text,
            "metadata": {
                "dataset": "wikimedia/wikipedia",
                "title": row.get("title"),
                "url": row.get("url"),
            },
        }


def natural_question_rows(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for index, row in enumerate(rows):
        query = row.get("query")
        if not isinstance(query, str) or not query.strip():
            continue
        yield {
            "id": str(row.get("id", f"natural-question-{index:08d}")),
            "text": query,
            "metadata": {
                "dataset": "sentence-transformers/natural-questions",
                "answer": row.get("answer"),
            },
            "delay_ms": 0,
        }


def prepare_wikipedia_nq(
    args: argparse.Namespace,
    load_dataset_fn: Callable[..., Iterable[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if args.corpus_count <= 0 or args.query_count <= 0:
        raise ValueError("corpus-count and query-count must be positive")
    if load_dataset_fn is None:
        from datasets import load_dataset

        load_dataset_fn = load_dataset

    output_dir = args.output_dir.resolve()
    ensure_empty_output_dir(output_dir)
    wikipedia = load_dataset_fn(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )
    natural_questions = load_dataset_fn(
        "sentence-transformers/natural-questions",
        split="train",
        streaming=True,
    )
    corpus_count = write_jsonl(
        output_dir / "corpus.jsonl",
        itertools.islice(wikipedia_rows(wikipedia), args.corpus_count),
    )
    query_count = write_jsonl(
        output_dir / "queries.jsonl",
        itertools.islice(natural_question_rows(natural_questions), args.query_count),
    )
    result = {
        "workload": "wikipedia-natural-questions",
        "output_dir": str(output_dir),
        "corpus_documents": corpus_count,
        "queries": query_count,
    }
    print(json.dumps(result, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    wikipedia_parser = subparsers.add_parser(
        "wikipedia-nq", help="prepare Wikipedia corpus and Natural Questions queries"
    )
    wikipedia_parser.add_argument("--output-dir", type=Path, required=True)
    wikipedia_parser.add_argument("--corpus-count", type=int, required=True)
    wikipedia_parser.add_argument("--query-count", type=int, required=True)
    wikipedia_parser.set_defaults(func=prepare_wikipedia_nq)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
