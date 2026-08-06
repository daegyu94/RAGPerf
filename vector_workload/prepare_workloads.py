#!/usr/bin/env python
"""Prepare canonical corpus/query inputs for portable RAGPerf workloads."""

from __future__ import annotations

import argparse
import json
import shutil
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


def take_rows(rows: Iterable[dict[str, Any]], count: int) -> Iterable[dict[str, Any]]:
    iterator = iter(rows)
    try:
        for _ in range(count):
            try:
                yield next(iterator)
            except StopIteration:
                return
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


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
        take_rows(wikipedia_rows(wikipedia), args.corpus_count),
    )
    query_count = write_jsonl(
        output_dir / "queries.jsonl",
        take_rows(natural_question_rows(natural_questions), args.query_count),
    )
    result = {
        "workload": "wikipedia-natural-questions",
        "output_dir": str(output_dir),
        "corpus_documents": corpus_count,
        "queries": query_count,
    }
    print(json.dumps(result, indent=2))
    return result


def pdf_text_rows(
    pdf_paths: Iterable[Path],
    reader_factory: Callable[[Path], Any] | None = None,
) -> Iterable[dict[str, Any]]:
    if reader_factory is None:
        from pypdf import PdfReader

        reader_factory = PdfReader
    for path in pdf_paths:
        reader = reader_factory(path)
        page_texts = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(part.strip() for part in page_texts if part.strip())
        if not text:
            continue
        yield {
            "id": path.stem,
            "text": text,
            "metadata": {
                "dataset": "common-pile/arxiv_papers",
                "source_file": path.name,
                "pages": len(reader.pages),
                "modality": "text",
            },
        }


def validate_query_file(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError(f"{path}:{line_number}: query requires string 'text'")
            count += 1
    if not count:
        raise ValueError(f"{path}: no queries found")
    return count


def prepare_arxiv_pdf_text(
    args: argparse.Namespace,
    reader_factory: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    if args.max_pdfs is not None and args.max_pdfs <= 0:
        raise ValueError("max-pdfs must be positive")
    pdf_paths = sorted(args.pdf_dir.rglob("*.pdf"))
    if args.max_pdfs is not None:
        pdf_paths = pdf_paths[: args.max_pdfs]
    if not pdf_paths:
        raise ValueError(f"no PDF files found under {args.pdf_dir}")
    query_count = validate_query_file(args.query_file)

    output_dir = args.output_dir.resolve()
    ensure_empty_output_dir(output_dir)
    corpus_count = write_jsonl(
        output_dir / "corpus.jsonl", pdf_text_rows(pdf_paths, reader_factory)
    )
    shutil.copyfile(args.query_file, output_dir / "queries.jsonl")
    result = {
        "workload": "arxiv-pdf-text",
        "output_dir": str(output_dir),
        "pdf_documents": corpus_count,
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
    arxiv_text_parser = subparsers.add_parser(
        "arxiv-pdf-text", help="extract text from local arXiv PDF files"
    )
    arxiv_text_parser.add_argument("--pdf-dir", type=Path, required=True)
    arxiv_text_parser.add_argument("--query-file", type=Path, required=True)
    arxiv_text_parser.add_argument("--output-dir", type=Path, required=True)
    arxiv_text_parser.add_argument("--max-pdfs", type=int)
    arxiv_text_parser.set_defaults(func=prepare_arxiv_pdf_text)
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
