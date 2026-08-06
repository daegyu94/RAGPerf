"""Shared helpers for workload-specific record/replay scripts."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


WORKLOAD_DIR = Path(__file__).resolve().parents[1]


def run_command(command: list[str]) -> None:
    print(f"$ {shlex.join(command)}", flush=True)
    subprocess.run(command, check=True)


def add_option(command: list[str], name: str, value: object | None) -> None:
    if value is not None:
        command.extend((name, str(value)))


def add_schedule_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--initial-corpus-ratio", type=float, default=1.0)
    parser.add_argument("--searches-per-insert", type=int, default=1)
    parser.add_argument("--insert-event-size", type=int, default=1)


def add_estimate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--estimate-dimension",
        type=int,
        help="override the dimension used for the pre-record size estimate",
    )
    parser.add_argument(
        "--estimate-overhead",
        type=float,
        default=1.25,
        help="multiplier for the rough Parquet artifact estimate",
    )


def add_replay_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--uri", default="http://127.0.0.1:19530")
    parser.add_argument("--result-file", type=Path)
    parser.add_argument("--metric", choices=("COSINE", "L2", "IP"))
    parser.add_argument("--warmup-queries", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--search-list", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--token-top-k", type=int)
    parser.add_argument("--storage-path-note")
    parser.add_argument("--respect-delay", action="store_true")


def ensure_empty_run_dir(path: Path) -> Path:
    run_dir = path.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {run_dir}")
    return run_dir


def dtype_bytes(dtype: str) -> int:
    if dtype == "float16":
        return 2
    if dtype == "float32":
        return 4
    raise ValueError(f"unsupported estimate dtype: {dtype}")


def format_bytes(value: int | float | None) -> str:
    if value is None:
        return "unknown"
    size = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if abs(size) < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TiB"


def directory_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def count_chunks(text: str, chunk_size: int, chunk_overlap: int) -> int:
    if chunk_size <= 0:
        return 1 if text.strip() else 0
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size")
    chunks = 0
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if text[start:end].strip():
            chunks += 1
        if end == len(text):
            break
        start = end - chunk_overlap
    return chunks


def jsonl_text_stats(
    path: Path,
    chunk_size: int,
    chunk_overlap: int,
    limit: int | None = None,
) -> dict[str, int]:
    documents = 0
    chunks = 0
    characters = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            if limit is not None and documents >= limit:
                break
            row = json.loads(line)
            text = row.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            documents += 1
            characters += len(text)
            chunks += count_chunks(text, chunk_size, chunk_overlap)
    if documents == 0:
        raise ValueError(f"no text records found in {path}")
    return {"documents": documents, "chunks": chunks, "characters": characters}


def jsonl_record_count(path: Path, limit: int | None = None) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                count += 1
                if limit is not None and count >= limit:
                    break
    if count == 0:
        raise ValueError(f"no records found in {path}")
    return count


def infer_dimension(
    model: str | None,
    explicit_dimension: int | None,
    smoke: bool = False,
    default_dimension: int | None = None,
) -> int | None:
    if explicit_dimension is not None:
        if explicit_dimension <= 0:
            raise ValueError("estimate-dimension must be positive")
        return explicit_dimension
    if smoke:
        return 384
    model_name = (model or "").lower()
    if "all-minilm" in model_name:
        return 384
    if "bge-m3" in model_name:
        return 1024
    return default_dimension


def print_estimate(
    workload: str,
    *,
    corpus_records: int,
    corpus_rows: int,
    query_records: int,
    query_rows: int,
    dimension: int | None,
    dtype: str,
    input_bytes: int,
    overhead: float,
    assumptions: list[str],
    extra: dict[str, Any] | None = None,
) -> None:
    if overhead < 1:
        raise ValueError("estimate-overhead must be at least 1")
    total_rows = corpus_rows + query_rows
    raw_bytes = total_rows * dimension * dtype_bytes(dtype) if dimension else None
    artifact_bytes = raw_bytes * overhead if raw_bytes is not None else None
    estimate: dict[str, Any] = {
        "workload": workload,
        "source_records": {"corpus": corpus_records, "queries": query_records},
        "estimated_vector_rows": {
            "corpus": corpus_rows,
            "queries": query_rows,
            "total": total_rows,
        },
        "estimated_input_bytes": format_bytes(input_bytes),
        "estimated_raw_vector_bytes": format_bytes(raw_bytes),
        "estimated_artifact_bytes": format_bytes(artifact_bytes),
        "assumptions": assumptions,
    }
    if extra:
        estimate["source_details"] = extra
    print("Estimated dataset:")
    print(json.dumps(estimate, indent=2, ensure_ascii=False))


def print_artifact_summary(artifact_dir: Path) -> None:
    manifest_path = artifact_dir / "workload-manifest.yaml"
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    artifacts = manifest.get("artifacts", [])
    rows_by_kind: dict[str, int] = {}
    artifact_bytes = 0
    for artifact in artifacts:
        kind = str(artifact.get("kind", "unknown"))
        rows_by_kind[kind] = rows_by_kind.get(kind, 0) + int(artifact.get("rows", 0))
        artifact_path = artifact_dir / str(artifact["path"])
        artifact_bytes += artifact_path.stat().st_size
    embedding = manifest.get("embedding", {})
    print(
        json.dumps(
            {
                "artifact_dir": str(artifact_dir.resolve()),
                "artifact_bytes": format_bytes(artifact_bytes),
                "rows_by_kind": rows_by_kind,
                "vector_layout": embedding.get("vector_layout"),
                "dimension": embedding.get("dimension"),
                "dtype": embedding.get("dtype"),
                "model": embedding.get("model"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def replay_workload(args: argparse.Namespace) -> None:
    artifact_dir = args.artifact_dir.resolve()
    print("Recorded dataset:")
    print_artifact_summary(artifact_dir)
    command = [
        sys.executable,
        str(WORKLOAD_DIR / "replay_workload.py"),
        "--artifact-dir",
        str(artifact_dir),
        "--collection",
        args.collection,
        "--uri",
        args.uri,
    ]
    add_option(command, "--result-file", args.result_file.resolve() if args.result_file else None)
    add_option(command, "--metric", args.metric)
    add_option(command, "--warmup-queries", args.warmup_queries)
    add_option(command, "--concurrency", args.concurrency)
    add_option(command, "--search-list", args.search_list)
    add_option(command, "--top-k", args.top_k)
    add_option(command, "--token-top-k", args.token_top_k)
    add_option(command, "--storage-path-note", args.storage_path_note)
    if args.respect_delay:
        command.append("--respect-delay")
    run_command(command)
