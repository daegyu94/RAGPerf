#!/usr/bin/env python
"""Create and verify portable corpus/query vector workload artifacts."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from sentence_transformers import SentenceTransformer

from artifact_utils import SCHEMA_VERSION, sha256_file, verify_artifact

def read_jsonl(path: Path, kind: str, limit: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: each line must be a JSON object")
            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{line_number}: 'text' must be a non-empty string")
            record_id = str(record.get("id", f"{kind}-{len(records):08d}"))
            if record_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate id {record_id!r}")
            seen_ids.add(record_id)
            normalized = {"id": record_id, "text": text}
            metadata = record.get("metadata", {})
            if metadata is not None and not isinstance(metadata, dict):
                raise ValueError(f"{path}:{line_number}: 'metadata' must be an object")
            normalized["metadata"] = metadata or {}
            if kind == "query":
                delay_ms = record.get("delay_ms", 0)
                if not isinstance(delay_ms, (int, float)) or delay_ms < 0:
                    raise ValueError(f"{path}:{line_number}: 'delay_ms' must be non-negative")
                normalized["delay_ms"] = float(delay_ms)
            records.append(normalized)
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"{path}: no records found")
    return records


def chunk_corpus(
    documents: Iterable[dict[str, Any]], chunk_size: int, chunk_overlap: int
) -> list[dict[str, Any]]:
    if chunk_size <= 0:
        return list(documents)
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size")

    chunks: list[dict[str, Any]] = []
    for document in documents:
        text = document["text"]
        start = 0
        chunk_index = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            if chunk_text.strip():
                metadata = dict(document["metadata"])
                metadata.update(
                    {
                        "source_id": document["id"],
                        "chunk_index": chunk_index,
                        "start_char": start,
                        "end_char": end,
                    }
                )
                chunks.append(
                    {
                        "id": f"{document['id']}#chunk-{chunk_index:05d}",
                        "text": chunk_text,
                        "metadata": metadata,
                    }
                )
            if end == len(text):
                break
            start = end - chunk_overlap
            chunk_index += 1
    if not chunks:
        raise ValueError("chunking produced no corpus records")
    return chunks


def load_encoder(model_name: str, revision: str | None, device: str) -> SentenceTransformer:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device!r} requested, but CUDA is not available")
    kwargs: dict[str, Any] = {"device": device}
    if revision is not None:
        kwargs["revision"] = revision
    return SentenceTransformer(model_name, **kwargs)


def encode_records(
    encoder: SentenceTransformer,
    records: list[dict[str, Any]],
    batch_size: int,
    normalize: bool,
    dtype: str,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    vectors = encoder.encode(
        [record["text"] for record in records],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=True,
    )
    elapsed = time.perf_counter() - started
    vectors = np.asarray(vectors, dtype=np.dtype(dtype), order="C")
    if vectors.ndim != 2 or vectors.shape[0] != len(records):
        raise RuntimeError(
            f"unexpected embedding shape {vectors.shape}; expected ({len(records)}, dimension)"
        )
    if not np.isfinite(vectors).all():
        raise RuntimeError("embedding output contains NaN or infinity")
    return vectors, elapsed


def vector_array(vectors: np.ndarray) -> pa.FixedSizeListArray:
    value_type = pa.float16() if vectors.dtype == np.float16 else pa.float32()
    values = pa.array(vectors.reshape(-1), type=value_type)
    return pa.FixedSizeListArray.from_arrays(values, vectors.shape[1])


def write_shards(
    output_dir: Path,
    prefix: str,
    records: list[dict[str, Any]],
    vectors: np.ndarray,
    rows_per_shard: int,
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    shard_count = math.ceil(len(records) / rows_per_shard)
    for shard_index in range(shard_count):
        start = shard_index * rows_per_shard
        end = min(start + rows_per_shard, len(records))
        shard_records = records[start:end]
        arrays: dict[str, pa.Array] = {
            "id": pa.array([record["id"] for record in shard_records], type=pa.string()),
            "text": pa.array([record["text"] for record in shard_records], type=pa.string()),
            "metadata_json": pa.array(
                [
                    json.dumps(record["metadata"], ensure_ascii=False, sort_keys=True)
                    for record in shard_records
                ],
                type=pa.string(),
            ),
            "vector": vector_array(vectors[start:end]),
        }
        if prefix == "queries":
            arrays["delay_ms"] = pa.array(
                [record["delay_ms"] for record in shard_records], type=pa.float64()
            )
        table = pa.table(arrays)
        path = output_dir / f"{prefix}-{shard_index:05d}.parquet"
        pq.write_table(table, path, compression="zstd", use_dictionary=["id"])
        artifacts.append(
            {
                "path": path.name,
                "kind": prefix,
                "rows": end - start,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return artifacts


def gpu_metadata(device: str) -> dict[str, Any] | None:
    if not device.startswith("cuda"):
        return None
    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "device": device,
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "torch_cuda_version": torch.version.cuda,
    }


def ensure_empty_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"output directory must be empty: {path}")


def export_artifact(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.rows_per_shard <= 0:
        raise ValueError("batch_size and rows_per_shard must be positive")
    output_dir = args.output_dir.resolve()
    ensure_empty_output_dir(output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    corpus_documents = read_jsonl(args.corpus_file, "corpus", args.max_corpus)
    corpus_records = chunk_corpus(corpus_documents, args.chunk_size, args.chunk_overlap)
    query_records = read_jsonl(args.query_file, "query", args.max_queries)

    encoder = load_encoder(args.model, args.revision, args.device)
    if hasattr(encoder, "get_embedding_dimension"):
        dimension = encoder.get_embedding_dimension()
    else:
        dimension = encoder.get_sentence_embedding_dimension()
    corpus_vectors, corpus_seconds = encode_records(
        encoder, corpus_records, args.batch_size, args.normalize, args.dtype
    )
    query_vectors, query_seconds = encode_records(
        encoder, query_records, args.batch_size, args.normalize, args.dtype
    )
    if corpus_vectors.shape[1] != dimension or query_vectors.shape[1] != dimension:
        raise RuntimeError("encoder dimension does not match generated vectors")

    artifacts = write_shards(
        output_dir, "corpus", corpus_records, corpus_vectors, args.rows_per_shard
    )
    artifacts.extend(
        write_shards(output_dir, "queries", query_records, query_vectors, args.rows_per_shard)
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "producer": "vector_workload/export_vectors.py",
        "inputs": {
            "corpus": {
                "file": args.corpus_file.name,
                "sha256": sha256_file(args.corpus_file),
                "documents": len(corpus_documents),
                "chunks": len(corpus_records),
            },
            "queries": {
                "file": args.query_file.name,
                "sha256": sha256_file(args.query_file),
                "count": len(query_records),
                "schedule": "parquet_row_order_with_delay_ms",
            },
        },
        "chunking": {
            "method": "character_window" if args.chunk_size > 0 else "none",
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
        },
        "embedding": {
            "model": args.model,
            "revision": args.revision or "default",
            "device": args.device,
            "dimension": dimension,
            "dtype": args.dtype,
            "normalized": args.normalize,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "corpus_seconds": corpus_seconds,
            "query_seconds": query_seconds,
            "corpus_vectors_per_second": len(corpus_records) / corpus_seconds,
            "query_vectors_per_second": len(query_records) / query_seconds,
            "gpu": gpu_metadata(args.device),
        },
        "workload": {
            "operation": "search",
            "query_order": "queries parquet row order",
            "timing": "delay_ms before each query",
        },
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "workload-manifest.yaml"
    with manifest_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False, allow_unicode=True)

    checksum_paths = [output_dir / artifact["path"] for artifact in artifacts]
    checksum_paths.append(manifest_path)
    checksums_path = output_dir / "SHA256SUMS"
    with checksums_path.open("w", encoding="utf-8") as stream:
        for path in checksum_paths:
            stream.write(f"{sha256_file(path)}  {path.name}\n")

    verify_artifact(output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "corpus_documents": len(corpus_documents),
                "corpus_vectors": len(corpus_records),
                "query_vectors": len(query_records),
                "dimension": dimension,
                "dtype": args.dtype,
                "device": args.device,
                "artifact_bytes": sum(path.stat().st_size for path in output_dir.iterdir()),
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="generate a vector workload artifact")
    export_parser.add_argument("--corpus-file", type=Path, required=True)
    export_parser.add_argument("--query-file", type=Path, required=True)
    export_parser.add_argument("--output-dir", type=Path, required=True)
    export_parser.add_argument(
        "--model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    export_parser.add_argument("--revision")
    export_parser.add_argument("--device", default="cuda:0")
    export_parser.add_argument("--batch-size", type=int, default=32)
    export_parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    export_parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    export_parser.add_argument("--chunk-size", type=int, default=512)
    export_parser.add_argument("--chunk-overlap", type=int, default=0)
    export_parser.add_argument("--rows-per-shard", type=int, default=100_000)
    export_parser.add_argument("--max-corpus", type=int)
    export_parser.add_argument("--max-queries", type=int)
    export_parser.add_argument("--seed", type=int, default=42)
    export_parser.set_defaults(func=export_artifact)

    verify_parser = subparsers.add_parser("verify", help="verify checksums and Parquet metadata")
    verify_parser.add_argument("--artifact-dir", type=Path, required=True)
    verify_parser.set_defaults(func=lambda args: verify_artifact(args.artifact_dir))
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
