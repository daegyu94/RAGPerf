#!/usr/bin/env python
"""Record a bounded-memory synthetic vector workload artifact."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from artifact_utils import SCHEMA_VERSION, sha256_file, verify_artifact


def normalized_random_vectors(
    rng: np.random.Generator, rows: int, dimension: int, dtype: str
) -> np.ndarray:
    vectors = rng.standard_normal((rows, dimension), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.asarray(vectors, dtype=np.dtype(dtype), order="C")


def vector_array(vectors: np.ndarray) -> pa.FixedSizeListArray:
    value_type = pa.float16() if vectors.dtype == np.float16 else pa.float32()
    values = pa.array(vectors.reshape(-1), type=value_type)
    return pa.FixedSizeListArray.from_arrays(values, vectors.shape[1])


def write_table(path: Path, table: pa.Table, kind: str) -> dict[str, Any]:
    pq.write_table(table, path, compression="zstd", use_dictionary=["id"])
    return {
        "path": path.name,
        "kind": kind,
        "rows": table.num_rows,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def ensure_empty_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"output directory must be empty: {path}")


def record(args: argparse.Namespace) -> None:
    if args.corpus_count <= 0 or args.query_count <= 0:
        raise ValueError("corpus-count and query-count must be positive")
    if args.dimension <= 0 or args.rows_per_shard <= 0:
        raise ValueError("dimension and rows-per-shard must be positive")
    if args.query_noise < 0 or args.query_delay_ms < 0:
        raise ValueError("query-noise and query-delay-ms must be non-negative")

    output_dir = args.output_dir.resolve()
    ensure_empty_output_dir(output_dir)
    started = time.perf_counter()
    vector_rng = np.random.default_rng(args.seed)
    selection_rng = np.random.default_rng(args.seed + 1)
    query_rng = np.random.default_rng(args.seed + 2)

    anchor_indices = selection_rng.choice(
        args.corpus_count,
        size=args.query_count,
        replace=args.query_count > args.corpus_count,
    )
    anchors_by_row: dict[int, list[int]] = {}
    for query_index, corpus_index in enumerate(anchor_indices.tolist()):
        anchors_by_row.setdefault(corpus_index, []).append(query_index)
    query_anchors = np.empty((args.query_count, args.dimension), dtype=np.float32)

    artifacts: list[dict[str, Any]] = []
    corpus_seconds_started = time.perf_counter()
    for shard_index, start in enumerate(
        range(0, args.corpus_count, args.rows_per_shard)
    ):
        rows = min(args.rows_per_shard, args.corpus_count - start)
        vectors = normalized_random_vectors(vector_rng, rows, args.dimension, args.dtype)
        for corpus_index in range(start, start + rows):
            for query_index in anchors_by_row.get(corpus_index, ()):  # usually zero or one
                query_anchors[query_index] = vectors[corpus_index - start]
        ids = [f"corpus-{row:09d}" for row in range(start, start + rows)]
        table = pa.table(
            {
                "id": pa.array(ids, type=pa.string()),
                "text": pa.array(
                    [f"Synthetic vector record {row}" for row in range(start, start + rows)],
                    type=pa.string(),
                ),
                "metadata_json": pa.array(
                    [
                        json.dumps(
                            {"generator": "normalized_gaussian", "row": row},
                            sort_keys=True,
                        )
                        for row in range(start, start + rows)
                    ],
                    type=pa.string(),
                ),
                "vector": vector_array(vectors),
            }
        )
        path = output_dir / f"corpus-{shard_index:05d}.parquet"
        artifacts.append(write_table(path, table, "corpus"))
    corpus_seconds = time.perf_counter() - corpus_seconds_started

    query_seconds_started = time.perf_counter()
    query_vectors = query_anchors
    if args.query_noise:
        query_vectors = query_vectors + query_rng.normal(
            0.0, args.query_noise, query_vectors.shape
        ).astype(np.float32)
    query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True)
    query_vectors = np.asarray(query_vectors, dtype=np.dtype(args.dtype), order="C")

    for shard_index, start in enumerate(
        range(0, args.query_count, args.rows_per_shard)
    ):
        rows = min(args.rows_per_shard, args.query_count - start)
        end = start + rows
        expected_ids = [f"corpus-{row:09d}" for row in anchor_indices[start:end]]
        table = pa.table(
            {
                "id": pa.array(
                    [f"query-{row:09d}" for row in range(start, end)], type=pa.string()
                ),
                "text": pa.array(
                    [f"Synthetic nearest-neighbor query {row}" for row in range(start, end)],
                    type=pa.string(),
                ),
                "metadata_json": pa.array(
                    [
                        json.dumps({"expected_id": expected_id}, sort_keys=True)
                        for expected_id in expected_ids
                    ],
                    type=pa.string(),
                ),
                "vector": vector_array(query_vectors[start:end]),
                "delay_ms": pa.array([args.query_delay_ms] * rows, type=pa.float64()),
            }
        )
        path = output_dir / f"queries-{shard_index:05d}.parquet"
        artifacts.append(write_table(path, table, "queries"))
    query_seconds = time.perf_counter() - query_seconds_started

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "producer": "vector_workload/record_synthetic.py",
        "inputs": {
            "corpus": {
                "generator": "normalized_gaussian",
                "documents": args.corpus_count,
                "chunks": args.corpus_count,
            },
            "queries": {
                "generator": "corpus_anchor_plus_gaussian_noise",
                "count": args.query_count,
                "schedule": "parquet_row_order_with_delay_ms",
            },
        },
        "chunking": {"method": "none", "chunk_size": 0, "chunk_overlap": 0},
        "embedding": {
            "model": "synthetic-normalized-gaussian",
            "revision": "1",
            "device": "cpu",
            "dimension": args.dimension,
            "dtype": args.dtype,
            "normalized": True,
            "batch_size": args.rows_per_shard,
            "seed": args.seed,
            "corpus_seconds": corpus_seconds,
            "query_seconds": query_seconds,
            "corpus_vectors_per_second": args.corpus_count / corpus_seconds,
            "query_vectors_per_second": args.query_count / query_seconds,
            "gpu": None,
        },
        "workload": {
            "operation": "search",
            "query_order": "queries parquet row order",
            "timing": "delay_ms before each query",
            "query_noise_standard_deviation": args.query_noise,
            "expected_neighbor_field": "metadata_json.expected_id",
        },
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "workload-manifest.yaml"
    with manifest_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False, allow_unicode=True)

    checksums_path = output_dir / "SHA256SUMS"
    with checksums_path.open("w", encoding="utf-8") as stream:
        for artifact in artifacts:
            stream.write(f"{artifact['sha256']}  {artifact['path']}\n")
        stream.write(f"{sha256_file(manifest_path)}  {manifest_path.name}\n")

    verify_artifact(output_dir)
    total_seconds = time.perf_counter() - started
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "corpus_vectors": args.corpus_count,
                "query_vectors": args.query_count,
                "dimension": args.dimension,
                "dtype": args.dtype,
                "parquet_shards": len(artifacts),
                "artifact_bytes": sum(path.stat().st_size for path in output_dir.iterdir()),
                "record_seconds": total_seconds,
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpus-count", type=int, default=100_000)
    parser.add_argument("--query-count", type=int, default=2_000)
    parser.add_argument("--dimension", type=int, default=384)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--rows-per-shard", type=int, default=25_000)
    parser.add_argument("--query-noise", type=float, default=0.01)
    parser.add_argument("--query-delay-ms", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    try:
        record(build_parser().parse_args())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
