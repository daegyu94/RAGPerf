#!/usr/bin/env python
"""Create a portable ColPali multi-vector workload artifact."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

try:
    from artifact_utils import SCHEMA_VERSION, sha256_file, verify_artifact
    from export_vectors import (
        ensure_empty_output_dir,
        gpu_metadata,
        read_jsonl,
        write_schedule,
        write_shards,
    )
except ModuleNotFoundError:  # Supports `python -m vector_workload.export_colpali`.
    from vector_workload.artifact_utils import SCHEMA_VERSION, sha256_file, verify_artifact
    from vector_workload.export_vectors import (
        ensure_empty_output_dir,
        gpu_metadata,
        read_jsonl,
        write_schedule,
        write_shards,
    )


DEFAULT_COLPALI_MODEL = "vidore/colpali-v1.2"


def read_image_jsonl(path: Path, limit: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            record_id = str(row.get("id", f"page-{len(records):08d}"))
            image_path = Path(str(row.get("image_path", ""))).expanduser()
            if record_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate id {record_id!r}")
            if not image_path.is_file():
                raise FileNotFoundError(f"{path}:{line_number}: missing image {image_path}")
            metadata = row.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError(f"{path}:{line_number}: metadata must be an object")
            seen_ids.add(record_id)
            records.append(
                {"id": record_id, "image_path": image_path.resolve(), "metadata": metadata}
            )
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"{path}: no image records found")
    return records


class ColPaliAdapter:
    def __init__(self, model_name: str, revision: str | None, device: str):
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {device!r} requested, but CUDA is not available")
        from colpali_engine.models import ColPali
        from colpali_engine.models.paligemma.colpali.processing_colpali import (
            ColPaliProcessor,
        )

        kwargs: dict[str, Any] = {"device_map": device}
        if revision:
            kwargs["revision"] = revision
        self.model = ColPali.from_pretrained(model_name, **kwargs).eval()
        self.processor = ColPaliProcessor.from_pretrained(model_name, revision=revision)
        self.device = device

    def _encode(self, batches: list[Any], process: Any) -> list[np.ndarray]:
        encoded: list[np.ndarray] = []
        for batch in batches:
            inputs = process(batch)
            inputs = {name: value.to(self.model.device) for name, value in inputs.items()}
            with torch.no_grad():
                output = self.model(**inputs)
            if hasattr(output, "last_hidden_state"):
                output = output.last_hidden_state
            encoded.extend(item.float().cpu().numpy() for item in torch.unbind(output))
        return encoded

    def encode_images(self, paths: list[Path], batch_size: int) -> list[np.ndarray]:
        from PIL import Image

        batches: list[list[Any]] = []
        for start in range(0, len(paths), batch_size):
            images = [Image.open(path).convert("RGB") for path in paths[start : start + batch_size]]
            batches.append(images)
        try:
            return self._encode(batches, self.processor.process_images)
        finally:
            for batch in batches:
                for image in batch:
                    image.close()

    def encode_queries(self, texts: list[str], batch_size: int) -> list[np.ndarray]:
        batches = [texts[start : start + batch_size] for start in range(0, len(texts), batch_size)]
        return self._encode(batches, self.processor.process_queries)


def normalize_embeddings(vectors: list[np.ndarray], dtype: str) -> list[np.ndarray]:
    normalized: list[np.ndarray] = []
    for vector in vectors:
        array = np.asarray(vector, dtype=np.dtype(dtype))
        if array.ndim != 2 or not len(array):
            raise ValueError(f"expected non-empty rank-2 multi-vector, got {array.shape}")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        array = array[norms[:, 0] > 0]
        norms = norms[norms[:, 0] > 0]
        if not len(array):
            raise ValueError("multi-vector contains only padding vectors")
        array = np.divide(array, norms, out=np.zeros_like(array), where=norms != 0)
        if not np.isfinite(array).all():
            raise ValueError("embedding contains NaN or infinity")
        normalized.append(array)
    return normalized


def flatten_multivectors(
    records: list[dict[str, Any]], vectors: list[np.ndarray], group_kind: str
) -> tuple[list[dict[str, Any]], np.ndarray]:
    flat_records: list[dict[str, Any]] = []
    flat_vectors: list[np.ndarray] = []
    group_key = "document_id" if group_kind == "document" else "query_id"
    for record, matrix in zip(records, vectors, strict=True):
        for sequence_id, vector in enumerate(matrix):
            metadata = dict(record.get("metadata", {}))
            metadata.update({group_key: record["id"], "sequence_id": sequence_id})
            flat_records.append(
                {
                    "id": f"{record['id']}#vector-{sequence_id:05d}",
                    "text": record.get("text", record["id"]),
                    "metadata": metadata,
                    "delay_ms": record.get("delay_ms", 0),
                }
            )
            flat_vectors.append(vector)
    return flat_records, np.asarray(flat_vectors)


def write_multivector_schedule(
    output_dir: Path,
    query_count: int,
    insert_vectors: list[np.ndarray],
    searches_per_insert: int,
    insert_event_size: int,
) -> dict[str, Any]:
    if searches_per_insert <= 0 or insert_event_size <= 0:
        raise ValueError("searches-per-insert and insert-event-size must be positive")
    token_counts = [len(vectors) for vectors in insert_vectors]
    grouped_counts: list[int] = []
    for start in range(0, len(token_counts), insert_event_size):
        grouped_counts.append(sum(token_counts[start : start + insert_event_size]))
    # Reuse the common writer when every document has one vector.
    if not grouped_counts or len(set(grouped_counts)) == 1:
        size = grouped_counts[0] if grouped_counts else 1
        return write_schedule(
            output_dir,
            query_count,
            sum(grouped_counts),
            searches_per_insert,
            size,
        )

    import pyarrow as pa
    import pyarrow.parquet as pq

    operations: list[str] = []
    counts: list[int] = []
    insert_index = 0
    for query_index in range(query_count):
        operations.append("search")
        counts.append(1)
        if (query_index + 1) % searches_per_insert == 0 and insert_index < len(grouped_counts):
            operations.append("insert")
            counts.append(grouped_counts[insert_index])
            insert_index += 1
    while insert_index < len(grouped_counts):
        operations.append("insert")
        counts.append(grouped_counts[insert_index])
        insert_index += 1
    path = output_dir / "schedule-00000.parquet"
    pq.write_table(
        pa.table(
            {
                "sequence": pa.array(range(len(operations)), type=pa.int64()),
                "operation": pa.array(operations, type=pa.string()),
                "count": pa.array(counts, type=pa.int32()),
            }
        ),
        path,
        compression="zstd",
        use_dictionary=["operation"],
    )
    return {
        "path": path.name,
        "kind": "schedule",
        "rows": len(operations),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def export_artifact(args: argparse.Namespace, encoder: Any | None = None) -> None:
    if args.batch_size <= 0 or args.rows_per_shard <= 0:
        raise ValueError("batch-size and rows-per-shard must be positive")
    if not 0 < args.initial_corpus_ratio <= 1:
        raise ValueError("initial-corpus-ratio must satisfy 0 < ratio <= 1")
    if args.max_corpus is not None and args.max_corpus <= 0:
        raise ValueError("max-corpus must be positive")
    if args.max_queries is not None and args.max_queries <= 0:
        raise ValueError("max-queries must be positive")
    output_dir = args.output_dir.resolve()
    ensure_empty_output_dir(output_dir)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    corpus = read_image_jsonl(args.corpus_file, args.max_corpus)
    queries = read_jsonl(args.query_file, "query", args.max_queries)
    if encoder is None:
        encoder = ColPaliAdapter(args.model, args.revision, args.device)
    corpus_started = time.perf_counter()
    corpus_vectors = normalize_embeddings(
        encoder.encode_images([row["image_path"] for row in corpus], args.batch_size),
        args.dtype,
    )
    corpus_seconds = time.perf_counter() - corpus_started
    query_started = time.perf_counter()
    query_vectors = normalize_embeddings(
        encoder.encode_queries([row["text"] for row in queries], args.batch_size), args.dtype
    )
    query_seconds = time.perf_counter() - query_started
    if len(corpus_vectors) != len(corpus) or len(query_vectors) != len(queries):
        raise RuntimeError("encoder output count does not match input count")
    dimension = int(corpus_vectors[0].shape[1])
    if any(vector.shape[1] != dimension for vector in corpus_vectors + query_vectors):
        raise RuntimeError("ColPali embedding dimensions do not match")

    initial_count = len(corpus)
    if args.initial_corpus_ratio < 1 and len(corpus) > 1:
        initial_count = max(1, int(len(corpus) * args.initial_corpus_ratio))
        initial_count = min(initial_count, len(corpus) - 1)
    initial_records, initial_vectors = flatten_multivectors(
        corpus[:initial_count], corpus_vectors[:initial_count], "document"
    )
    insert_records, insert_vectors = flatten_multivectors(
        corpus[initial_count:], corpus_vectors[initial_count:], "document"
    )
    query_records, flat_query_vectors = flatten_multivectors(queries, query_vectors, "query")

    artifacts = write_shards(
        output_dir, "corpus", initial_records, initial_vectors, args.rows_per_shard
    )
    if insert_records:
        artifacts.extend(
            write_shards(
                output_dir, "inserts", insert_records, insert_vectors, args.rows_per_shard
            )
        )
    artifacts.extend(
        write_shards(
            output_dir, "queries", query_records, flat_query_vectors, args.rows_per_shard
        )
    )
    artifacts.append(
        write_multivector_schedule(
            output_dir,
            len(queries),
            corpus_vectors[initial_count:],
            args.searches_per_insert,
            args.insert_event_size,
        )
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "producer": "vector_workload/export_colpali.py",
        "inputs": {
            "corpus": {"file": args.corpus_file.name, "sha256": sha256_file(args.corpus_file)},
            "queries": {"file": args.query_file.name, "sha256": sha256_file(args.query_file)},
        },
        "embedding": {
            "model": args.model,
            "revision": args.revision or "default",
            "device": args.device,
            "dimension": dimension,
            "dtype": args.dtype,
            "normalized": True,
            "vector_layout": "multi_vector",
            "scoring": "maxsim",
            "batch_size": args.batch_size,
            "seed": args.seed,
            "corpus_seconds": corpus_seconds,
            "query_seconds": query_seconds,
            "gpu": gpu_metadata(args.device),
        },
        "workload": {
            "name": "arxiv-pdf-image",
            "operation": "search_insert_mixed" if insert_records else "search",
            "initial_documents": initial_count,
            "scheduled_insert_documents": len(corpus) - initial_count,
            "initial_vector_rows": len(initial_records),
            "scheduled_insert_vector_rows": len(insert_records),
            "queries": len(queries),
            "schedule": "schedule parquet row order",
        },
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "workload-manifest.yaml"
    with manifest_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False, allow_unicode=True)
    checksum_paths = [output_dir / artifact["path"] for artifact in artifacts] + [manifest_path]
    with (output_dir / "SHA256SUMS").open("w", encoding="utf-8") as stream:
        for path in checksum_paths:
            stream.write(f"{sha256_file(path)}  {path.name}\n")
    verify_artifact(output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "documents": len(corpus),
                "document_vector_rows": sum(len(vector) for vector in corpus_vectors),
                "queries": len(queries),
                "query_vector_rows": sum(len(vector) for vector in query_vectors),
                "dimension": dimension,
                "model": args.model,
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-file", type=Path, required=True)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_COLPALI_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--rows-per-shard", type=int, default=100_000)
    parser.add_argument("--max-corpus", type=int)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--initial-corpus-ratio", type=float, default=1.0)
    parser.add_argument("--searches-per-insert", type=int, default=1)
    parser.add_argument(
        "--insert-event-size", type=int, default=1, help="documents per insert event"
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        export_artifact(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
