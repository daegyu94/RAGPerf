#!/usr/bin/env python
"""Load a vector artifact into Milvus DISKANN and replay its query schedule."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow.parquet as pq
import yaml
from pymilvus import DataType, MilvusClient

try:
    from artifact_utils import verify_artifact
except ModuleNotFoundError:  # Supports `python -m vector_workload.replay_milvus`.
    from vector_workload.artifact_utils import verify_artifact


def percentile_summary(values_ms: list[float]) -> dict[str, float]:
    if not values_ms:
        return {}
    values = np.asarray(values_ms, dtype=np.float64)
    return {
        "min": float(values.min()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def artifact_shards(manifest: dict[str, Any], artifact_dir: Path, kind: str) -> list[Path]:
    paths = [
        artifact_dir / artifact["path"]
        for artifact in manifest["artifacts"]
        if artifact["kind"] == kind
    ]
    if not paths:
        raise ValueError(f"artifact contains no {kind} shards")
    return paths


def iter_rows(path: Path, columns: list[str]) -> Iterator[dict[str, Any]]:
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=1024, columns=columns
    ):
        data = batch.to_pydict()
        for row in range(batch.num_rows):
            yield {column: data[column][row] for column in columns}


def iter_queries(paths: list[Path], max_queries: int | None) -> Iterator[dict[str, Any]]:
    emitted = 0
    for path in paths:
        for row in iter_rows(path, ["id", "metadata_json", "vector", "delay_ms"]):
            metadata = json.loads(row["metadata_json"] or "{}")
            yield {
                "id": row["id"],
                "vector": np.asarray(row["vector"], dtype=np.float32),
                "delay_ms": float(row["delay_ms"]),
                "expected_id": metadata.get("expected_id"),
            }
            emitted += 1
            if max_queries is not None and emitted >= max_queries:
                return


def create_schema(dimension: int, max_payload_length: int) -> Any:
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(
        field_name="id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        max_length=max_payload_length,
    )
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=max_payload_length,
    )
    schema.add_field(
        field_name="metadata_json",
        datatype=DataType.VARCHAR,
        max_length=max_payload_length,
    )
    schema.add_field(
        field_name="vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=dimension,
    )
    return schema


def insert_shard(
    client: MilvusClient,
    collection: str,
    path: Path,
    insert_batch_size: int,
    max_payload_length: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    rows = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=insert_batch_size):
        data = batch.to_pydict()
        records: list[dict[str, Any]] = []
        for row in range(batch.num_rows):
            values = {
                "id": str(data["id"][row]),
                "text": str(data["text"][row]),
                "metadata_json": str(data["metadata_json"][row] or "{}"),
                # Milvus FLOAT_VECTOR is float32. This also normalizes float16 artifacts.
                "vector": np.asarray(data["vector"][row], dtype=np.float32).tolist(),
            }
            for field in ("id", "text", "metadata_json"):
                if len(values[field].encode("utf-8")) > max_payload_length:
                    raise ValueError(
                        f"{path.name}:{field} exceeds Milvus max_length={max_payload_length}"
                    )
            records.append(values)
        client.insert(collection_name=collection, data=records)
        rows += len(records)
    seconds = time.perf_counter() - started
    return {
        "path": path.name,
        "rows": rows,
        "seconds": seconds,
        "rows_per_second": rows / seconds if seconds else 0.0,
    }


def result_id(result: dict[str, Any]) -> str | None:
    value = result.get("id")
    if value is not None:
        return str(value)
    entity = result.get("entity") or {}
    value = entity.get("id")
    return None if value is None else str(value)


def run_search(
    client: MilvusClient,
    collection: str,
    query: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    search_params = {
        "metric_type": args.metric,
        "params": {"search_list": args.search_list},
    }
    response = client.search(
        collection_name=collection,
        data=[query["vector"].tolist()],
        anns_field="vector",
        limit=args.top_k,
        output_fields=["id"],
        search_params=search_params,
    )
    rows = response[0] if response else []
    latency_ms = (time.perf_counter() - started) * 1000
    ids = [result_id(row) for row in rows]
    expected_id = query["expected_id"]
    return {
        "id": query["id"],
        "latency_ms": latency_ms,
        "top1_correct": expected_id is not None and bool(ids) and ids[0] == expected_id,
        "expected_neighbor_available": expected_id is not None,
        "result_count": len(ids),
    }


def replay(args: argparse.Namespace) -> None:
    artifact_dir = args.artifact_dir.resolve()
    verify_artifact(artifact_dir)
    with (artifact_dir / "workload-manifest.yaml").open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)

    dimension = int(manifest["embedding"]["dimension"])
    if args.max_payload_length <= 0 or args.max_payload_length > 65535:
        raise ValueError("max-payload-length must be between 1 and 65535")
    if args.collection == "default":
        raise ValueError("use a non-default collection name for a replay run")

    result_file = args.result_file.resolve() if args.result_file else Path(
        f"milvus-{args.collection}-replay-result.json"
    ).resolve()
    if result_file.exists():
        raise FileExistsError(f"result file already exists: {result_file}")

    client = MilvusClient(uri=args.uri, token=args.token, db_name=args.database)
    server_version = (
        client.get_server_version() if hasattr(client, "get_server_version") else "unknown"
    )
    if client.has_collection(args.collection):
        raise FileExistsError(
            f"Milvus collection already exists; choose a new collection: {args.collection}"
        )
    schema = create_schema(dimension, args.max_payload_length)
    client.create_collection(collection_name=args.collection, schema=schema)

    corpus_paths = artifact_shards(manifest, artifact_dir, "corpus")
    query_paths = artifact_shards(manifest, artifact_dir, "queries")
    insert_started = time.perf_counter()
    insert_batches = [
        insert_shard(
            client,
            args.collection,
            path,
            args.insert_batch_size,
            args.max_payload_length,
        )
        for path in corpus_paths
    ]
    client.flush(collection_name=args.collection)
    insert_seconds = time.perf_counter() - insert_started
    inserted_rows = sum(batch["rows"] for batch in insert_batches)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type=args.index_type,
        metric_type=args.metric,
        index_name="vector_index",
    )
    index_started = time.perf_counter()
    client.create_index(collection_name=args.collection, index_params=index_params)
    index_seconds = time.perf_counter() - index_started
    client.load_collection(collection_name=args.collection)

    queries = list(iter_queries(query_paths, args.max_queries))
    if not queries:
        raise ValueError("no queries selected for replay")
    warmup_count = min(args.warmup_queries, len(queries))
    warmup_started = time.perf_counter()
    for query in queries[:warmup_count]:
        run_search(client, args.collection, query, args)
    warmup_seconds = time.perf_counter() - warmup_started

    replay_started = time.perf_counter()
    query_results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures: list[concurrent.futures.Future[dict[str, Any]]] = []
        for query in queries:
            if args.respect_delay and query["delay_ms"]:
                time.sleep(query["delay_ms"] / 1000)
            futures.append(
                executor.submit(run_search, client, args.collection, query, args)
            )
        for future in concurrent.futures.as_completed(futures):
            query_results.append(future.result())
    replay_seconds = time.perf_counter() - replay_started

    latencies = [result["latency_ms"] for result in query_results]
    scored = [result for result in query_results if result["expected_neighbor_available"]]
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact_dir": str(artifact_dir),
        "database": {
            "engine": "milvus",
            "client_version": getattr(__import__("pymilvus"), "__version__", "unknown"),
            "server_version": server_version,
            "uri": args.uri,
            "database": args.database,
            "collection": args.collection,
            "rows": inserted_rows,
            "index_type": args.index_type,
            "metric": args.metric,
            "storage_path_note": args.storage_path_note,
        },
        "insert": {
            "seconds": insert_seconds,
            "rows_per_second": inserted_rows / insert_seconds if insert_seconds else 0.0,
            "batches": insert_batches,
        },
        "index": {
            "type": args.index_type,
            "seconds": index_seconds,
            "search_list": args.search_list,
        },
        "replay": {
            "queries": len(query_results),
            "concurrency": args.concurrency,
            "top_k": args.top_k,
            "metric": args.metric,
            "respect_delay": args.respect_delay,
            "warmup_queries": warmup_count,
            "warmup_seconds": warmup_seconds,
            "seconds": replay_seconds,
            "queries_per_second": len(query_results) / replay_seconds
            if replay_seconds
            else 0.0,
            "latency_ms": percentile_summary(latencies),
            "expected_neighbor_queries": len(scored),
            "top1_recall": (
                sum(result["top1_correct"] for result in scored) / len(scored)
                if scored
                else None
            ),
        },
    }
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with result_file.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    print(f"Result written to {result_file}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--uri", default="http://localhost:19530")
    parser.add_argument("--token", default="root:Milvus")
    parser.add_argument("--database", default="default")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--result-file", type=Path)
    parser.add_argument("--storage-path-note", default=None)
    parser.add_argument("--insert-batch-size", type=int, default=10_000)
    parser.add_argument("--index-type", choices=("DISKANN",), default="DISKANN")
    parser.add_argument("--metric", choices=("COSINE", "L2", "IP"), default="COSINE")
    parser.add_argument("--search-list", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup-queries", type=int, default=100)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--respect-delay", action="store_true")
    parser.add_argument("--max-payload-length", type=int, default=65535)
    return parser


def main() -> int:
    try:
        args = build_parser().parse_args()
        if args.insert_batch_size <= 0 or args.concurrency <= 0 or args.top_k <= 0:
            raise ValueError("batch size, concurrency and top-k must be positive")
        if args.warmup_queries < 0 or args.search_list <= 0:
            raise ValueError("warmup-queries must be non-negative and search-list must be positive")
        if args.max_queries is not None and args.max_queries <= 0:
            raise ValueError("max-queries must be positive")
        replay(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
