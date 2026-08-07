"""Streaming open-loop replay for RAGPerf Milvus traces."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import pyarrow.parquet as pq

from .artifact import iter_manifest_rows, sha256_file, verify_artifact


class ReplayError(RuntimeError):
    pass


class OnlineHistogram:
    """Bounded logarithmic histogram that never retains individual samples."""

    def __init__(self, accuracy: float = 0.005):
        self.factor = (1 + accuracy) / (1 - accuracy)
        self.log_factor = math.log(self.factor)
        self.buckets: dict[int, int] = {}
        self.count = self.total = self.maximum = 0
        self.minimum: int | None = None

    def record(self, value: int) -> None:
        value = max(0, int(value))
        bucket = 0 if not value else int(math.log(value) / self.log_factor) + 1
        self.buckets[bucket] = self.buckets.get(bucket, 0) + 1
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def percentile(self, value: float) -> int:
        if not self.count:
            return 0
        target, seen = max(1, math.ceil(self.count * value / 100)), 0
        for bucket in sorted(self.buckets):
            seen += self.buckets[bucket]
            if seen >= target:
                return 0 if bucket == 0 else min(self.maximum, int(self.factor**bucket))
        return self.maximum

    def summary(self) -> dict:
        return {
            "count": self.count,
            "min_ns": self.minimum or 0,
            "mean_ns": self.total / self.count if self.count else 0,
            "p50_ns": self.percentile(50),
            "p95_ns": self.percentile(95),
            "p99_ns": self.percentile(99),
            "max_ns": self.maximum,
        }


class PayloadReader:
    """Read references while retaining at most one bounded-size shard."""

    def __init__(self, root: Path):
        self.root, self.path, self.table = root, None, None

    def read(self, references: list[dict]) -> list[dict]:
        result = []
        for reference in references:
            if reference["path"] != self.path:
                self.path = reference["path"]
                self.table = pq.read_table(self.root / self.path)
            result.extend(
                self.table.slice(
                    reference["row_start"], reference["row_count"]
                ).to_pylist()
            )
        return result


@dataclasses.dataclass
class ReplayConfig:
    artifact_dir: Path
    uri: str
    collection: str
    token: str = "root:Milvus"
    timing: str = "original"
    time_scale: float = 1.0
    max_in_flight: int = 1024
    bootstrap_batch_size: int = 1024
    warmup: int = 0
    result_file: Path | None = None

    def validate(self) -> None:
        if self.timing not in {"original", "none"} or self.time_scale <= 0:
            raise ValueError("invalid timing or time scale")
        if self.max_in_flight <= 0 or self.bootstrap_batch_size <= 0:
            raise ValueError("batch and in-flight limits must be positive")


class ReplayStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.latency = {
            name: OnlineHistogram() for name in ("insert", "search", "query")
        }
        self.lag = OnlineHistogram()
        self.counts = {name: 0 for name in self.latency}
        self.failures: list[str] = []
        self.in_flight = self.maximum_in_flight = 0

    def submit(self, operation: str, lag: int) -> None:
        with self.lock:
            self.counts[operation] += 1
            self.lag.record(lag)
            self.in_flight += 1
            self.maximum_in_flight = max(self.maximum_in_flight, self.in_flight)

    def complete(
        self, operation: str, latency: int, error: BaseException | None
    ) -> None:
        with self.lock:
            self.in_flight -= 1
            self.latency[operation].record(latency)
            if error:
                self.failures.append(f"{operation}: {type(error).__name__}: {error}")


class MilvusTraceReplayer:
    def __init__(
        self,
        config: ReplayConfig,
        client: Any = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
    ):
        config.validate()
        self.config, self.client = config, client
        self.clock_ns, self.sleep, self.stats = clock_ns, sleep, ReplayStats()

    def get_client(self):
        if self.client is None:
            from pymilvus import MilvusClient

            self.client = MilvusClient(uri=self.config.uri, token=self.config.token)
        return self.client

    def create_target(self, manifest: dict) -> None:
        client, metadata = self.get_client(), manifest.get("collection", {})
        if client.has_collection(self.config.collection):
            raise ReplayError(
                f"target collection already exists: {self.config.collection}"
            )
        dimension = metadata.get("dimension") or manifest.get("vector", {}).get(
            "dimension"
        )
        if not dimension:
            raise ReplayError("artifact does not contain a vector dimension")
        client.create_collection(
            collection_name=self.config.collection,
            dimension=dimension,
            auto_id=metadata.get("auto_id", True),
            consistency_level=metadata.get("consistency_level", "Eventually"),
        )

    @staticmethod
    def insert_rows(rows: list[dict]) -> list[dict]:
        result = []
        for row in rows:
            item = json.loads(row["scalar_json"])
            item["vector"] = row["vector"]
            result.append(item)
        return result

    def bootstrap(self, manifest: dict) -> tuple[int, int]:
        started, batch, count = self.clock_ns(), [], 0
        for row in iter_manifest_rows(self.config.artifact_dir, manifest, "corpus"):
            batch.append(row)
            if len(batch) == self.config.bootstrap_batch_size:
                self.get_client().insert(
                    collection_name=self.config.collection, data=self.insert_rows(batch)
                )
                count += len(batch)
                batch.clear()
        if batch:
            self.get_client().insert(
                collection_name=self.config.collection, data=self.insert_rows(batch)
            )
            count += len(batch)
        self.get_client().flush(collection_name=self.config.collection)
        return count, self.clock_ns() - started

    def build_index(self, manifest: dict) -> int:
        started, index = self.clock_ns(), manifest.get("index", {})
        if index:
            client = self.get_client()
            if hasattr(client, "list_indexes"):
                existing_indexes = client.list_indexes(
                    collection_name=self.config.collection
                )
                if existing_indexes:
                    client.release_collection(collection_name=self.config.collection)
                    client.drop_index(
                        collection_name=self.config.collection,
                        index_name=existing_indexes[0],
                    )
            params = client.prepare_index_params()
            values = {
                "field_name": index.get("field_name", "vector"),
                "metric_type": index.get("metric_type", "L2"),
                "index_type": index.get("index_type", "AUTOINDEX"),
                "params": index.get("params", {}),
            }
            if index.get("index_name"):
                values["index_name"] = index["index_name"]
            params.add_index(**values)
            client.create_index(
                collection_name=self.config.collection, index_params=params
            )
        self.get_client().load_collection(collection_name=self.config.collection)
        return self.clock_ns() - started

    def invoke(self, event: dict, rows: list[dict]):
        operation, params = event["operation"], json.loads(event["params_json"])
        if operation == "insert":
            params.pop("progress_bar", None)
            return self.get_client().insert(
                collection_name=self.config.collection,
                data=self.insert_rows(rows),
                **params,
            )
        if operation == "search":
            return self.get_client().search(
                collection_name=self.config.collection,
                data=[row["vector"] for row in rows],
                **params,
            )
        if operation == "query":
            request = json.loads(rows[0]["request_json"])
            return self.get_client().query(
                collection_name=self.config.collection,
                filter=request.pop("filter"),
                **request,
            )
        raise ReplayError(f"unsupported operation: {operation}")

    def submit(self, executor, event: dict, rows: list[dict], scheduled: int):
        operation, submitted = event["operation"], self.clock_ns()
        self.stats.submit(operation, max(0, submitted - scheduled))

        def call():
            started, error = self.clock_ns(), None
            try:
                return self.invoke(event, rows)
            except BaseException as caught:
                error = caught
                raise
            finally:
                self.stats.complete(operation, self.clock_ns() - started, error)

        return executor.submit(call)

    def warmup(self, manifest: dict) -> int:
        if not self.config.warmup:
            return 0
        reader = PayloadReader(self.config.artifact_dir)
        for event in iter_manifest_rows(self.config.artifact_dir, manifest, "events"):
            if event["operation"] == "search":
                rows = reader.read(json.loads(event["payload_refs_json"]))
                for _ in range(self.config.warmup):
                    self.invoke(event, rows)
                return self.config.warmup
        return 0

    def replay(self, manifest: dict) -> tuple[int, int]:
        reader, started = PayloadReader(self.config.artifact_dir), self.clock_ns()
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_in_flight
        )
        futures = set()
        try:
            for event in iter_manifest_rows(
                self.config.artifact_dir, manifest, "events"
            ):
                if self.config.timing == "original":
                    target = started + int(
                        event["relative_timestamp_ns"] / self.config.time_scale
                    )
                    remaining = target - self.clock_ns()
                    if remaining > 0:
                        self.sleep(remaining / 1e9)
                else:
                    target = self.clock_ns()
                futures = {future for future in futures if not future.done()}
                if len(futures) >= self.config.max_in_flight:
                    raise ReplayError(
                        f"max-in-flight ({self.config.max_in_flight}) reached at event "
                        f"{event['sequence']}"
                    )
                rows = reader.read(json.loads(event["payload_refs_json"]))
                futures.add(self.submit(executor, event, rows, target))
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception:
                    pass
        finally:
            executor.shutdown(wait=True)
        return sum(self.stats.counts.values()), self.clock_ns() - started

    def run(self) -> dict:
        manifest = verify_artifact(self.config.artifact_dir)
        self.create_target(manifest)
        bootstrap_rows, bootstrap_ns = self.bootstrap(manifest)
        index_ns = self.build_index(manifest)
        warmup_requests = self.warmup(manifest)
        events, replay_ns = self.replay(manifest)
        result = {
            "artifact": {
                "format": manifest["format"],
                "manifest_sha256": sha256_file(
                    self.config.artifact_dir / "workload-manifest.yaml"
                ),
            },
            "target": {
                "uri": self.config.uri,
                "collection": self.config.collection,
                "client_version": self.client_version(),
                "server_version": self.server_version(),
            },
            "bootstrap": {
                "rows": bootstrap_rows,
                "insert_and_flush_ns": bootstrap_ns,
                "index_and_load_ns": index_ns,
                "warmup_requests": warmup_requests,
            },
            "replay": {
                "events": events,
                "elapsed_ns": replay_ns,
                "throughput_ops_per_second": (
                    events / (replay_ns / 1e9) if replay_ns else 0
                ),
                "operation_counts": self.stats.counts,
                "latency": {
                    key: value.summary() for key, value in self.stats.latency.items()
                },
                "scheduler_lag": self.stats.lag.summary(),
                "maximum_in_flight": self.stats.maximum_in_flight,
                "failures": self.stats.failures,
                "timing": self.config.timing,
                "time_scale": self.config.time_scale,
            },
        }
        if self.config.result_file:
            self.config.result_file.parent.mkdir(parents=True, exist_ok=True)
            with self.config.result_file.open("w", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2)
                stream.write("\n")
        if self.stats.failures:
            raise ReplayError(f"{len(self.stats.failures)} replay request(s) failed")
        return result

    @staticmethod
    def client_version():
        try:
            import pymilvus

            return pymilvus.__version__
        except Exception:
            return None

    def server_version(self):
        try:
            return self.get_client().get_server_version()
        except Exception:
            return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay a RAGPerf Milvus request trace"
    )
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--uri", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--token", default="root:Milvus")
    parser.add_argument("--timing", choices=("original", "none"), default="original")
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--max-in-flight", type=int, default=1024)
    parser.add_argument("--bootstrap-batch-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--result-file", type=Path)
    args = parser.parse_args(argv)
    MilvusTraceReplayer(ReplayConfig(**vars(args))).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
