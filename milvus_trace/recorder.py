"""Non-blocking, bounded-memory Milvus trace recorder."""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .artifact import FORMAT_NAME, sha256_file


class TraceRecordingError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class TraceConfig:
    enabled: bool = False
    output_dir: str = ""
    max_queue_bytes: int = 256 * 1024 * 1024
    rows_per_shard: int = 65_536
    compression: str = "zstd"
    on_overflow: str = "invalidate"

    @classmethod
    def from_mapping(cls, value: dict | None) -> "TraceConfig":
        value = dict(value or {})
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"unknown trace setting(s): {', '.join(sorted(unknown))}")
        if "output_dir" in value:
            value["output_dir"] = os.path.expanduser(
                os.path.expandvars(value["output_dir"])
            )
            unresolved = re.findall(
                r"\$(?:\{[^}]+\}|[A-Za-z_][A-Za-z0-9_]*)", value["output_dir"]
            )
            if unresolved:
                raise ValueError(
                    "trace.output_dir contains unset environment variable(s): "
                    + ", ".join(unresolved)
                )
        config = cls(**value)
        if config.enabled and not config.output_dir:
            raise ValueError("trace.output_dir is required when tracing is enabled")
        if config.max_queue_bytes <= 0 or config.rows_per_shard <= 0:
            raise ValueError("trace queue and shard limits must be positive")
        if config.on_overflow != "invalidate":
            raise ValueError("only trace.on_overflow=invalidate is supported")
        return config


_PAYLOAD_SCHEMAS = {
    "corpus": pa.schema(
        [("vector", pa.list_(pa.float32())), ("scalar_json", pa.string())]
    ),
    "inserts": pa.schema(
        [("vector", pa.list_(pa.float32())), ("scalar_json", pa.string())]
    ),
    "searches": pa.schema([("vector", pa.list_(pa.float32()))]),
    "scalar-queries": pa.schema([("request_json", pa.string())]),
    "events": pa.schema(
        [
            ("sequence", pa.int64()),
            ("relative_timestamp_ns", pa.int64()),
            ("operation", pa.string()),
            ("collection", pa.string()),
            ("batch_size", pa.int64()),
            ("session_id", pa.string()),
            ("params_json", pa.string()),
            ("payload_refs_json", pa.string()),
        ]
    ),
}


def _json_value(value: Any, path: str = "value") -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{path} contains a non-string mapping key")
        return {key: _json_value(item, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{path}[]") for item in value]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist(), path)
    if hasattr(value, "item"):
        return _json_value(value.item(), path)
    raise TypeError(f"{path} has unsupported type {type(value).__name__}")


def _json_dump(value: Any) -> str:
    return json.dumps(_json_value(value), ensure_ascii=False, separators=(",", ":"))


def _vectors(value: Any) -> list[list[float]]:
    converted = _json_value(value, "vector data")
    if not isinstance(converted, list):
        raise TypeError("vector data must be a list or array")
    if not converted:
        return []
    if isinstance(converted[0], (int, float)):
        converted = [converted]
    if any(not isinstance(row, list) for row in converted):
        raise TypeError("vector data must be one- or two-dimensional")
    try:
        return [[float(item) for item in row] for row in converted]
    except (TypeError, ValueError) as error:
        raise TypeError("vector data contains a non-numeric value") from error


class _ByteQueue:
    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.bytes = 0
        self.items: deque[tuple[int, dict] | None] = deque()
        self.condition = threading.Condition()

    def put_nowait(self, item: dict, size: int) -> bool:
        with self.condition:
            if size > self.max_bytes - self.bytes:
                return False
            self.items.append((size, item))
            self.bytes += size
            self.condition.notify()
            return True

    def get(self) -> dict | None:
        with self.condition:
            while not self.items:
                self.condition.wait()
            entry = self.items.popleft()
            if entry is None:
                return None
            size, item = entry
            self.bytes -= size
            return item

    def close(self) -> None:
        with self.condition:
            self.items.append(None)
            self.condition.notify()


class _ShardWriter:
    def __init__(self, root: Path, kind: str, rows_per_shard: int, compression: str):
        self.root = root
        self.kind = kind
        self.rows_per_shard = rows_per_shard
        self.compression = compression
        self.index = 0
        self.rows = 0
        self.total_rows = 0
        self.writer: pq.ParquetWriter | None = None
        self.files: list[dict] = []

    def _open(self) -> None:
        name = f"{self.kind}-{self.index:05d}.parquet"
        self.writer = pq.ParquetWriter(
            self.root / name, _PAYLOAD_SCHEMAS[self.kind], compression=self.compression
        )
        self.files.append({"kind": self.kind, "path": name, "rows": 0})

    def append(self, rows: list[dict]) -> list[dict]:
        refs = []
        offset = 0
        while offset < len(rows):
            if self.writer is None:
                self._open()
            count = min(len(rows) - offset, self.rows_per_shard - self.rows)
            chunk = rows[offset : offset + count]
            table = pa.Table.from_pylist(chunk, schema=_PAYLOAD_SCHEMAS[self.kind])
            self.writer.write_table(table)
            refs.append(
                {
                    "path": self.files[-1]["path"],
                    "row_start": self.rows,
                    "row_count": count,
                }
            )
            self.rows += count
            self.total_rows += count
            self.files[-1]["rows"] += count
            offset += count
            if self.rows == self.rows_per_shard:
                self.writer.close()
                self.writer = None
                self.rows = 0
                self.index += 1
        return refs

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


class TraceRecorder:
    """Capture requests before they are submitted without waiting for disk I/O."""

    def __init__(
        self,
        config: TraceConfig | dict,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        session_id: str | None = None,
    ):
        self.config = (
            config
            if isinstance(config, TraceConfig)
            else TraceConfig.from_mapping(config)
        )
        self.clock_ns = clock_ns
        self.session_id = session_id or f"pid-{os.getpid()}"
        self.root = Path(self.config.output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.queue = _ByteQueue(self.config.max_queue_bytes)
        self.lock = threading.Lock()
        self.sequence = 0
        self.marker_ns: int | None = None
        self.index_created = False
        self.closed = False
        self.incomplete_reason: str | None = None
        self.collection: dict[str, Any] = {}
        self.index: dict[str, Any] = {}
        self.event_counts = {"insert": 0, "search": 0, "query": 0}
        self.thread = threading.Thread(
            target=self._run_writer, name="milvus-trace-writer"
        )
        self.thread.start()

    def describe_collection(self, **metadata: Any) -> None:
        with self.lock:
            self.collection.update(_json_value(metadata, "collection metadata"))

    def mark_index_created(self, **metadata: Any) -> None:
        with self.lock:
            self.index_created = True
            self.index.update(_json_value(metadata, "index metadata"))

    def begin_timed_workload(self, timestamp_ns: int | None = None) -> None:
        with self.lock:
            if self.marker_ns is not None:
                return
            self.marker_ns = self.clock_ns() if timestamp_ns is None else timestamp_ns

    def _next_event(self, timestamp_ns: int, operation: str) -> tuple[int, int]:
        with self.lock:
            if self.marker_ns is None:
                self.marker_ns = timestamp_ns
            sequence = self.sequence
            self.sequence += 1
            self.event_counts[operation] += 1
            return sequence, timestamp_ns - self.marker_ns

    def _enqueue(self, item: dict) -> None:
        encoded_size = len(_json_dump(item).encode("utf-8"))
        if not self.queue.put_nowait(item, encoded_size):
            self._invalidate(
                f"trace queue overflow: {encoded_size} bytes could not fit in "
                f"{self.config.max_queue_bytes}-byte queue"
            )

    def _invalidate(self, reason: str) -> None:
        with self.lock:
            if self.incomplete_reason is None:
                self.incomplete_reason = reason

    def record_insert(
        self, collection: str, data: Any, timestamp_ns: int | None = None, **params: Any
    ) -> None:
        rows = []
        for row in data:
            if not isinstance(row, dict) or "vector" not in row:
                raise TypeError("each insert row must be a mapping containing vector")
            vectors = _vectors(row["vector"])
            if len(vectors) != 1:
                raise TypeError("each insert row must contain exactly one vector")
            scalars = {key: value for key, value in row.items() if key != "vector"}
            rows.append({"vector": vectors[0], "scalar_json": _json_dump(scalars)})
        now = self.clock_ns() if timestamp_ns is None else timestamp_ns
        with self.lock:
            bootstrap = not self.index_created and self.marker_ns is None
        if bootstrap:
            self._enqueue({"kind": "corpus", "rows": rows})
            return
        sequence, relative = self._next_event(now, "insert")
        self._enqueue(
            self._event_item(sequence, relative, "insert", collection, rows, params)
        )

    def record_search(
        self, collection: str, data: Any, timestamp_ns: int | None = None, **params: Any
    ) -> None:
        rows = [{"vector": vector} for vector in _vectors(data)]
        now = self.clock_ns() if timestamp_ns is None else timestamp_ns
        sequence, relative = self._next_event(now, "search")
        self._enqueue(
            self._event_item(sequence, relative, "search", collection, rows, params)
        )

    def record_query(
        self,
        collection: str,
        filter_expr: str,
        timestamp_ns: int | None = None,
        **params: Any,
    ) -> None:
        request = {"filter": filter_expr, **params}
        request_json = _json_dump(request)
        now = self.clock_ns() if timestamp_ns is None else timestamp_ns
        sequence, relative = self._next_event(now, "query")
        item = self._event_item(
            sequence,
            relative,
            "query",
            collection,
            [{"request_json": request_json}],
            {},
        )
        self._enqueue(item)

    def _event_item(
        self,
        sequence: int,
        relative: int,
        operation: str,
        collection: str,
        rows: list[dict],
        params: dict,
    ) -> dict:
        return {
            "kind": {
                "insert": "inserts",
                "search": "searches",
                "query": "scalar-queries",
            }[operation],
            "rows": rows,
            "event": {
                "sequence": sequence,
                "relative_timestamp_ns": relative,
                "operation": operation,
                "collection": collection,
                "batch_size": len(rows),
                "session_id": self.session_id,
                "params_json": _json_dump(params),
            },
        }

    def _run_writer(self) -> None:
        writers = {
            kind: _ShardWriter(
                self.root, kind, self.config.rows_per_shard, self.config.compression
            )
            for kind in _PAYLOAD_SCHEMAS
        }
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                refs = writers[item["kind"]].append(item["rows"])
                if "event" in item:
                    event = dict(item["event"])
                    event["payload_refs_json"] = _json_dump(refs)
                    writers["events"].append([event])
        except Exception as error:  # the application request path must remain available
            self._invalidate(f"trace writer failed: {type(error).__name__}: {error}")
        finally:
            for writer in writers.values():
                try:
                    writer.close()
                except Exception as error:
                    self._invalidate(
                        f"trace writer close failed: {type(error).__name__}: {error}"
                    )
            self._write_manifest(writers)

    def _write_manifest(self, writers: dict[str, _ShardWriter]) -> None:
        artifacts = []
        for writer in writers.values():
            for item in writer.files:
                item["sha256"] = sha256_file(self.root / item["path"])
                artifacts.append(item)
        with self.lock:
            manifest = {
                "format": FORMAT_NAME,
                "incomplete": self.incomplete_reason is not None,
                "incomplete_reason": self.incomplete_reason,
                "timing_model": "monotonic-open-loop-arrival",
                "session_id": self.session_id,
                "event_count": sum(self.event_counts.values()),
                "operation_counts": dict(self.event_counts),
                "rows_per_shard": self.config.rows_per_shard,
                "compression": self.config.compression,
                "collection": dict(self.collection),
                "index": dict(self.index),
                "vector": {
                    "dimension": self.collection.get("dimension"),
                    "dtype": "float32",
                },
                "recorder": {"python": platform.python_version()},
                "artifacts": artifacts,
            }
        manifest_path = self.root / "workload-manifest.yaml"
        with manifest_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(manifest, stream, sort_keys=False)
        checksum_files = [manifest_path] + [
            self.root / item["path"] for item in artifacts
        ]
        with (self.root / "SHA256SUMS").open("w", encoding="utf-8") as stream:
            for path in checksum_files:
                stream.write(f"{sha256_file(path)}  {path.name}\n")

    def close(self) -> None:
        if self.closed:
            if self.incomplete_reason:
                raise TraceRecordingError(self.incomplete_reason)
            return
        self.closed = True
        self.queue.close()
        self.thread.join()
        if self.incomplete_reason:
            raise TraceRecordingError(self.incomplete_reason)

    def __enter__(self) -> "TraceRecorder":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
