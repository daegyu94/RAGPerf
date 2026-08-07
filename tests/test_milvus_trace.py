import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pyarrow.parquet as pq

from milvus_trace.artifact import iter_manifest_rows, verify_artifact
from milvus_trace.recorder import TraceConfig, TraceRecorder, TraceRecordingError
from milvus_trace.replay import MilvusTraceReplayer, ReplayConfig


class Clock:
    def __init__(self, value=0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, nanoseconds):
        self.value += nanoseconds

    def sleep(self, seconds):
        self.advance(round(seconds * 1_000_000_000))


class IndexParams:
    def __init__(self):
        self.values = []

    def add_index(self, **kwargs):
        self.values.append(kwargs)


class FakeMilvus:
    def __init__(self):
        self.calls = []

    def has_collection(self, name):
        return False

    def create_collection(self, **kwargs):
        self.calls.append(("create_collection", kwargs))

    def insert(self, **kwargs):
        self.calls.append(("insert", kwargs))

    def flush(self, **kwargs):
        self.calls.append(("flush", kwargs))

    def list_indexes(self, **kwargs):
        return ["vector"]

    def release_collection(self, **kwargs):
        self.calls.append(("release_collection", kwargs))

    def drop_index(self, **kwargs):
        self.calls.append(("drop_index", kwargs))

    def prepare_index_params(self):
        self.index = IndexParams()
        return self.index

    def create_index(self, **kwargs):
        self.calls.append(("create_index", kwargs))

    def load_collection(self, **kwargs):
        self.calls.append(("load_collection", kwargs))

    def search(self, **kwargs):
        self.calls.append(("search", kwargs))

    def query(self, **kwargs):
        self.calls.append(("query", kwargs))

    def get_server_version(self):
        return "fake"


class TraceRoundTripTest(unittest.TestCase):
    def record(self, root: Path):
        clock = Clock(10_000)
        recorder = TraceRecorder(
            TraceConfig(enabled=True, output_dir=str(root), rows_per_shard=2), clock
        )
        recorder.describe_collection(
            name="source", dimension=2, auto_id=True, consistency_level="Eventually"
        )
        recorder.record_insert(
            "source",
            [
                {"vector": [1, 2], "text": "bootstrap", "doc_id": 7},
                {"vector": [3, 4], "text": "bootstrap two", "doc_id": 8},
            ],
        )
        recorder.mark_index_created(index_type="IVF_FLAT", metric_type="L2")
        recorder.begin_timed_workload()
        clock.advance(50)
        recorder.record_search(
            "source",
            [[0.1, 0.2]],
            limit=3,
            filter="doc_id > 0",
            output_fields=["text"],
            search_params={"metric_type": "L2", "params": {"nprobe": 4}},
            consistency_level="Eventually",
        )
        clock.advance(75)
        recorder.record_query(
            "source", "doc_id in [7]", output_fields=["text", "vector"], limit=10
        )
        clock.advance(25)
        recorder.record_insert(
            "source", [{"vector": [5, 6], "text": "timed", "metadata": {"x": True}}]
        )
        recorder.close()

    def test_payload_timing_and_replay_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.record(root)
            manifest = verify_artifact(root)
            self.assertEqual(manifest["format"], "ragperf-milvus-trace")
            self.assertEqual(manifest["event_count"], 3)
            events = list(iter_manifest_rows(root, manifest, "events"))
            self.assertEqual(
                [row["operation"] for row in events], ["search", "query", "insert"]
            )
            self.assertEqual(
                [row["relative_timestamp_ns"] for row in events], [50, 125, 150]
            )
            corpus = list(iter_manifest_rows(root, manifest, "corpus"))
            self.assertEqual(json.loads(corpus[0]["scalar_json"])["text"], "bootstrap")
            all_bytes = b"".join(
                path.read_bytes() for path in root.iterdir() if path.is_file()
            )
            self.assertNotIn(b"What is the original question?", all_bytes)

            client, clock = FakeMilvus(), Clock()
            result = MilvusTraceReplayer(
                ReplayConfig(root, "fake://milvus", "target", time_scale=2),
                client=client,
                clock_ns=clock,
                sleep=clock.sleep,
            ).run()
            self.assertEqual(result["bootstrap"]["rows"], 2)
            self.assertEqual(
                result["replay"]["operation_counts"],
                {"insert": 1, "search": 1, "query": 1},
            )
            operations = [name for name, _ in client.calls]
            self.assertEqual(
                operations[:7],
                [
                    "create_collection",
                    "insert",
                    "flush",
                    "release_collection",
                    "drop_index",
                    "create_index",
                    "load_collection",
                ],
            )
            search = next(value for name, value in client.calls if name == "search")
            self.assertEqual(search["search_params"]["params"]["nprobe"], 4)
            timed_insert = [value for name, value in client.calls if name == "insert"][
                -1
            ]
            self.assertEqual(timed_insert["data"][0]["metadata"], {"x": True})

    def test_trace_output_dir_expands_environment_variables(self):
        with mock.patch.dict("os.environ", {"MNTPNT": "/var/tmp/ragperf"}):
            config = TraceConfig.from_mapping(
                {"enabled": True, "output_dir": "${MNTPNT}/traces/run-001"}
            )
        self.assertEqual(config.output_dir, "/var/tmp/ragperf/traces/run-001")

    def test_trace_output_dir_rejects_unset_environment_variables(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "unset environment variable"):
                TraceConfig.from_mapping(
                    {"enabled": True, "output_dir": "${MNTPNT}/traces/run-001"}
                )

    def test_overflow_marks_artifact_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = TraceRecorder(
                TraceConfig(enabled=True, output_dir=directory, max_queue_bytes=1)
            )
            recorder.describe_collection(dimension=2)
            recorder.record_insert("source", [{"vector": [1, 2], "text": "x"}])
            with self.assertRaises(TraceRecordingError):
                recorder.close()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                verify_artifact(directory)

    def test_unsupported_scalar_fails_before_enqueue(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = TraceRecorder(TraceConfig(enabled=True, output_dir=directory))
            with self.assertRaisesRegex(TypeError, "unsupported type"):
                recorder.record_insert("source", [{"vector": [1, 2], "bad": object()}])
            recorder.close()


if __name__ == "__main__":
    unittest.main()
