from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow.parquet as pq
import yaml

from vector_workload import export_vectors, prepare_workloads, replay_milvus


class FakeEncoder:
    def get_sentence_embedding_dimension(self) -> int:
        return 4

    def encode(self, texts, **kwargs):
        del kwargs
        vectors = []
        for text in texts:
            value = float(sum(text.encode("utf-8")) % 17 + 1)
            vectors.append([value, value + 1, value + 2, value + 3])
        return np.asarray(vectors, dtype=np.float32)


class FakeSchema:
    def add_field(self, **kwargs):
        del kwargs


class FakeIndexParams:
    def add_index(self, **kwargs):
        del kwargs


class FakeMilvusClient:
    last_instance = None

    def __init__(self, **kwargs):
        del kwargs
        self.records = []
        self.collection = None
        FakeMilvusClient.last_instance = self

    @staticmethod
    def create_schema(**kwargs):
        del kwargs
        return FakeSchema()

    def get_server_version(self):
        return "test"

    def has_collection(self, collection):
        del collection
        return False

    def create_collection(self, collection_name, schema, **kwargs):
        del schema, kwargs
        self.collection = collection_name

    def insert(self, collection_name, data):
        assert collection_name == self.collection
        self.records.extend(data)

    def flush(self, collection_name):
        assert collection_name == self.collection

    def prepare_index_params(self):
        return FakeIndexParams()

    def create_index(self, collection_name, index_params):
        del index_params
        assert collection_name == self.collection

    def load_collection(self, collection_name):
        assert collection_name == self.collection

    def search(self, collection_name, data, limit, **kwargs):
        del kwargs
        assert collection_name == self.collection
        query = np.asarray(data[0], dtype=np.float32)
        ranked = sorted(
            self.records,
            key=lambda row: float(np.dot(query, np.asarray(row["vector"]))),
            reverse=True,
        )[:limit]
        return [[{"id": row["id"]} for row in ranked]]


def export_args(corpus_file: Path, query_file: Path, output_dir: Path):
    return argparse.Namespace(
        corpus_file=corpus_file,
        query_file=query_file,
        output_dir=output_dir,
        batch_size=2,
        rows_per_shard=2,
        smoke=True,
        model=None,
        revision=None,
        device="cpu",
        normalize=True,
        dtype="float32",
        chunk_size=0,
        chunk_overlap=0,
        max_corpus=None,
        max_queries=None,
        seed=42,
        initial_corpus_ratio=0.5,
        searches_per_insert=1,
        insert_event_size=1,
    )


class TextMixedReplayTest(unittest.TestCase):
    def test_wikipedia_nq_preparation(self):
        datasets = {
            "wikimedia/wikipedia": [
                {"id": "wiki-1", "title": "One", "url": "u1", "text": "alpha"},
                {"id": "wiki-2", "title": "Two", "url": "u2", "text": "beta"},
            ],
            "sentence-transformers/natural-questions": [
                {"query": "what is alpha?", "answer": "alpha"}
            ],
        }

        def load_dataset(name, *args, **kwargs):
            del args, kwargs
            return datasets[name]

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "prepared"
            args = argparse.Namespace(output_dir=output_dir, corpus_count=2, query_count=1)
            result = prepare_workloads.prepare_wikipedia_nq(args, load_dataset)
            self.assertEqual(result["corpus_documents"], 2)
            self.assertEqual(result["queries"], 1)
            query = json.loads((output_dir / "queries.jsonl").read_text().splitlines()[0])
            self.assertEqual(query["metadata"]["answer"], "alpha")

    def test_export_and_mixed_insert_search_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus.jsonl"
            queries = root / "queries.jsonl"
            corpus.write_text(
                "".join(
                    json.dumps({"id": f"doc-{i}", "text": f"document {i}"}) + "\n"
                    for i in range(4)
                ),
                encoding="utf-8",
            )
            queries.write_text(
                "".join(
                    json.dumps({"id": f"query-{i}", "text": f"question {i}"}) + "\n"
                    for i in range(2)
                ),
                encoding="utf-8",
            )
            artifact_dir = root / "artifact"
            with mock.patch.object(export_vectors, "load_encoder", return_value=FakeEncoder()):
                export_vectors.export_artifact(export_args(corpus, queries, artifact_dir))

            with (artifact_dir / "workload-manifest.yaml").open(encoding="utf-8") as stream:
                manifest = yaml.safe_load(stream)
            self.assertEqual(manifest["embedding"]["vector_layout"], "single_vector")
            schedule = pq.read_table(artifact_dir / "schedule-00000.parquet").to_pydict()
            self.assertEqual(schedule["operation"], ["search", "insert", "search", "insert"])
            result_file = root / "result.json"
            args = argparse.Namespace(
                artifact_dir=artifact_dir,
                max_payload_length=65535,
                collection="mixed-test",
                result_file=result_file,
                uri="test",
                token="test",
                database="default",
                insert_batch_size=2,
                index_type="DISKANN",
                metric="IP",
                search_list=10,
                top_k=2,
                concurrency=2,
                warmup_queries=0,
                max_queries=None,
                respect_delay=False,
                storage_path_note=None,
                consistency_level="Strong",
            )
            with mock.patch.object(replay_milvus, "MilvusClient", FakeMilvusClient):
                replay_milvus.replay(args)

            result = json.loads(result_file.read_text(encoding="utf-8"))
            self.assertEqual(result["initial_load"]["rows_per_second"] >= 0, True)
            self.assertEqual(result["replay"]["queries"], 2)
            self.assertEqual(result["replay"]["insert_events"], 2)
            self.assertEqual(result["replay"]["inserted_rows"], 2)
            self.assertEqual(len(FakeMilvusClient.last_instance.records), 4)


if __name__ == "__main__":
    unittest.main()
