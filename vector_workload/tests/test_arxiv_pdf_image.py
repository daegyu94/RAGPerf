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

from vector_workload import export_colpali, prepare_workloads, replay_milvus
from vector_workload.tests.test_text_mixed_replay import FakeIndexParams, FakeSchema


class FakeImage:
    def __init__(self, content: bytes):
        self.content = content

    def save(self, path: Path, image_format: str):
        assert image_format == "PNG"
        path.write_bytes(self.content)

    def close(self):
        pass


def fake_pdf_converter(path: Path):
    return [FakeImage(path.stem.encode() + b"-1"), FakeImage(path.stem.encode() + b"-2")]


class FakeColPaliEncoder:
    @staticmethod
    def _vectors(count: int):
        vectors = []
        for index in range(count):
            matrix = np.zeros((2, 4), dtype=np.float32)
            matrix[0, index % 4] = 1
            matrix[1, (index + 1) % 4] = 1
            vectors.append(matrix)
        return vectors

    def encode_images(self, paths, batch_size):
        del batch_size
        return self._vectors(len(paths))

    def encode_queries(self, texts, batch_size):
        del batch_size
        return self._vectors(len(texts))


class FakeMultiVectorMilvusClient:
    last_instance = None

    def __init__(self, **kwargs):
        del kwargs
        self.records = []
        self.collection = None
        FakeMultiVectorMilvusClient.last_instance = self

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
        responses = []
        for query in data:
            query_array = np.asarray(query, dtype=np.float32)
            ranked = sorted(
                self.records,
                key=lambda row: float(np.dot(query_array, np.asarray(row["vector"]))),
                reverse=True,
            )[:limit]
            responses.append(
                [
                    {
                        "distance": float(np.dot(query_array, np.asarray(row["vector"]))),
                        "entity": {"group_id": row["group_id"]},
                    }
                    for row in ranked
                ]
            )
        return responses


class ArxivPdfImageTest(unittest.TestCase):
    def test_prepare_export_and_multivector_mixed_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_dir = root / "pdfs"
            pdf_dir.mkdir()
            (pdf_dir / "paper-a.pdf").write_bytes(b"pdf-a")
            (pdf_dir / "paper-b.pdf").write_bytes(b"pdf-b")
            query_file = root / "queries.jsonl"
            query_file.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "query-0",
                                "text": "first page",
                                "metadata": {"expected_doc_id": "paper-a-page-00001"},
                            }
                        ),
                        json.dumps({"id": "query-1", "text": "second page"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            prepared = root / "prepared"
            prepare_args = argparse.Namespace(
                pdf_dir=pdf_dir,
                query_file=query_file,
                output_dir=prepared,
                max_pdfs=None,
            )
            prepared_result = prepare_workloads.prepare_arxiv_pdf_image(
                prepare_args, fake_pdf_converter
            )
            self.assertEqual(prepared_result["page_images"], 4)

            artifact = root / "artifact"
            export_args = argparse.Namespace(
                corpus_file=prepared / "corpus.jsonl",
                query_file=prepared / "queries.jsonl",
                output_dir=artifact,
                model="fake-colpali",
                revision="test",
                device="cpu",
                batch_size=2,
                dtype="float32",
                rows_per_shard=100,
                max_corpus=None,
                max_queries=None,
                initial_corpus_ratio=0.5,
                searches_per_insert=1,
                insert_event_size=1,
                seed=42,
            )
            export_colpali.export_artifact(export_args, FakeColPaliEncoder())
            manifest = yaml.safe_load(
                (artifact / "workload-manifest.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["embedding"]["vector_layout"], "multi_vector")
            schedule = pq.read_table(artifact / "schedule-00000.parquet").to_pydict()
            self.assertEqual(schedule["operation"], ["search", "insert", "search", "insert"])
            self.assertEqual(schedule["count"], [1, 2, 1, 2])

            result_file = root / "result.json"
            replay_args = argparse.Namespace(
                artifact_dir=artifact,
                max_payload_length=65535,
                collection="colpali-mixed-test",
                result_file=result_file,
                uri="test",
                token="test",
                database="default",
                insert_batch_size=100,
                index_type="DISKANN",
                metric="IP",
                search_list=10,
                top_k=2,
                token_top_k=10,
                concurrency=1,
                warmup_queries=0,
                max_queries=None,
                respect_delay=False,
                storage_path_note=None,
                consistency_level="Strong",
            )
            with mock.patch.object(
                replay_milvus, "MilvusClient", FakeMultiVectorMilvusClient
            ):
                replay_milvus.replay(replay_args)
            result = json.loads(result_file.read_text(encoding="utf-8"))
            self.assertEqual(result["database"]["vector_layout"], "multi_vector")
            self.assertEqual(result["database"]["rows"], 8)
            self.assertEqual(result["replay"]["queries"], 2)
            self.assertEqual(result["replay"]["insert_events"], 2)
            self.assertEqual(result["replay"]["inserted_rows"], 4)
            self.assertEqual(result["replay"]["top1_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
