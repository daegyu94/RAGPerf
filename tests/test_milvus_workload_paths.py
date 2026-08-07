import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from RAGPipeline.retriever.BaseRetriever import BaseRetriever
from vectordb.milvus_api import milvus_client


class FakeMilvusTransport:
    def __init__(self):
        self.exists = False
        self.inserted = []

    def has_collection(self, _name):
        return self.exists

    def create_collection(self, *_args, **_kwargs):
        self.exists = True

    def insert(self, collection_name, data, **_kwargs):
        self.inserted.extend(data)
        return {"insert_count": len(data), "collection": collection_name}

    def load_collection(self, _name):
        return None

    def search(self, collection_name, data, **_kwargs):
        return [
            [{"entity": {"doc_id": 7}, "id": 1, "distance": 0.0}]
            for _ in data
        ]


class FakeImageClient:
    type = "milvus"

    def __init__(self, filepath):
        self.filepath = filepath
        self.filter_expr = None

    def query_search_image(self, *_args, **_kwargs):
        return {7}

    def query(self, collection_name, filter_expr, **_kwargs):
        self.filter_expr = filter_expr
        return [
            {"vector": [1.0, 0.0], "filepath": self.filepath},
            {"vector": [0.0, 1.0], "filepath": self.filepath},
        ]


class MilvusWorkloadPathTest(unittest.TestCase):
    def test_image_insert_continues_after_creating_collection(self):
        wrapper = milvus_client(
            db_path="fake://milvus",
            collection_name="image",
            trace=None,
        )
        wrapper.client = FakeMilvusTransport()
        rows = [{"vector": [0.1, 0.2], "doc_id": 1}]
        wrapper.insert_data(
            rows,
            collection_name="image",
            insert_batch_size=1,
            create_collection=True,
        )
        self.assertEqual(wrapper.client.inserted, rows)

    def test_single_thread_image_search_returns_doc_ids(self):
        wrapper = milvus_client(
            db_path="fake://milvus",
            collection_name="image",
            trace=None,
        )
        wrapper.client = FakeMilvusTransport()
        wrapper.client.exists = True
        doc_ids = wrapper.query_search_image(
            [[0.1, 0.2], [0.2, 0.1]],
            topk=2,
            collection_name="image",
        )
        self.assertEqual(doc_ids, {7})

    def test_milvus_image_rerank_computes_score(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "page.png"
            Image.new("RGB", (2, 2)).save(image_path)
            client = FakeImageClient(str(image_path))
            retriever = BaseRetriever("image", top_k=1, client=client)
            images = retriever.search_db_image(np.asarray([[1.0, 0.0], [0.0, 1.0]]))
        self.assertEqual(client.filter_expr, "doc_id in [7]")
        self.assertEqual(len(images), 1)


if __name__ == "__main__":
    unittest.main()
