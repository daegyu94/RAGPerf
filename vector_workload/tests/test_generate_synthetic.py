from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from vector_workload import generate_synthetic
from vector_workload.artifact_utils import verify_artifact


def generation_args(output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=output_dir,
        corpus_count=7,
        query_count=4,
        dimension=8,
        dtype="float32",
        rows_per_shard=3,
        query_noise=0.0,
        query_delay_ms=2.5,
        seed=42,
    )


class GenerateSyntheticTest(unittest.TestCase):
    def test_generates_reproducible_expected_neighbors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first"
            second = root / "second"

            generate_synthetic.generate(generation_args(first))
            generate_synthetic.generate(generation_args(second))
            verify_artifact(first)

            with (first / "workload-manifest.yaml").open(encoding="utf-8") as stream:
                manifest = yaml.safe_load(stream)
            self.assertEqual(manifest["producer"], "vector_workload/generate_synthetic.py")
            self.assertEqual(manifest["embedding"]["vector_layout"], "single_vector")
            self.assertEqual(manifest["workload"]["operation"], "search")

            first_corpus = pq.read_table(first / "corpus-00000.parquet").to_pydict()
            second_corpus = pq.read_table(second / "corpus-00000.parquet").to_pydict()
            np.testing.assert_array_equal(first_corpus["vector"], second_corpus["vector"])

            corpus_vectors: dict[str, np.ndarray] = {}
            for path in sorted(first.glob("corpus-*.parquet")):
                corpus = pq.read_table(path, columns=["id", "vector"]).to_pydict()
                corpus_vectors.update(
                    {
                        record_id: np.asarray(vector, dtype=np.float32)
                        for record_id, vector in zip(
                            corpus["id"], corpus["vector"], strict=True
                        )
                    }
                )

            expected_ids: list[str] = []
            delays: list[float] = []
            query_vectors: list[np.ndarray] = []
            for path in sorted(first.glob("queries-*.parquet")):
                queries = pq.read_table(path).to_pydict()
                expected_ids.extend(
                    json.loads(metadata)["expected_id"]
                    for metadata in queries["metadata_json"]
                )
                delays.extend(queries["delay_ms"])
                query_vectors.extend(
                    np.asarray(vector, dtype=np.float32) for vector in queries["vector"]
                )

            self.assertEqual(len(expected_ids), 4)
            self.assertTrue(set(expected_ids).issubset(corpus_vectors))
            self.assertEqual(delays, [2.5] * 4)
            for query_vector, expected_id in zip(
                query_vectors, expected_ids, strict=True
            ):
                np.testing.assert_allclose(
                    query_vector, corpus_vectors[expected_id], rtol=1e-6, atol=1e-7
                )
                top1_id = max(
                    corpus_vectors,
                    key=lambda corpus_id: float(
                        np.dot(query_vector, corpus_vectors[corpus_id])
                    ),
                )
                self.assertEqual(top1_id, expected_id)


if __name__ == "__main__":
    unittest.main()
