from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vector_workload import export_vectors, prepare_workloads
from vector_workload.tests.test_text_mixed_replay import FakeEncoder, export_args


class FakePage:
    def __init__(self, text: str):
        self.text = text

    def extract_text(self):
        return self.text


class FakeReader:
    def __init__(self, path: Path):
        self.pages = [FakePage(f"Extracted text from {path.stem}."), FakePage("Second page.")]


class ArxivPdfTextTest(unittest.TestCase):
    def test_prepare_export_and_mixed_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_dir = root / "pdfs"
            pdf_dir.mkdir()
            (pdf_dir / "paper-001.pdf").write_bytes(b"test fixture")
            (pdf_dir / "paper-002.pdf").write_bytes(b"test fixture")
            query_file = root / "input-queries.jsonl"
            query_file.write_text(
                json.dumps({"id": "query-1", "text": "What is in the paper?"}) + "\n",
                encoding="utf-8",
            )
            prepared = root / "prepared"
            prepare_args = argparse.Namespace(
                pdf_dir=pdf_dir,
                query_file=query_file,
                output_dir=prepared,
                max_pdfs=None,
            )
            result = prepare_workloads.prepare_arxiv_pdf_text(prepare_args, FakeReader)
            self.assertEqual(result["pdf_documents"], 2)
            corpus_rows = [
                json.loads(line)
                for line in (prepared / "corpus.jsonl").read_text().splitlines()
            ]
            self.assertEqual(corpus_rows[0]["metadata"]["pages"], 2)
            self.assertEqual(corpus_rows[0]["metadata"]["modality"], "text")

            artifact = root / "artifact"
            args = export_args(
                prepared / "corpus.jsonl", prepared / "queries.jsonl", artifact
            )
            with mock.patch.object(export_vectors, "load_encoder", return_value=FakeEncoder()):
                export_vectors.export_artifact(args)
            manifest = (artifact / "workload-manifest.yaml").read_text(encoding="utf-8")
            self.assertIn("operation: search_insert_mixed", manifest)


if __name__ == "__main__":
    unittest.main()
