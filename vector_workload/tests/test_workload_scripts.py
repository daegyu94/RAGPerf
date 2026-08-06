import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from vector_workload.scripts import common, text


class WorkloadScriptTests(unittest.TestCase):
    def test_count_chunks_matches_character_window(self):
        self.assertEqual(common.count_chunks("a" * 512, 512, 0), 1)
        self.assertEqual(common.count_chunks("a" * 513, 512, 0), 2)
        self.assertEqual(common.count_chunks("a" * 513, 256, 32), 3)

    def test_jsonl_text_stats_counts_documents_and_chunks(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corpus.jsonl"
            path.write_text(
                json.dumps({"id": "a", "text": "a" * 513}) + "\n"
                + json.dumps({"id": "b", "text": "short"}) + "\n",
                encoding="utf-8",
            )
            stats = common.jsonl_text_stats(path, 512, 0)
        self.assertEqual(stats["documents"], 2)
        self.assertEqual(stats["chunks"], 3)

    def test_text_parser_accepts_count_based_dataset_record(self):
        args = text.build_parser().parse_args(
            [
                "record",
                "--record-count",
                "1000",
                "--query-count",
                "100",
                "--output-dir",
                "/tmp/text-run",
                "--device",
                "cpu",
            ]
        )
        self.assertEqual(args.record_count, 1000)
        self.assertEqual(args.query_count, 100)
        self.assertEqual(args.func, text.record)

    def test_print_estimate_is_human_readable(self):
        output = io.StringIO()
        with redirect_stdout(output):
            common.print_estimate(
                "synthetic",
                corpus_records=100,
                corpus_rows=100,
                query_records=10,
                query_rows=10,
                dimension=384,
                dtype="float32",
                input_bytes=0,
                overhead=1.25,
                assumptions=["test"],
            )
        self.assertIn('"estimated_vector_rows"', output.getvalue())
        self.assertIn("165.00 KiB", output.getvalue())


if __name__ == "__main__":
    unittest.main()
