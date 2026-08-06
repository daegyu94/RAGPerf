from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from vector_workload import record_workload, replay_workload


def option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


class RecordWorkloadTest(unittest.TestCase):
    def test_text_records_and_verifies_standard_artifact_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = record_workload.build_parser().parse_args(
                [
                    "text",
                    "--corpus-file",
                    str(root / "corpus.jsonl"),
                    "--query-file",
                    str(root / "queries.jsonl"),
                    "--output-dir",
                    str(root / "run"),
                    "--smoke",
                    "--device",
                    "cpu",
                    "--initial-corpus-ratio",
                    "0.5",
                ]
            )
            commands: list[list[str]] = []

            artifact_dir = record_workload.record_text(args, commands.append)

            self.assertEqual(artifact_dir, (root / "run" / "artifact").resolve())
            self.assertEqual(len(commands), 2)
            self.assertIn("--smoke", commands[0])
            self.assertEqual(option_value(commands[0], "--device"), "cpu")
            self.assertEqual(option_value(commands[0], "--initial-corpus-ratio"), "0.5")
            self.assertEqual(option_value(commands[1], "--artifact-dir"), str(artifact_dir))

    def test_audio_asr_prepares_exports_and_verifies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = record_workload.build_parser().parse_args(
                [
                    "audio-asr",
                    "--audio-dir",
                    str(root / "audio"),
                    "--query-file",
                    str(root / "queries.jsonl"),
                    "--output-dir",
                    str(root / "run"),
                    "--asr-device",
                    "cpu",
                    "--embedding-device",
                    "cpu",
                ]
            )
            commands: list[list[str]] = []

            artifact_dir = record_workload.record_audio_asr(args, commands.append)

            self.assertEqual(len(commands), 3)
            self.assertIn("audio-asr", commands[0])
            self.assertEqual(
                option_value(commands[1], "--corpus-file"),
                str((root / "run" / "input" / "corpus.jsonl").resolve()),
            )
            self.assertEqual(option_value(commands[2], "--artifact-dir"), str(artifact_dir))

    def test_colpali_prepares_exports_and_verifies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = record_workload.build_parser().parse_args(
                [
                    "colpali",
                    "--pdf-dir",
                    str(root / "pdfs"),
                    "--query-file",
                    str(root / "queries.jsonl"),
                    "--output-dir",
                    str(root / "run"),
                    "--max-pdfs",
                    "10",
                ]
            )
            commands: list[list[str]] = []

            artifact_dir = record_workload.record_colpali(args, commands.append)

            self.assertEqual(len(commands), 3)
            self.assertIn("arxiv-pdf-image", commands[0])
            self.assertEqual(option_value(commands[0], "--max-pdfs"), "10")
            self.assertTrue(commands[1][1].endswith("export_colpali.py"))
            self.assertEqual(option_value(commands[2], "--artifact-dir"), str(artifact_dir))

    def test_record_rejects_nonempty_run_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            (run_dir / "existing").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                record_workload.prepare_run_dir(run_dir)


class ReplayWorkloadTest(unittest.TestCase):
    def write_manifest(self, artifact_dir: Path, layout: str) -> None:
        artifact_dir.mkdir()
        with (artifact_dir / "workload-manifest.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump({"embedding": {"vector_layout": layout}}, stream)

    def test_single_vector_uses_cosine(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact_dir = Path(temporary) / "artifact"
            self.write_manifest(artifact_dir, "single_vector")
            args = replay_workload.build_parser().parse_args(
                ["--artifact-dir", str(artifact_dir), "--collection", "text_run"]
            )

            _, replay, layout, result_file = replay_workload.build_commands(args)

            self.assertEqual(layout, "single_vector")
            self.assertEqual(option_value(replay, "--metric"), "COSINE")
            self.assertEqual(result_file, Path(temporary) / "replay-result.json")

    def test_multi_vector_uses_ip_and_token_top_k(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact_dir = Path(temporary) / "artifact"
            self.write_manifest(artifact_dir, "multi_vector")
            args = replay_workload.build_parser().parse_args(
                ["--artifact-dir", str(artifact_dir), "--collection", "colpali_run"]
            )

            _, replay, layout, _ = replay_workload.build_commands(args)

            self.assertEqual(layout, "multi_vector")
            self.assertEqual(option_value(replay, "--metric"), "IP")
            self.assertEqual(option_value(replay, "--token-top-k"), "100")


if __name__ == "__main__":
    unittest.main()
