from __future__ import annotations

import argparse
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import pyarrow.parquet as pq

from vector_workload import export_vectors, prepare_workloads, replay_milvus
from vector_workload.tests.test_text_mixed_replay import (
    FakeEncoder,
    FakeMilvusClient,
    export_args,
)


def write_silent_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(b"\x00\x00" * 1_600)


class FakeTranscriber:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, path: str, **kwargs):
        self.calls.append((path, kwargs))
        return {"text": f"Transcript for {Path(path).stem}."}


class AudioAsrWorkloadTest(unittest.TestCase):
    def test_audio_asr_export_and_mixed_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / "audio"
            for index in range(4):
                write_silent_wav(audio_dir / f"set-{index % 2}" / f"clip-{index}.wav")
            (audio_dir / "ignored.txt").write_text("not audio", encoding="utf-8")
            query_file = root / "queries.jsonl"
            query_file.write_text(
                "".join(
                    json.dumps({"id": f"query-{index}", "text": f"question {index}"})
                    + "\n"
                    for index in range(2)
                ),
                encoding="utf-8",
            )

            prepared = root / "prepared"
            transcriber = FakeTranscriber()
            prepare_args = argparse.Namespace(
                audio_dir=audio_dir,
                query_file=query_file,
                output_dir=prepared,
                model="test-whisper",
                revision="test-revision",
                device="cpu",
                dtype="auto",
                language="en",
                batch_size=2,
                chunk_length_seconds=15,
                max_audio_files=None,
                audio_extensions=("wav",),
                dataset_name="test-audio",
            )
            prepared_result = prepare_workloads.prepare_audio_asr(
                prepare_args, transcriber
            )
            self.assertEqual(prepared_result["audio_documents"], 4)
            self.assertEqual(prepared_result["asr_dtype"], "float32")
            self.assertEqual(len(transcriber.calls), 4)
            self.assertEqual(
                transcriber.calls[0][1]["generate_kwargs"], {"language": "en"}
            )
            self.assertEqual(transcriber.calls[0][1]["chunk_length_s"], 15)

            corpus_rows = [
                json.loads(line)
                for line in (prepared / "corpus.jsonl").read_text().splitlines()
            ]
            self.assertEqual(corpus_rows[0]["metadata"]["modality"], "audio")
            self.assertEqual(
                corpus_rows[0]["metadata"]["representation"], "asr_transcript"
            )
            self.assertEqual(
                corpus_rows[0]["metadata"]["asr_revision"], "test-revision"
            )
            self.assertEqual(len(corpus_rows[0]["metadata"]["source_sha256"]), 64)

            artifact = root / "artifact"
            with mock.patch.object(
                export_vectors, "load_encoder", return_value=FakeEncoder()
            ):
                export_vectors.export_artifact(
                    export_args(
                        prepared / "corpus.jsonl",
                        prepared / "queries.jsonl",
                        artifact,
                    )
                )
            schedule = pq.read_table(artifact / "schedule-00000.parquet").to_pydict()
            self.assertEqual(
                schedule["operation"], ["search", "insert", "search", "insert"]
            )

            result_file = root / "result.json"
            replay_args = argparse.Namespace(
                artifact_dir=artifact,
                max_payload_length=65535,
                collection="audio-asr-mixed-test",
                result_file=result_file,
                uri="test",
                token="test",
                database="default",
                insert_batch_size=2,
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
            with mock.patch.object(replay_milvus, "MilvusClient", FakeMilvusClient):
                replay_milvus.replay(replay_args)
            replay_result = json.loads(result_file.read_text(encoding="utf-8"))
            self.assertEqual(replay_result["database"]["rows"], 4)
            self.assertEqual(replay_result["replay"]["queries"], 2)
            self.assertEqual(replay_result["replay"]["insert_events"], 2)
            self.assertEqual(replay_result["replay"]["inserted_rows"], 2)

    def test_cpu_float16_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "float16 ASR"):
            prepare_workloads.resolve_asr_dtype("cpu", "float16")

    def test_english_only_model_omits_language_override(self):
        transcriber = FakeTranscriber()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "sample.wav"
            write_silent_wav(audio)
            rows = list(
                prepare_workloads.audio_transcript_rows(
                    [audio],
                    root,
                    transcriber,
                    "openai/whisper-tiny.en",
                    None,
                    "en",
                    30,
                    1,
                    "test-audio",
                )
            )
            self.assertEqual(len(rows), 1)
            self.assertNotIn("generate_kwargs", transcriber.calls[-1][1])

            with self.assertRaisesRegex(ValueError, "English-only ASR model"):
                list(
                    prepare_workloads.audio_transcript_rows(
                        [audio],
                        root,
                        transcriber,
                        "openai/whisper-tiny.en",
                        None,
                        "fr",
                        30,
                        1,
                        "test-audio",
                    )
                )


if __name__ == "__main__":
    unittest.main()
