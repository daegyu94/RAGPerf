import unittest

from datasetLoader.AudioDatasetLoader import AudioDatasetLoader
from encoder.AudioEncoder import AudioEncoder
from RAGPipeline.AudioRAGPipeline import AudioRAGPipeline


class FakeEncoder:
    def __init__(self):
        self.loaded = 0
        self.freed = 0

    def load_encoder(self):
        self.loaded += 1

    def embedding_with_text(self, audios):
        return [[float(index), 1.0] for index, _ in enumerate(audios)], ["audio"] * len(
            audios
        )

    def free_encoder(self):
        self.freed += 1


class FakeClient:
    def __init__(self):
        self.markers = 0
        self.searches = []

    def begin_timed_workload(self):
        self.markers += 1

    def query_search(self, vectors, top_k, **kwargs):
        self.searches.append((vectors, top_k, kwargs))
        return []


class FakeSentenceEncoder:
    def encode(self, texts, **kwargs):
        return [[1.0, 0.0, 0.5] for _ in texts]


class AudioWorkloadTest(unittest.TestCase):
    def test_audio_pipeline_is_batched_and_marks_workload(self):
        encoder = FakeEncoder()
        client = FakeClient()
        pipeline = AudioRAGPipeline(
            encoder=encoder,
            client=client,
            collection_name="audio",
            top_k=3,
            retrieval_batch_size=2,
        )
        self.assertEqual(pipeline.retrieve(({"array": [0.0]} for _ in range(5))), 5)
        self.assertEqual([len(item[0]) for item in client.searches], [2, 2, 1])
        self.assertEqual(client.markers, 1)
        self.assertEqual(encoder.loaded, 1)
        self.assertEqual(encoder.freed, 1)

    def test_audio_encoder_combines_asr_and_embedding(self):
        encoder = AudioEncoder(device="cpu")
        encoder.asr = lambda _audio: {"text": "synthetic transcript"}
        encoder.encoder = FakeSentenceEncoder()
        vectors, texts = encoder.embedding_with_text(
            [{"array": [0.0], "sampling_rate": 16000}] * 2
        )
        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), 3)
        self.assertEqual(texts, ["synthetic transcript"] * 2)

    def test_audio_input_preserves_sampling_rate(self):
        value = AudioEncoder._asr_input({"array": [0, 1], "sampling_rate": 8000})
        self.assertEqual(value["sampling_rate"], 8000)
        self.assertEqual(value["raw"].dtype.name, "float32")

    def test_audio_metadata_excludes_payload(self):
        metadata = AudioDatasetLoader._metadata(
            {"audio": {"array": [1]}, "text": "hello", "speaker_id": 7}
        )
        self.assertEqual(metadata, {"speaker_id": 7})


if __name__ == "__main__":
    unittest.main()
