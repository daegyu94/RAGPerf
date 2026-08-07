"""ASR plus sentence-embedding encoder used by the Audio RAG workload."""

import gc

import numpy as np

from encoder.BaseEncoder import BaseEncoder


class AudioEncoder(BaseEncoder):
    """Transcribe audio and embed the transcript.

    Heavy dependencies and models are loaded only by ``load_encoder`` so that
    importing RAGPerf and running artifact replay does not require Whisper.
    """

    def __init__(
        self,
        asr_model="openai/whisper-tiny",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        batch_size=8,
    ):
        self.asr_model = asr_model
        self.embedding_model = embedding_model
        self.device = device
        self.batch_size = batch_size
        self.asr = None
        self.encoder = None
        self.dim = None

    def load_encoder(self):
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import pipeline

        if self.asr is not None and self.encoder is not None:
            return
        if str(self.device).startswith("cuda"):
            device_name = str(self.device)
            device = int(device_name.split(":", 1)[1]) if ":" in device_name else 0
        else:
            device = -1
        self.asr = pipeline(
            "automatic-speech-recognition",
            model=self.asr_model,
            device=device,
        )
        encoder_kwargs = {}
        if device >= 0:
            encoder_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
        self.encoder = SentenceTransformer(
            self.embedding_model, device=self.device, **encoder_kwargs
        )
        self.dim = self.encoder.get_sentence_embedding_dimension()

    @staticmethod
    def _asr_input(audio):
        """Convert a datasets.Audio value to a transformers pipeline input."""
        if isinstance(audio, dict) and "array" in audio:
            return {
                "raw": np.asarray(audio["array"], dtype=np.float32),
                "sampling_rate": int(audio.get("sampling_rate", 16000)),
            }
        return audio

    def transcribe(self, audios):
        if self.asr is None:
            raise RuntimeError("AudioEncoder.load_encoder() must be called first")
        transcripts = []
        for audio in audios:
            result = self.asr(self._asr_input(audio))
            transcripts.append(
                result.get("text", "") if isinstance(result, dict) else str(result)
            )
        return transcripts

    def embedding_with_text(self, audios):
        if self.encoder is None:
            raise RuntimeError("AudioEncoder.load_encoder() must be called first")
        transcripts = self.transcribe(audios)
        vectors = self.encoder.encode(
            transcripts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32).tolist(), transcripts

    def embedding(self, audios):
        vectors, _ = self.embedding_with_text(audios)
        return vectors

    def multi_gpus_embedding(self, audios):
        return self.embedding(audios)

    def free_encoder(self):
        self.asr = None
        self.encoder = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except (ImportError, RuntimeError):
            pass
