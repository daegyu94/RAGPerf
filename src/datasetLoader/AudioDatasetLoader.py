"""Streaming Hugging Face audio dataset loader for Audio RAG."""

import json

import pandas as pd
from datasets import config as datasets_config
from datasets import load_dataset

from datasetLoader.BaseDatasetLoader import BaseDatasetLoader


class AudioDatasetLoader(BaseDatasetLoader):
    """Load audio samples without materialising the complete dataset.

    ``iter_samples`` is the preferred API for record runs.  ``get_dataset_slice``
    is provided for compatibility with the existing RAGPerf loaders and only
    materialises the requested slice.
    """

    def __init__(
        self,
        dataset_name="openslr/librispeech_asr",
        dataset_config="clean",
        split="train.100",
        streaming=False,
        cache_dir=None,
    ):
        super().__init__(dataset_name=dataset_name)
        self.dataset_config = dataset_config
        self.split = split
        self.streaming = streaming
        try:
            self.dataset = load_dataset(
                dataset_name,
                dataset_config,
                split=split,
                streaming=streaming,
                cache_dir=cache_dir,
            )
        except ConnectionError as exc:
            if datasets_config.HF_DATASETS_OFFLINE:
                print(
                    "***Dataset autodownload disabled and no audio dataset was found "
                    f"under HF_CACHE_HOME: <{datasets_config.HF_CACHE_HOME}>"
                )
            raise exc
        self.total_length = None if streaming else len(self.dataset)

    @staticmethod
    def _metadata(sample):
        """Return JSON-safe metadata while excluding the audio payload itself."""
        metadata = {
            key: value
            for key, value in sample.items()
            if key not in {"audio", "text", "sentence"}
        }
        try:
            json.dumps(metadata)
        except (TypeError, ValueError):
            metadata = {key: str(value) for key, value in metadata.items()}
        return metadata

    def iter_samples(self, limit=None, offset=0):
        """Yield ``audio``, transcript and metadata dictionaries in order."""
        if offset:
            dataset = self.dataset.skip(offset)
        else:
            dataset = self.dataset
        for index, sample in enumerate(dataset):
            if limit is not None and index >= limit:
                break
            transcript = sample.get("text", sample.get("sentence", ""))
            yield {
                "audio": sample.get("audio"),
                "text": transcript or "",
                "metadata": self._metadata(sample),
            }

    def get_dataset_slice(self, length, offset):
        rows = list(self.iter_samples(limit=length, offset=offset * length))
        if not rows:
            raise ValueError(f"Audio slice is empty: offset={offset}, length={length}")
        return pd.DataFrame(rows, columns=["audio", "text", "metadata"])
