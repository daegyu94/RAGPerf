"""Streaming Audio RAG retrieval pipeline."""

from itertools import islice


class AudioRAGPipeline:
    """Embed audio queries and submit Milvus searches in bounded batches.

    The pipeline deliberately does not retain search results.  A caller that
    needs to consume them can provide a ``result_sink`` callback.
    """

    def __init__(
        self,
        encoder,
        client,
        collection_name,
        top_k=5,
        retrieval_batch_size=1,
        result_sink=None,
    ):
        self.encoder = encoder
        self.client = client
        self.collection_name = collection_name
        self.top_k = top_k
        self.retrieval_batch_size = retrieval_batch_size
        self.result_sink = result_sink

    @staticmethod
    def _batches(items, batch_size):
        iterator = iter(items)
        while True:
            batch = list(islice(iterator, batch_size))
            if not batch:
                return
            yield batch

    def retrieve(self, audios, batch_size=None):
        """Replay audio queries and return the number of submitted queries."""
        batch_size = batch_size or self.retrieval_batch_size
        self.encoder.load_encoder()
        self.client.begin_timed_workload()
        submitted = 0
        try:
            for audio_batch in self._batches(audios, batch_size):
                vectors, _transcripts = self.encoder.embedding_with_text(audio_batch)
                results = self.client.query_search(
                    vectors,
                    self.top_k,
                    collection_name=self.collection_name,
                    search_batch_size=batch_size,
                    multithread=False,
                    consistency_level="Eventually",
                )
                submitted += len(audio_batch)
                if self.result_sink is not None:
                    self.result_sink(results)
        finally:
            self.encoder.free_encoder()
        return submitted

    process = retrieve
