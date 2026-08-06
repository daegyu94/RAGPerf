# Workload Scripts

These entrypoints accept a workload record count, print an estimate before embedding, and write the
verified artifact under `<output-dir>/artifact`. The estimate is based on the selected count and a
workload-specific row assumption; the final manifest is authoritative.

All commands run from the repository root with the project-local `.venv` activated.

## Text Embedding

`--record-count` is the number of Wikipedia source documents. The script downloads that many
Wikipedia documents and Natural Questions queries, then reports the post-chunk vector row estimate.

```bash
python vector_workload/scripts/text.py record \
  --record-count 100000 \
  --query-count 2000 \
  --output-dir /MNTPNT/ragperf/wikipedia-nq-100k \
  --device cuda:0 \
  --batch-size 32
```

For an existing JSONL corpus, pass `--corpus-file` and `--query-file` instead of `--dataset`.
`--record-count` and `--query-count` then act as optional limits.

```bash
python vector_workload/scripts/text.py replay \
  --artifact-dir /MNTPNT/ragperf/wikipedia-nq-100k/artifact \
  --collection ragperf_wikipedia_100k
```

## Audio ASR + Text Embedding

`--record-count` is the maximum number of audio files. Since audio duration changes transcript
length, the script estimates rows after ASR preparation and reports raw audio bytes separately.

```bash
python vector_workload/scripts/audio_asr.py record \
  --audio-dir /MNTPNT/ragperf/audio/files \
  --query-file /MNTPNT/ragperf/audio/queries.jsonl \
  --record-count 1000 \
  --output-dir /MNTPNT/ragperf/audio-1000 \
  --asr-device cuda:0 \
  --embedding-device cuda:0
```

```bash
python vector_workload/scripts/audio_asr.py replay \
  --artifact-dir /MNTPNT/ragperf/audio-1000/artifact \
  --collection ragperf_audio_1000
```

## ColPali PDF Image

`--record-count` is the maximum number of PDF files, not the number of pages or multi-vector rows.
Use `--estimated-vectors-per-page` to make the pre-record estimate explicit; the exported manifest
contains the actual page/token row counts.

```bash
python vector_workload/scripts/colpali.py record \
  --pdf-dir /MNTPNT/ragperf/arxiv/pdfs \
  --query-file /MNTPNT/ragperf/arxiv/queries.jsonl \
  --record-count 10000 \
  --output-dir /MNTPNT/ragperf/arxiv-colpali-10000 \
  --estimated-vectors-per-page 1000 \
  --device cuda:0
```

```bash
python vector_workload/scripts/colpali.py replay \
  --artifact-dir /MNTPNT/ragperf/arxiv-colpali-10000/artifact \
  --collection ragperf_colpali_10000
```

## Synthetic

For Synthetic, `--record-count` is exactly the corpus vector row count, so the size estimate is
deterministic for a given dimension and dtype.

```bash
python vector_workload/scripts/synthetic.py record \
  --record-count 1000000 \
  --query-count 2000 \
  --dimension 384 \
  --dtype float32 \
  --output-dir /MNTPNT/ragperf/synthetic-1m
```

```bash
python vector_workload/scripts/synthetic.py replay \
  --artifact-dir /MNTPNT/ragperf/synthetic-1m/artifact \
  --collection ragperf_synthetic_1m
```

Replay options such as `--uri`, `--concurrency`, `--warmup-queries` and `--respect-delay` are
available on every workload script. For sizing formulas and Milvus storage headroom, see the
[Record Guide](../docs/RECORD.md#estimate-dataset-and-storage-size).
