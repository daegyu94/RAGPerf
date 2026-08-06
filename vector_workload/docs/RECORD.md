# Record a Vector Workload

**Record**는 corpus와 query를 embedding하고 실행 순서를 정의해, 다른 server에서 동일하게
replay할 수 있는 portable artifact로 기록하는 과정이다.

```text
corpus.jsonl + queries.jsonl
              ↓
       embedding + schedule
              ↓
Parquet shards + manifest + checksums
```

Artifact 구조와 입력 schema의 기준은 [Artifact Format](ARTIFACT_FORMAT.md)에 있다. Wikipedia,
arXiv, Audio ASR 같은 dataset별 input 준비 방법은 [Workflow Guide](WORKFLOWS.md)를 참고한다.

## Environment

모든 명령은 repository root에서 project-local environment를 활성화한 뒤 실행한다.

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r vector_workload/requirements.txt
```

기본 text embedding model은 `BAAI/bge-m3`, device는 `cuda:0`이다. CPU에서 작은 경로만
확인하려면 `--smoke --device cpu`를 사용한다.

## Record Text Corpus and Queries

UTF-8 JSONL corpus와 query를 artifact로 export한다. Input과 output path만 필수이며 나머지는
기본값을 사용할 수 있다.

```bash
RUN_DIR=/MNTPNT/ragperf/workload-001

python vector_workload/export_vectors.py export \
  --corpus-file /DATASET/corpus.jsonl \
  --query-file /DATASET/queries.jsonl \
  --output-dir "$RUN_DIR/artifact"
```

Exporter는 비어 있지 않은 output directory를 덮어쓰지 않는다. Model, revision, chunking,
normalization, dtype과 seed는 manifest에 기록된다. 여러 target을 비교할 때는 artifact를 target마다
다시 만들지 말고 같은 artifact를 복사해 사용한다.

## Define Search and Insert Order

기본 `--initial-corpus-ratio 1.0`은 corpus 전체를 index build 전에 적재하는 search-only
workload다. Search와 insert가 섞인 workload를 기록하려면 initial ratio를 줄이고 schedule
간격과 insert event 크기를 지정한다.

```bash
python vector_workload/export_vectors.py export \
  --corpus-file /DATASET/corpus.jsonl \
  --query-file /DATASET/queries.jsonl \
  --output-dir "$RUN_DIR/artifact" \
  --initial-corpus-ratio 0.8 \
  --searches-per-insert 10 \
  --insert-event-size 100
```

- `--initial-corpus-ratio`: index build 전에 적재할 corpus 비율
- `--searches-per-insert`: scheduled insert 사이의 search event 수
- `--insert-event-size`: 한 insert event가 추가할 최대 vector row 수
- `--rows-per-shard`: 큰 artifact의 Parquet shard당 최대 row 수

Query JSONL의 `delay_ms`도 artifact에 보존된다. Replay에서 `--respect-delay`를 지정한 경우에만
적용되며, 지정하지 않으면 가능한 빠르게 요청한다.

## Verify the Artifact

전송하거나 replay하기 전에 manifest, row count와 checksum을 검증한다.

```bash
python vector_workload/export_vectors.py verify \
  --artifact-dir "$RUN_DIR/artifact"
```

검증된 artifact directory 전체를 target server로 복사한다. 일부 Parquet shard나
`workload-manifest.yaml`, `SHA256SUMS`만 따로 복사하면 검증에 실패한다.

## Other Artifact Producers

- ColPali PDF image: `export_colpali.py`가 multi-vector artifact를 기록한다.
- Synthetic baseline: `generate_synthetic.py`가 embedding model 없이 deterministic artifact를
  기록한다.
- Audio ASR: `prepare_workloads.py audio-asr`로 transcript JSONL을 만든 뒤 text exporter를
  사용한다.

각 command의 dataset별 예시는 [Workflow Guide](WORKFLOWS.md)와
[Audio ASR Workflow](AUDIO_ASR.md)에 있다. 기록을 마쳤으면 [Replay Guide](REPLAY.md)에 따라
artifact를 Milvus에서 실행한다.
