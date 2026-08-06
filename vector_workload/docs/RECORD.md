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

## Record by Workload Type

`record_workload.py`는 input preparation, embedding export와 artifact 검증을 순서대로 실행한다.
모든 workload는 `<output-dir>/artifact`에 같은 형식으로 기록된다. `--output-dir`은 비어 있거나
존재하지 않는 새 run directory여야 한다.

### Text Embedding

UTF-8 JSONL corpus와 query를 single-vector artifact로 기록한다.

```bash
python vector_workload/record_workload.py text \
  --corpus-file /DATASET/corpus.jsonl \
  --query-file /DATASET/queries.jsonl \
  --output-dir /MNTPNT/ragperf/text-001
```

### Audio ASR + Text Embedding

Local audio를 ASR transcript로 변환한 뒤 transcript와 query를 single-vector artifact로 기록한다.

```bash
python vector_workload/record_workload.py audio-asr \
  --audio-dir /DATASET/audio \
  --query-file /DATASET/queries.jsonl \
  --output-dir /MNTPNT/ragperf/audio-001
```

기본 ASR model은 `openai/whisper-small`, embedding model은 `BAAI/bge-m3`이며 둘 다 `cuda:0`을
사용한다. 다른 device가 필요하면 `--asr-device`와 `--embedding-device`를 각각 지정한다.
지원 audio 형식과 transcript metadata는 [Audio ASR Workflow](AUDIO_ASR.md)에 있다.

### ColPali PDF Image

PDF page를 image로 render하고 ColPali multi-vector artifact로 기록한다.

```bash
python vector_workload/record_workload.py colpali \
  --pdf-dir /DATASET/pdfs \
  --query-file /DATASET/queries.jsonl \
  --output-dir /MNTPNT/ragperf/colpali-001
```

기본 model은 `vidore/colpali-v1.2`, device는 `cuda:0`이다. PDF 수를 제한하려면
`--max-pdfs`를 지정한다.

각 command는 성공 전에 artifact checksum과 Parquet shard를 검증하고 다음 경로를 출력한다.

```text
Artifact ready: /MNTPNT/ragperf/<run>/artifact
```

Model, revision과 batch size 같은 선택 항목은 workload별 `--help`에서 확인한다.

```bash
python vector_workload/record_workload.py text --help
python vector_workload/record_workload.py audio-asr --help
python vector_workload/record_workload.py colpali --help
```

## Define Search and Insert Order

기본 `--initial-corpus-ratio 1.0`은 corpus 전체를 index build 전에 적재하는 search-only
workload다. Search와 insert가 섞인 workload를 기록하려면 initial ratio를 줄이고 schedule
간격과 insert event 크기를 지정한다.

```bash
python vector_workload/record_workload.py text \
  --corpus-file /DATASET/corpus.jsonl \
  --query-file /DATASET/queries.jsonl \
  --output-dir /MNTPNT/ragperf/text-mixed-001 \
  --initial-corpus-ratio 0.8 \
  --searches-per-insert 10 \
  --insert-event-size 100
```

- `--initial-corpus-ratio`: index build 전에 적재할 corpus 비율
- `--searches-per-insert`: scheduled insert 사이의 search event 수
- `--insert-event-size`: 한 insert event가 추가할 최대 vector row 수

Query JSONL의 `delay_ms`도 artifact에 보존된다. Replay에서 `--respect-delay`를 지정한 경우에만
적용되며, 지정하지 않으면 가능한 빠르게 요청한다.

## Artifact Validation and Reuse

상위 Record command는 검증까지 자동으로 수행한다. 전송 후 artifact를 별도로 다시 확인하려면
다음 명령을 사용한다.

```bash
python vector_workload/export_vectors.py verify \
  --artifact-dir /MNTPNT/ragperf/text-001/artifact
```

검증된 artifact directory 전체를 target server로 복사한다. 일부 Parquet shard나
`workload-manifest.yaml`, `SHA256SUMS`만 따로 복사하면 검증에 실패한다.

Exporter는 비어 있지 않은 output directory를 덮어쓰지 않는다. Model, revision, chunking,
normalization, dtype과 seed는 manifest에 기록된다. 여러 target을 비교할 때는 artifact를 target마다
다시 만들지 말고 같은 artifact를 복사해 사용한다.

## Synthetic Baseline

`generate_synthetic.py`는 embedding model 없이 deterministic artifact를 기록한다. Synthetic
workload의 크기와 dimension을 직접 지정하는 예시는 [Workflow Guide](WORKFLOWS.md)에 있다.

상위 CLI가 제공하지 않는 세부 option이나 dataset별 preparation은 [Workflow Guide](WORKFLOWS.md)에
있다. 기록을 마쳤으면 [Replay Guide](REPLAY.md)에 따라 artifact를 Milvus에서 실행한다.
