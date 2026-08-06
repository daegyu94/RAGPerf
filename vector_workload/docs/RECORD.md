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
arXiv, Audio ASR 같은 dataset별 input 준비 방법은 [Dataset Workflows](DATASET_WORKFLOWS.md)를 참고한다.

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
Audio input 규칙과 transcript metadata는 [Dataset Workflows](DATASET_WORKFLOWS.md)에 있다.

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

## Estimate Dataset and Storage Size

`--corpus-count`는 원본 document 수이며, 실제 vector record 수가 아니다. Text workload는
document를 chunk로 나누므로 다음 값을 구분해야 한다.

```text
vector rows ≈ corpus documents × average chunks per document
raw vector bytes ≈ vector rows × embedding dimension × bytes per value
```

실제 값은 record가 끝난 뒤 `workload-manifest.yaml`의 `inputs.corpus.documents`와
`inputs.corpus.chunks`, Parquet artifact의 `rows`에서 확인한다. 먼저 1,000~10,000개 document로
pilot을 기록한 뒤 다음 비율로 확대하는 것이 안전하다.

이 repository의 Wikipedia sample에서는 100 documents가 6,380 chunks로 기록되어 평균
63.8 chunks/document였다. 이 비율이 유지되고 BGE-M3의 1,024차원 `float32` vector를 사용한다고
가정하면 대략 다음과 같다. Dataset의 document 길이 분포에 따라 실제 값은 달라진다.

| 원본 documents | 예상 vector rows | raw vectors | 예상 artifact 규모 |
| ---: | ---: | ---: | ---: |
| 100,000 | 약 6.4M | 약 26 GB | 약 30 GB 이상 |
| 1,000,000 | 약 64M | 약 262 GB | 약 0.3 TB 이상 |
| 3,000,000~4,000,000 | 약 190M~255M | 약 0.8~1.0 TB | 약 1 TB 이상 |

여기서 artifact 규모에는 vector 외에 text, metadata와 Parquet overhead가 포함된다. Milvus
volume을 1TB 수준으로 맞추려는 경우에는 DISKANN index build용 임시 공간과 segment metadata를
위해 artifact 자체를 1TB까지 채우지 말고 최소 20~30%의 여유 공간을 둔다.

`--chunk-size 256 --chunk-overlap 32`처럼 chunk step을 줄이면 같은 text에서 생성되는 row가
기본 `512 / 0`보다 약 2.3배 늘어날 수 있다. 이 경우 위 표의 document 수를 그대로 적용하지
말고 pilot의 실제 `chunks` 값을 기준으로 다시 계산한다. 1TB 실험은 먼저 10,000 documents를
record해 `du -sh`와 manifest의 chunk 수를 확인한 뒤 확대한다.

### Workload-specific Record Counts

Workload마다 `record`가 의미하는 단위가 다르다.

| Workload | 원본 단위 | 실제 용량 산정 단위 |
| --- | --- | --- |
| Text embedding | document | text chunk vector row |
| Audio ASR + text embedding | audio file | ASR transcript chunk vector row |
| ColPali PDF image | PDF page | page의 multi-vector token row |
| Synthetic | vector row | `--corpus-count` 자체 |

#### Audio ASR

Audio file 하나가 먼저 하나의 transcript document가 되고, 그 transcript가 Text workload와 같은
chunking과 embedding 단계를 거친다. 따라서 `--max-audio-files`는 vector row 수가 아니며,
다음과 같이 계산한다.

```text
audio vector rows ≈ audio files × average transcript chunks per file
```

`--chunk-length-seconds`는 ASR 처리 window와 record 시간에 영향을 주지만 artifact row 수를
직접 결정하지는 않는다. 원본 audio는 artifact에 복사되지 않으므로 raw audio 저장 공간은
별도로 계산해야 한다. 최종 artifact의 `inputs.corpus.chunks`와 `artifacts[].rows`를 기준으로
1TB 목표를 조정한다. BGE-M3를 사용한다면 Text와 같은 1024차원 vector 계산을 적용할 수 있다.

#### ColPali PDF Image

ColPali는 PDF를 page image로 만들고, 한 page에서 여러 token vector를 생성한다. 따라서 PDF 수나
page 수만으로는 artifact 크기를 예측할 수 없다.

```text
ColPali vector rows = sum(token vectors per page)
raw vector bytes ≈ vector rows × ColPali dimension × bytes per value
```

`workload.initial_vector_rows`, `workload.scheduled_insert_vector_rows`와 query shard의
artifact `rows`가 기준 값이다. Export command의 summary에는 `query_vector_rows`도 출력된다.
`--insert-event-size`의 단위는 page document 수이지 multi-vector row 수가 아니다. 예를 들어
dimension이 128이고 page당 1,000 token vector가
생성되면 page 하나의 raw vector만 약 0.5 MB이므로, 1TB raw vector에는 약 200만 page가
필요하다. 실제 token 수와 dimension은 model/revision에 따라 달라지므로 먼저 수백~수천 page를
record해 manifest의 실제 row 수를 측정해야 한다. Render된 PNG page도 record 단계에서는 별도
저장 공간을 사용한다.

#### Synthetic

Synthetic workload는 chunking이나 model inference를 하지 않으므로 `--corpus-count`가 corpus
vector row 수와 같다.

```text
raw vector bytes = corpus-count × dimension × bytes per value
```

기본 dimension 384, `float32`에서는 다음과 같다.

| `--corpus-count` | raw vectors |
| ---: | ---: |
| 100,000 | 약 154 MB |
| 1,000,000 | 약 1.5 GB |
| 약 650,000,000 | 약 1 TB |

Synthetic은 크기 예측이 가장 쉽지만, 1TB 수준에서는 Parquet metadata, Milvus index와 운영
여유 공간을 포함해 실제 target volume을 더 크게 잡아야 한다. `float16`을 사용하면 vector
부분은 대략 절반으로 줄어든다.

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
workload의 크기와 dimension을 직접 지정하는 예시는 [Dataset Workflows](DATASET_WORKFLOWS.md)에 있다.

상위 CLI가 제공하지 않는 세부 option이나 dataset별 preparation은 [Dataset Workflows](DATASET_WORKFLOWS.md)에
있다. 기록을 마쳤으면 [Replay Guide](REPLAY.md)에 따라 artifact를 Milvus에서 실행한다.
