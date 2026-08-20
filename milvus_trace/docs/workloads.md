# Workload별 기록과 재생 예시

이 문서는 repository root에서 명령을 실행한다고 가정합니다. record에는 RAGPerf의
dataset, embedding/ASR, generation 의존성이 필요하고 replay에는 Milvus client만
필요합니다. 0.5TB, 1TB처럼 용량을 통제한 synthetic DISKANN dataset은 상세 내용을
여기에 중복하지 않고 [Vector workload 기록](vector_workloads.md)에서 설명합니다.

먼저 [Milvus trace 설치](../README.md#1-host별-설치와-실행)와 Milvus endpoint 준비를
완료합니다. 처음 테스트에서는 하나의 서버를 record와 replay에 함께 사용하고, source
collection과 target collection만 서로 다르게 지정하면 됩니다.

## Workload별 요구 사항

| Workload | Record host의 주요 요구 사항 | Replay host에서 불필요한 항목 |
| --- | --- | --- |
| Text | Sentence Transformers, vLLM 지원 GPU, Wikipedia/Natural Questions download | Dataset, embedding/generation model, GPU |
| Image | Poppler, ColPali, vision LLM 지원 GPU, ArXiv PDF download | PDF, Poppler, ColPali, vision LLM, GPU |
| Audio | `torchcodec`, Whisper, Sentence Transformers. 작은 run은 CPU 가능 | Audio file/decoder, Whisper, embedding model, GPU |

모든 record 명령은 monitoring system을 시작하므로 `src/monitoring_sys/libmsys*.so`와
`--msys-config`가 필요합니다. Python replay에는 이 monitoring module이 필요하지
않습니다.

Audio smoke config는 generation/evaluate를 끄므로 vLLM 없이 실행할 수 있습니다. Text/Image
generation 또는 evaluator를 켜는 config는 해당 vLLM 의존성을 추가로 설치해야 합니다.

## 공통 준비

source Milvus는 record 시 RAGPerf가 사용하는 endpoint이고, target Milvus는 artifact를
재생할 endpoint입니다. 두 endpoint는 같아도 되지만 replay collection은 존재하지 않는
새 이름이어야 합니다.

처음 실행할 때는 다음처럼 같은 Milvus endpoint를 두 변수에 지정합니다.

```bash
export MNTPNT=/path/to/ragperf-data
export MILVUS_URI=http://localhost:19530
export REPLAY_MILVUS_URI="$MILVUS_URI"
export RAG_DEVICE=cuda:0
export GENERATION_DEVICE=cuda:1
export MSYS_CONFIG=config/monitor/example_config.yaml
export MILVUS_TOKEN=root:Milvus
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$MNTPNT/artifacts" "$MNTPNT/results"
```

Workload YAML의 환경변수는 `MNTPNT`, `MILVUS_URI`, `RAG_DEVICE`, `GENERATION_DEVICE`에서
확장됩니다. `MSYS_CONFIG`는 YAML 값이 아니라 `--msys-config`에 전달하는 monitoring
config 경로입니다.

서로 다른 Milvus 환경의 성능을 비교할 때만 두 URI를 각각 다른 서버로 바꿉니다.

GPU가 하나라면 두 device 변수를 같은 값으로 지정할 수 있습니다. 모델이 CPU를
지원하는 단계는 `RAG_DEVICE=cpu`로 실행할 수 있지만, VLLM/멀티모달 generation은
GPU가 필요할 수 있습니다.

모든 예시 config는 corpus 준비, index 생성, timed query를 한 process에서 수행합니다.
따라서 index 이전 insert는 bootstrap corpus가 되고, model load 이후 search/query는
원래 도착 시각을 가진 event가 됩니다.

Record 전에 config의 `sys.vector_db.collection_name`을 이번 run만의 source collection
이름으로 바꿉니다. 예시의 trace artifact directory도 존재하지 않거나 비어 있어야 합니다.
Recorder는 기존 파일이 있는 경로를 거부하므로 run마다 새 경로를 사용합니다.
Replayer의 `--collection` 역시 매 실행마다 새 target 이름을 사용합니다. 자세한 이유는
[기록 가이드](record.md#run마다-새로-정할-값)를 참조합니다.

## Text RAG

설정 파일은 [`config/milvus_trace_text.yaml`](../../config/milvus_trace_text.yaml)입니다.
Wikipedia corpus를 chunking/embedding한 뒤 Natural Questions query를 실행합니다.

데이터 크기는 `bench.preprocessing.dataset_ratio`, query 수는
`rag.retrieval.question_num`으로 조절합니다.

### Record

```bash
python src/run_new.py \
  --config config/milvus_trace_text.yaml \
  --msys-config "$MSYS_CONFIG"
```

완료되면 `$MNTPNT/artifacts/text`에 bootstrap corpus, timed search event, manifest와
`SHA256SUMS`가 생성됩니다. Text pipeline에 timed query/insert가 추가된 경우에만 해당
payload shard도 생성됩니다.

### Python replay

```bash
python -m milvus_trace.replay \
  --artifact-dir "$MNTPNT/artifacts/text" \
  --uri "$REPLAY_MILVUS_URI" \
  --token "$MILVUS_TOKEN" \
  --collection replay_text \
  --result-file "$MNTPNT/results/text.json"
```

## Image RAG

설정 파일은 [`config/milvus_trace_image.yaml`](../../config/milvus_trace_image.yaml)입니다.
ArXiv PDF를 `$MNTPNT/datasets/arxiv`에 내려받아 page image로 변환하고 ColPali로
embedding합니다. PDF 변환을 위해 host에 Poppler가 설치되어 있어야 합니다.

PDF 수는 `bench.preprocessing.dataset_ratio`, query 수는
`rag.retrieval.question_num`으로 조절합니다. page별 multi-vector와 `doc_id`,
`seq_id`, `filepath`가 insert payload에 포함됩니다.

### Record

```bash
python src/run_new.py \
  --config config/milvus_trace_image.yaml \
  --msys-config "$MSYS_CONFIG"
```

Image retrieval은 vector search 후 같은 `doc_id`의 vector를 Milvus `query`로 읽어
late-interaction score를 계산합니다. 따라서 artifact에는 `search`와 scalar `query`
event가 함께 기록될 수 있습니다.

### Python replay

```bash
python -m milvus_trace.replay \
  --artifact-dir "$MNTPNT/artifacts/image" \
  --uri "$REPLAY_MILVUS_URI" \
  --token "$MILVUS_TOKEN" \
  --collection replay_image \
  --result-file "$MNTPNT/results/image.json"
```

Python replay는 recorded Milvus payload만 전송하므로 target host에 PDF, Poppler, ColPali,
vision LLM이 필요하지 않습니다. `filepath`는 scalar payload로 보존되지만 replay가 그
파일을 열지는 않습니다.

## Audio RAG

설정 파일은 [`config/milvus_audio.yaml`](../../config/milvus_audio.yaml)입니다.
LibriSpeech audio를 Whisper로 transcript한 뒤 sentence vector를 생성합니다.

Audio record host에는 `torchcodec`와 FFmpeg shared library가 필요합니다.
`ffmpeg -version`으로 FFmpeg를 확인할 수 있습니다. 작은 기능 확인에는
`config/milvus_audio_smoke.yaml`을 사용합니다. 이 설정은 streaming dummy dataset에서
8개 sample만 처리하므로 `sample_count: 8`을 사용합니다.

Whisper와 embedding model은 record 때만 필요합니다. Replay는 artifact에 저장된 Milvus
payload를 재생하므로 Whisper, `torch`, CUDA, audio decoder는 필요하지 않습니다.

정확한 데이터 수는 `rag.audio.sample_count`로 지정합니다. 값이 `null`이면
`bench.preprocessing.dataset_ratio`를 사용합니다. streaming dataset에서는 전체
길이를 알 수 없으므로 `sample_count`가 필수입니다.

### Record

```bash
python src/run_new.py \
  --config config/milvus_audio.yaml \
  --msys-config "$MSYS_CONFIG"
```

insert에는 vector, ASR transcript와 JSON metadata가 기록됩니다. query audio와 query
transcript 원문은 artifact에 저장되지 않고, Milvus에 전달한 search vector만 저장됩니다.

Replay는 저장된 typed search vector와 event timing을 사용합니다. 따라서 record 중
ASR와 embedding에 걸린 시간은 event arrival interval에 포함되지만, replay에서 ASR
계산이나 CPU/GPU 사용량을 다시 실행하지는 않습니다.

### Python replay

```bash
python -m milvus_trace.replay \
  --artifact-dir "$MNTPNT/artifacts/audio" \
  --uri "$REPLAY_MILVUS_URI" \
  --token "$MILVUS_TOKEN" \
  --collection replay_audio \
  --result-file "$MNTPNT/results/audio.json"
```

## 크기와 부하 조절

| 목적 | Text/Image | Audio | Replay |
| --- | --- | --- | --- |
| corpus 크기 | `bench.preprocessing.dataset_ratio` | `rag.audio.sample_count` 또는 `dataset_ratio` | recorded corpus 그대로 |
| query 수 | `rag.retrieval.question_num` | `rag.retrieval.question_num` | recorded event 그대로 |
| embedding batch | `rag.embedding.batch_size` | `rag.audio.batch_size` | 해당 없음 |
| insert batch | `rag.insert.batch_size` | `rag.insert.batch_size` | `--bootstrap-batch-size`로 변경 가능 |
| arrival 속도 | RAG 실행에서 자연스럽게 결정 | ASR/embedding 지연 포함 | `--time-scale`로 변경 가능 |

record가 끝난 artifact의 event 수를 임의로 잘라 replay하지 않습니다. 다른 corpus/query
크기가 필요하면 record config를 조정해 새 artifact를 생성합니다.
