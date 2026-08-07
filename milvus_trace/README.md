# Milvus 요청 기록과 재생

`milvus_trace`는 RAGPerf가 Milvus에 보내는 `insert`, `search`, `query` 요청을
artifact로 기록하고, 같은 payload와 도착 간격을 다른 Milvus endpoint에 재생합니다.
Text, Image, Audio workload가 같은 artifact 형식과 replayer를 사용합니다.

```text
Record host                                      Replay host

dataset + model + RAGPerf ──> source Milvus      artifact ──> target Milvus
             │                                      ▲
             └──────── trace artifact ──────────────┘
```

Replay는 embedding, ASR, reranking, generation을 다시 실행하지 않습니다. 이 단계에서
생긴 지연은 Milvus 요청 사이의 도착 간격으로만 재현됩니다. Replay image에도 Milvus
server는 포함되지 않으므로 source와 target Milvus는 별도로 준비해야 합니다.

## 어떤 실행 방법을 선택해야 하나요?

| 목적 | 필요한 환경 | 시작 문서 |
| --- | --- | --- |
| 실제 RAG workload 기록 | RAGPerf 전체 의존성, dataset/model, source Milvus | [기록 가이드](docs/RECORD.md) |
| 현재 host에서 native replay | Python, replay 의존성, target Milvus | [재생 가이드](docs/REPLAY.md) |
| 격리된 CPU container에서 replay | Docker, target Milvus | [Docker 가이드](docs/DOCKER.md) |
| Text/Image/Audio별 완전한 명령 | workload별 model과 config | [workload 예시](docs/WORKLOADS.md) |

아래 명령은 모두 repository root에서 실행한다고 가정합니다.

## 1. 설치

### Record host

실제 workload를 기록하려면 먼저 루트 [Installation](../README.md#installation)에 따라
RAGPerf 전체 의존성과 monitoring system을 설치합니다. 기존 project virtual
environment가 있으면 새 환경을 만들지 말고 그 환경을 활성화합니다.

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

`src/run_new.py`는 trace 사용 여부와 관계없이 monitoring module을 import하므로
`src/monitoring_sys/libmsys*.so`도 build되어 있어야 합니다. Workload별 추가 요구 사항은
[workload 예시](docs/WORKLOADS.md#workload별-요구-사항)를 확인합니다.

Milvus server 설치는 [Vector Database Module](../src/vectordb/README.md#2-milvus-gpu-via-docker-compose)을
참조합니다. Record host에는 collection 생성, insert, index 생성, search/query 권한이
필요합니다.

### Native replay만 사용하는 host

RAGPerf model 의존성 없이 replay만 실행할 때는 replay image와 같은 lock file을 설치할
수 있습니다. Python 3.10 이상을 사용합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r milvus_trace/docker/requirements-replay.lock
```

Docker replay를 사용할 host에는 Python 환경이 필요하지 않습니다.

## 2. Recorder smoke test

먼저 Milvus나 GPU 없이 synthetic corpus artifact를 만들어 recorder와 설치 상태를
확인할 수 있습니다. `--artifact-dir`에는 존재하지 않거나 비어 있는 경로를 지정합니다.

```bash
export TRACE_ROOT="$PWD/artifacts/smoke-test"

python -m milvus_trace.stress \
  --mode record \
  --artifact-dir "$TRACE_ROOT" \
  --rows 1000 \
  --dimension 128 \
  --batch-size 100

python -c "from milvus_trace.artifact import verify_artifact; m = verify_artifact('$TRACE_ROOT'); print(m['format'], m['event_count'])"
```

마지막 명령이 `ragperf-milvus-trace 0`을 출력하면 recorder와 checksum 검증이
동작한 것입니다. 이 smoke artifact는 bootstrap insert만 만들기 때문에 timed event가
0개입니다. 실제 도착 패턴은 다음 단계의 RAG workload로 기록합니다.

## 3. 실제 workload 기록

Text 예시의 공통 환경을 준비합니다. 경로와 endpoint는 환경에 맞게 바꿉니다.

```bash
export MNTPNT=/path/to/ragperf-data/run-001
export MILVUS_URI=http://source-milvus.example:19530
export RAG_DEVICE=cuda:0
export GENERATION_DEVICE=cuda:1
export MSYS_CONFIG=config/monitor/example_config.yaml
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$MNTPNT/artifacts" "$MNTPNT/results"
```

실행 전 [`config/milvus_trace_text.yaml`](../config/milvus_trace_text.yaml)의
`sys.vector_db.collection_name`을 이번 run 전용 이름으로 바꿉니다. 예를 들어
`ragperf_trace_text_run_001`을 사용합니다. Artifact directory와 source collection을
이전 run과 공유하지 마십시오. Recorder는 기존 경로를 자동으로 비우지 않습니다.

```bash
python src/run_new.py \
  --config config/milvus_trace_text.yaml \
  --msys-config "$MSYS_CONFIG"
```

정상 종료 후 artifact를 검증합니다.

```bash
python -c "from milvus_trace.artifact import verify_artifact; m = verify_artifact('$MNTPNT/artifacts/text'); print('events:', m['event_count'], 'operations:', m['operation_counts'])"
```

Text config는 Wikipedia corpus와 Natural Questions query, embedding model, generation
model을 다운로드하므로 첫 실행은 오래 걸릴 수 있습니다. 작은 실행을 만드는 설정과
Image/Audio 명령은 [workload 예시](docs/WORKLOADS.md)를 참조합니다.

## 4. Artifact 재생

Target Milvus에 아직 존재하지 않는 collection 이름을 선택합니다. Source와 target이
같은 Milvus여도 되지만 source collection과 다른 이름이어야 합니다.

```bash
export REPLAY_MILVUS_URI=http://target-milvus.example:19530
export MILVUS_TOKEN=root:Milvus

python -m milvus_trace.replay \
  --artifact-dir "$MNTPNT/artifacts/text" \
  --uri "$REPLAY_MILVUS_URI" \
  --token "$MILVUS_TOKEN" \
  --collection replay_text_run_001 \
  --result-file "$MNTPNT/results/text-run-001.json"

python -m json.tool "$MNTPNT/results/text-run-001.json"
```

Replayer는 checksum을 먼저 검증하고 target collection을 생성한 뒤 bootstrap corpus,
index, timed event 순서로 처리합니다. 기본값은 기록된 도착 간격과 1배 속도입니다.
CLI는 `--result-file`을 생략하면 결과를 화면에 출력하지 않으므로 결과를 보존하려면
항상 이 옵션을 지정하는 것이 좋습니다.

2배 빠른 replay와 도착 간격을 무시한 replay는 각각 다음 옵션을 사용합니다. 각 실행은
새 target collection 이름을 사용해야 합니다.

```text
--time-scale 2
--timing none
```

전체 timing, concurrency, warm-up 옵션은 [재생 가이드](docs/REPLAY.md)를 참조합니다.

## 5. Docker replay

Docker image는 CPU-only replay 환경을 만들 때 사용합니다.

```bash
export REPLAYER_IMAGE=ragperf-milvus-replayer:local

docker build \
  -f milvus_trace/docker/Dockerfile \
  -t "$REPLAYER_IMAGE" .
```

```bash
export DOCKER_NETWORK=milvus-network

docker run --rm --network "$DOCKER_NETWORK" \
  -v "$MNTPNT/artifacts/text:/artifact:ro" \
  -v "$MNTPNT/results:/output" \
  "$REPLAYER_IMAGE" \
  --artifact-dir /artifact \
  --uri "$REPLAY_MILVUS_URI" \
  --token "$MILVUS_TOKEN" \
  --collection replay_text_docker_run_001 \
  --result-file /output/text-docker-run-001.json
```

Container에서 `REPLAY_MILVUS_URI`의 hostname을 해석하고 접속할 수 있어야 합니다.
Network 선택, 사설 CA, mount 권한과 GHCR 배포는 [Docker 가이드](docs/DOCKER.md)에
설명되어 있습니다.

## Artifact 공유

완성된 artifact는 Parquet shard를 다시 작성하지 않고 deterministic `tar.zst`로 묶을
수 있습니다. Host에 `tar`와 `zstd` executable이 필요하며 output archive는 기존에
존재하지 않아야 합니다.

```bash
python -m milvus_trace.package_artifact \
  --artifact-dir "$MNTPNT/artifacts/text" \
  --output "$MNTPNT/text-run-001.tar.zst"
```

명령은 archive와 `text-run-001.tar.zst.sha256`을 함께 생성합니다. 자세한 directory
구조와 검증 계약은 [artifact 형식](docs/ARTIFACT_FORMAT.md)을 참조합니다.

## 문서 안내

| 문서 | 내용 |
| --- | --- |
| [RECORD.md](docs/RECORD.md) | trace 설정, 기록 시점, 완료 확인, 실패 처리 |
| [REPLAY.md](docs/REPLAY.md) | 실행 순서, CLI 옵션, 결과 JSON, 오류 처리 |
| [WORKLOADS.md](docs/WORKLOADS.md) | Text, Image, Audio의 record/native/Docker 예시 |
| [DOCKER.md](docs/DOCKER.md) | replay image build, network, mount, GHCR |
| [ARTIFACT_FORMAT.md](docs/ARTIFACT_FORMAT.md) | manifest, Parquet shard, checksum 계약 |
| [BENCHMARK_METHODOLOGY.md](docs/BENCHMARK_METHODOLOGY.md) | timing과 결과 해석 |
| [AUDIO.md](docs/AUDIO.md) | Audio loader, ASR 경계, 개인정보 범위 |

## 개발 검증

Project virtual environment에서 관련 회귀 테스트와 syntax 검사를 실행합니다.

```bash
source .venv/bin/activate
PYTHONPATH=.:src python -m unittest -v \
  tests.test_milvus_trace \
  tests.test_audio_workload \
  tests.test_config_env \
  tests.test_milvus_workload_paths
python -m py_compile milvus_trace/*.py
```

GPU, model, dataset이 없는 환경에서도 이 테스트와 synthetic recorder smoke test는
실행할 수 있습니다.
