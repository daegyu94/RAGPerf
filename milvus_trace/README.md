# Milvus 요청 기록과 재생

`milvus_trace`는 RAGPerf가 Milvus에 보내는 `insert`, `search`, `query` 요청을
artifact로 기록하고, 같은 payload와 도착 간격을 다른 Milvus endpoint에 재생합니다.
Text, Image, Audio workload가 같은 artifact 형식과 replayer를 사용합니다. 여기서 `trace`는
record session의 논리적 Milvus 요청 기록이고, `trace artifact`는 이를 replay할 수 있도록
manifest, checksum, Parquet payload를 묶은 directory bundle입니다. 상세 용어는
[artifact 형식](docs/artifact_format.md#용어)을 참조합니다.

```text
Record host                                      Replay host

dataset + model + RAGPerf ──> source Milvus      artifact ──> target Milvus
             │                                      ▲
             └──────── trace artifact ──────────────┘
```

Replay는 embedding, ASR, reranking, generation을 다시 실행하지 않습니다. 이 단계에서
생긴 지연은 Milvus 요청 사이의 도착 간격으로만 재현됩니다. Replay image에도 Milvus
server는 포함되지 않습니다. 처음에는 하나의 Milvus 서버를 record와 replay에 함께
사용할 수 있고, 서로 다른 성능 환경을 비교할 때만 source와 target을 분리합니다.

## Milvus 서버 준비

실제 record와 replay에는 실행 중인 **standalone Milvus 서버**가 필요합니다. 이
repository는 Milvus client, recorder, replayer와 함께 RAGPerf용 standalone 서버를
시작하는 helper를 제공합니다. Docker Engine과 Docker Compose v2 자체는 먼저
설치되어 있어야 합니다.

repository root에서 다음 명령을 실행하면 됩니다.

```bash
./milvus_trace/scripts/milvus-standalone.sh start
```

helper는 저장소에 포함된
[`milvus-standalone-compose.yml`](docker/milvus-standalone-compose.yml)을 사용하여
Milvus, embedded etcd, MinIO 컨테이너를 시작하고 `19530` 포트가 열릴 때까지
기다립니다. 기본 Milvus image는 E2E smoke test에서 확인한 `v2.4.15`입니다.
image를 Docker Hub에서 받으므로 첫 실행에는 네트워크와 충분한 disk 공간이 필요합니다.

구성은 다음처럼 client가 Milvus standalone에 접속하고, standalone이 metadata는 etcd에,
object data는 MinIO에 맡기는 최소 3-service 구조입니다.

```text
                    Client / PyMilvus
                            |
                      localhost:19530
                            v
                    +----------------+
                    | Milvus         |
                    | standalone     |
                    +----------------+
                       |             |
                   metadata     object data
                       v             v
                    +------+      +-------+
                    | etcd |      | MinIO |
                    +------+      +-------+

Host storage ($MILVUS_DATA_DIR, default: $PWD/.milvus/volumes)
├── etcd/       -> /etcd
├── minio/      -> /minio_data
└── milvus/     -> /var/lib/milvus
```

실제 service와 container 경로의 단일 출처는 위 Compose 파일이며, `MILVUS_DATA_DIR`의
세 하위 디렉터리만 host에 영속화됩니다. Docker image layer와 container metadata는
Docker daemon storage에 남습니다.

설치 위치와 데이터 위치는 다음과 같이 구분됩니다.

| 항목 | 기본 위치 | 설명 |
| --- | --- | --- |
| Docker image/container 저장소 | Docker daemon 관리 위치 | `docker info --format '{{.DockerRootDir}}'`로 확인합니다. 일반적인 rootful Linux Docker에서는 `/var/lib/docker`입니다. |
| Milvus/etcd/MinIO 영속 데이터 | `$PWD/.milvus/volumes` | Git에 포함되지 않으며, container를 `stop` 또는 `down`해도 유지됩니다. |
| Milvus gRPC | `http://localhost:19530` | RAGPerf의 `MILVUS_URI`와 replay `--uri`에 사용합니다. |
| Milvus WebUI | `http://localhost:9091/webui/` | 상태를 확인할 때 사용합니다. |

데이터를 repository 밖에 두려면 시작 전에 `MILVUS_DATA_DIR`를 지정합니다.

```bash
export MILVUS_DATA_DIR=/path/to/milvus-data
./milvus_trace/scripts/milvus-standalone.sh start
```

주요 lifecycle 명령은 다음과 같습니다.

```bash
./milvus_trace/scripts/milvus-standalone.sh status  # container와 저장 위치 확인
./milvus_trace/scripts/milvus-standalone.sh logs    # Milvus log 확인
./milvus_trace/scripts/milvus-standalone.sh stop    # 중지, 데이터 유지
./milvus_trace/scripts/milvus-standalone.sh down    # container 제거, 데이터 유지
```

Docker가 설치되어 있지 않거나 Docker daemon에 접근할 수 없는 경우에는 helper가
명확한 오류를 출력하고 종료합니다. 운영/분산 Milvus를 사용할 때는 helper 대신
[Vector Database Module의 standalone 또는 GPU 설정 안내](../src/vectordb/README.md#2-milvus-gpu-via-docker-compose)를
따릅니다.

처음 테스트할 때는 standalone 서버 하나만 `http://localhost:19530`에 실행하면
됩니다. 같은 endpoint를 source와 target으로 사용하되, record collection과 replay
collection 이름은 다르게 지정합니다. 두 서버가 필요한 경우는 서로 다른 Milvus
환경의 성능을 비교할 때뿐입니다. Docker replay image도 서버를 포함하지 않으므로
image에서 이 endpoint로 접근할 수 있어야 합니다.

서버가 준비되었는지 확인하려면 다음을 실행합니다.

```bash
source .venv/bin/activate
python -c "from pymilvus import MilvusClient; print(MilvusClient(uri='http://localhost:19530', token='root:Milvus').list_collections())"
```

출력이 다음처럼 `[]`이면 정상입니다. 이는 **Milvus 서버에는 연결되었지만 아직
collection이 하나도 없다**는 뜻입니다.

```text
[]
```

record를 실행한 뒤에는 생성된 collection 이름이 목록에 표시됩니다. 반대로
`connection refused`, timeout, 인증 오류가 나오면 서버 실행 상태나 URI/token을
확인해야 합니다.

## 어떤 실행 방법을 선택해야 하나요?

| 목적 | 필요한 환경 | 시작 문서 |
| --- | --- | --- |
| Record host: 실제 RAG workload 기록 | RAGPerf 전체 의존성, dataset/model, source Milvus | [기록 가이드](docs/record.md) |
| Replay host: host Python replay (Milvus: Docker standalone) | Python, replay 의존성, target Milvus | [재생 가이드](docs/replay.md) |
| Replay host: 격리된 CPU container에서 replay | Docker, target Milvus | [Docker 가이드](docs/docker.md) |
| Text/Image/Audio별 완전한 명령 | workload별 model과 config | [workload 예시](docs/workloads.md) |
| 0.5TB 이상 capacity artifact | GPU 또는 CPU, 충분한 controller storage | [Vector workload](docs/vector_workloads.md) |
| 외부망 없는 VM의 xfs/3FS/pNFS replay | Controller SSH, offline wheels/images, mounted backend | [Staged remote replay](docs/staged_remote_replay.md) |
| 반복 실험 matrix | 준비된 세 topology와 trace archive | [실험 계획](benchmarks/evaluation/README.md) |

아래 명령은 모두 repository root에서 실행한다고 가정합니다.

## 1. Host별 설치와 실행

### Record host (source workload) 준비와 실행

Record host는 dataset과 model을 실행해 실제 RAG 요청을 만들고, source Milvus 요청과 도착
간격을 artifact로 기록합니다. Record와 replay를 같은 host에서 실행할 수도 있지만, 성능 비교
실험에서는 replay host와 target Milvus를 별도로 둡니다.

`setup_venv.sh`를 실행하기 전에 다음 record host prerequisite를 준비합니다.

| 범위 | 필수 항목 | 비고 |
| --- | --- | --- |
| Python | Python 3.10 이상, `venv`/`ensurepip` | 새 virtual environment 생성에 사용 |
| Native build | CMake 3.22 이상, C++20 compatible compiler | monitoring module `libmsys_pymod` 빌드에 사용 |
| Python 설치원 | PyPI 또는 접근 가능한 사내 package index | RAGPerf requirements와 generated build requirements 설치 |
| Record/서버 | Docker Engine, Compose v2, CUDA driver/toolkit | setup script가 설치하지 않으며 workload와 source Milvus에 필요 |

Ubuntu 24.04 기준으로 record/controller host에는 다음 package를 설치합니다.
`python3-venv`가 venv 안의 pip bootstrap을 제공하므로 별도 system-wide pip는 필수가
아닙니다. 이미 checkout한 repository를 사용하면 `git`은 생략할 수 있습니다.

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git \
  python3 python3-venv python3-dev \
  docker.io docker-compose-v2 \
  openssh-client rsync zstd
```

Docker daemon을 일반 사용자로 사용하려면 logout/login 후 다음을 실행합니다.

```bash
sudo usermod -aG docker "$USER"
```

GPU record를 실행할 때는 NVIDIA driver/CUDA를 host 이미지에 맞게 설치해야 합니다. 3FS와
pNFS는 Ubuntu 공통 package가 아니라 각 storage 환경의 client/mount package와 설정을
사용하므로 record host 설치 목록에는 고정하지 않습니다.

위 prerequisite를 준비한 뒤, 처음 설치하는 host 또는 새 shell에서 다음 공통 흐름을
실행합니다. 이미 project virtual environment가 있으면 첫 줄은 건너뛰고 마지막 두 줄만
실행합니다.

```bash
# 처음 준비하는 경우에만 실행
./milvus_trace/scripts/setup_venv.sh

# setup 완료 후 또는 새 shell에서 실행
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

`setup_venv.sh`는 root의 RAGPerf `resource/requirements.in`을 바탕으로 temporary lock을
만들고, 현재 PyPI와 호환되지 않는 일부 버전을 `milvus_trace` 범위에서만 보정한 뒤 Python
package를 설치합니다. 마지막에는 `libmsys_pymod`를 build하므로 record 실행 전에
`src/monitoring_sys/libmsys*.so`가 생성되어 있어야 합니다. `--skip-monitoring` 옵션으로
실행했다면 record 전에 옵션 없이 setup을 다시 실행합니다. Docker, CUDA, C++ compiler와
나머지 system package는 이 스크립트가 설치하지 않습니다.

compatibility override는 다음과 같습니다. `pymilvus==2.3.7`은 root requirements의
Milvus client pin이고, 나머지는 현재 RAGPerf full environment와의 호환성 보정입니다.
생성된 monitoring protobuf 모듈이 요구하는 runtime과 맞추기 위해 `protobuf`도 trace
환경에서만 고정합니다.

```text
vllm==0.8.5.post1
marshmallow<4
setuptools>=74.1.1,<81
protobuf==6.32.0
```

그 외 transitive package의 정확한 버전은 setup 시 생성되는 temporary lock이 관리하므로
README에 별도로 복사하지 않습니다. `source`는 현재 shell에 virtual environment를
적용하는 단계이고, `PYTHONPATH`는 repository의 `src`와 root 모듈을 `run_new.py` 및
workload script가 찾도록 합니다. Workload별 추가 요구 사항은
[workload 예시](docs/workloads.md#workload별-요구-사항)를 확인합니다.

Source Milvus는 앞의 standalone helper 또는 별도 endpoint로 준비합니다. Record host에는
collection 생성, insert, index 생성, search/query 권한이 필요합니다.

### Replay host 준비

Record가 정상 종료되면 `workload-manifest.yaml`, Parquet shard, `SHA256SUMS`가 있는
artifact를 replay host로 전달한 뒤 다음 두 방식 중 하나를 선택합니다. Replay는 artifact에 저장된
Milvus payload와 도착 간격만 재현하며, record의 dataset/model pipeline을 다시 실행하지
않습니다. 따라서 replay host에는 source Milvus, dataset, embedding/ASR/generation model,
GPU와 monitoring system이 필요하지 않습니다. Target Milvus endpoint는 이미 실행 중이어야
하며 replayer가 server를 시작하지는 않습니다.

| 방식 | Replayer 실행 위치 | 필요한 것 | 필요하지 않은 것 |
| --- | --- | --- | --- |
| Python replayer | host의 `python -m milvus_trace.replay` | Python 3.10+, replay lock, artifact, target Milvus | RAGPerf 전체 의존성, dataset/model, GPU, monitoring, source Milvus |
| Docker replayer | replay container | Docker, replayer image, artifact mount, target Milvus | host Python, RAGPerf 전체 의존성, dataset/model, GPU, monitoring, source Milvus |

#### Host Python replayer (표준)

이 표준 경로는 **replayer process를 host Python에서 실행**합니다. Target Milvus는
Docker standalone으로 실행하며, replayer container는 사용하지 않습니다.
Record host에서 사용한 full RAGPerf environment를 재사용하거나, 다른 package와 분리된
replay 전용 environment를 만들 수 있습니다. `setup_venv.sh`는 record용 script이므로
replay host에서 실행할 필요가 없습니다.

기존 project virtual environment를 사용할 때는 다음만 실행합니다.

```bash
source .venv/bin/activate
```

별도 replay environment를 사용할 때는 다음처럼 만듭니다. 다른 package와의 충돌을 피하려면
이 방식을 권장합니다.

```bash
python -m venv .venv-replay
source .venv-replay/bin/activate
```

두 경우 모두 활성화한 environment에 replay lock만 설치합니다.

```bash
python -m pip install --upgrade pip
python -m pip install -r milvus_trace/docker/requirements-replay.lock
```

외부망이 없는 staged replay VM에서는 위의 PyPI 설치를 실행하지 않습니다. Controller에서
wheelhouse를 만든 뒤 `staged_remote_replay.sh prepare-replay`가 source와 wheel을 VM으로
전달하고, VM에서 `pip --no-index`로 같은 lock을 설치합니다. 표준 staged 경로에서는
replay VM의 Docker Compose가 Milvus standalone, embedded etcd, MinIO를 실행해야 하며,
replayer 자체는 staged Python process로 실행합니다. `prepare-replay`는 image를
load하지만 server를 시작하지 않고, `replay` phase의 `run_diskann_replay.sh`가
Compose를 시작하고 health check를 통과한 뒤 replay를 시작합니다.
자세한 경로와 `xfs`/`3FS`/`pNFS` mount 검사는
[staged remote replay](docs/staged_remote_replay.md)를 따릅니다.

#### Docker replayer를 실행하는 경우

다음은 일반 replay에서 replayer process 자체를 container로 격리하는 선택적 경로입니다.
Staged remote replay의 표준 경로에서는 Docker replayer image를 사용하지 않고, Docker는
Milvus standalone server를 실행하는 데만 사용합니다.

이 방식은 replayer 자체를 `milvus_trace/docker/Dockerfile`로 만든 container에서 실행합니다.
Host에는 Python virtual environment나 replay package를 설치하지 않고 Docker Engine만
준비하면 됩니다. Image에는 replay code와 최소 Python dependency만 들어 있으며 Milvus
server, RAGPerf pipeline, dataset/model은 포함되지 않습니다. Artifact를 read-only로
mount하고 target Milvus URI를 container에서 접근 가능한 주소로 지정해야 합니다.

Image build, offline `docker load`, artifact mount 명령은
[Docker replay 가이드](docs/docker.md)를 사용합니다. 외부망 없는 VM에서 image와
wheel을 함께 전달하는 경우에는 [staged remote replay](docs/staged_remote_replay.md)의
controller bundle 절차를 사용합니다.

외부망이 제한된 replay VM에서 Docker로 Milvus standalone까지 실행하는 topology라면
내부 apt mirror에서 다음 package를 설치합니다. 순수 Python replayer가 원격 target Milvus에
접속하는 경우에는 Docker package가 필요하지 않습니다.

```bash
sudo apt-get update
sudo apt-get install -y \
  bash coreutils findutils util-linux tar zstd rsync openssh-server \
  python3 python3-venv docker.io docker-compose-v2 xfsprogs
```

## 2. Smoke test: 실제 record → replay

이 문서에서 smoke test는 synthetic artifact 확인이 아니라, 실제 RAG record를 수행한
뒤 같은 artifact를 새 collection에 replay하는 전체 경로입니다. Artifact-only 단계는
설치 문제를 좁히기 위한 선택적인 사전 진단이며 smoke test에 포함하지 않습니다.

| 종류 | 확인하는 것 | 필요한 것 | 실제 record/replay 전에 필수인가? |
| --- | --- | --- | --- |
| Artifact-only 설치 확인 | Python package, recorder, Parquet/checksum 형식 | Python replay package만 | 아니요. 실제 smoke test가 실패할 때 원인을 좁히는 선택 단계입니다. |
| End-to-end smoke test (이 절의 smoke test) | 실제 Audio RAG의 insert/index/search와 trace/replay | Record host 전체 환경 + standalone Milvus 1개 | 네. 실제 RAG trace 경로를 확인하려면 실행합니다. |

### 2.1 Artifact-only 설치 확인 (실제 smoke test 아님)

Milvus나 GPU 없이 synthetic corpus artifact를 만들어 recorder 파일 형식만 확인합니다.
이 테스트는 실제 RAG 요청이나 Milvus 연결을 검증하지 않습니다. `--artifact-dir`에는
존재하지 않거나 비어 있는 경로를 지정합니다.

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

마지막 명령이 `ragperf-milvus-trace 0`을 출력하면 recorder와 checksum 검증만
동작한 것입니다. 이 결과만으로 실제 RAG record가 준비되었다고 판단하지 마십시오.

### 2.2 End-to-end smoke test: record → replay (기본)

실제 record와 같은 `src/run_new.py`와 Audio RAG pipeline을 사용하되, dataset sample을
8개, query를 4개로 줄입니다. 따라서 다음 단계의 full run으로 가기 전에 dataset/model
download, monitoring, source Milvus insert/index/search, artifact 생성과 replay를 한 번에
검증할 수 있습니다.

이 테스트에는 Audio record에 필요한 Python 의존성(`datasets`, `torchcodec`,
`transformers`, `sentence-transformers`, `soundfile`, `pymilvus`), monitoring
`libmsys*.so`, Whisper와 embedding model, 그리고 실행 중인 standalone Milvus 서버가
필요합니다. `run_new.py`가 Milvus 서버를 시작해 주지는 않으므로 이 단계를 먼저
완료해야 합니다. GPU는 필요하지 않도록 CPU 설정을 사용합니다. 단, model download와
CPU inference 때문에 시간이 걸릴 수 있습니다.

이 smoke config는 `hf-internal-testing/librispeech_asr_dummy`의 `validation` split에서
실제 오디오를 8개만 읽습니다. 전체 LibriSpeech를 받지 않으므로 첫 기능 확인에 적합하며,
일반 Audio run은 `config/milvus_audio.yaml`의 `openslr/librispeech_asr`를 사용합니다.
Audio record host에는 `torchcodec`가 필요합니다. 이 config는 `generation: false`와
`evaluate: false`이므로 Audio smoke만 실행할 때 vLLM은 필요하지 않습니다.


먼저 repository root에서 standalone Milvus 서버를 시작하고 연결을 확인합니다. 이미
서버가 실행 중이면 `start` 명령은 생략할 수 있습니다. 같은 서버를 source와 target
역할에 함께 사용하며, replay 때는 새로운 collection 이름을 사용합니다.

```bash
./milvus_trace/scripts/milvus-standalone.sh start
./milvus_trace/scripts/milvus-standalone.sh status

source .venv/bin/activate
python -c "from pymilvus import MilvusClient; print(MilvusClient(uri='http://localhost:19530', token='root:Milvus').list_collections())"
```

연결 확인 명령이 `[]` 또는 기존 collection 목록을 출력하면 다음 단계로 진행합니다.
`connection refused`, timeout, 인증 오류가 나오면 Milvus 서버 상태나 URI/token을
먼저 확인합니다.

```bash
export MNTPNT="$PWD/artifacts/audio-smoke-run-001"
export MILVUS_URI=http://localhost:19530
export RAG_DEVICE=cpu
export GENERATION_DEVICE=cpu
export MSYS_CONFIG=config/monitor/example_config.yaml
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$MNTPNT/artifacts" "$MNTPNT/results"

python src/run_new.py \
  --config config/milvus_audio_smoke.yaml \
  --msys-config "$MSYS_CONFIG"
```

정상 종료 후 trace를 검증하고, 같은 Milvus 서버에 새 collection으로 replay합니다.

```bash
python -c "from milvus_trace.artifact import verify_artifact; m = verify_artifact('$MNTPNT/artifacts/audio-smoke'); print('events:', m['event_count'], 'operations:', m['operation_counts'])"

python -m milvus_trace.replay \
  --artifact-dir "$MNTPNT/artifacts/audio-smoke" \
  --uri "$MILVUS_URI" \
  --token root:Milvus \
  --collection ragperf_audio_smoke_replay_001 \
  --result-file "$MNTPNT/results/audio-smoke-replay.json"
```

이 단계가 이 프로젝트에서 말하는 실제 smoke test입니다. 성공하면 full record는 같은
명령에서 `config/milvus_audio_smoke.yaml` 대신 원하는 workload config를 사용하고,
dataset/query 설정만 늘려 실행합니다. 즉 smoke test는 별도의 가짜 경로가 아니라 실제
record→replay 경로의 크기만 줄인 실행입니다.

## 3. 실제 workload 기록

Text record는 `config/milvus_trace_text.yaml`을 사용합니다. Image/Audio는 각각
`config/milvus_trace_image.yaml`, `config/milvus_audio.yaml`로 바꿉니다. `MNTPNT`,
`MILVUS_URI`, `RAG_DEVICE`, `GENERATION_DEVICE`는 해당 YAML의 환경변수 값을 확장합니다.
`MSYS_CONFIG`는 monitoring 설정 경로이고, `PYTHONPATH`는 repository module import에 사용합니다.

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
`ragperf_trace_text_run_001`을 사용합니다. `sys.vector_db.trace.output_dir`도 이번 run 전용
trace artifact directory를 가리키도록 확인합니다. Trace artifact directory와 source collection을
이전 run과 공유하지 마십시오. Recorder는 기존 파일이 있는 artifact directory를 거부하며,
자동으로 비우거나 덮어쓰지 않습니다. 용어 정의는
[artifact 형식](docs/artifact_format.md#용어)을 참조합니다.

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
Image/Audio 명령은 [workload 예시](docs/workloads.md)를 참조합니다.

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

전체 timing, concurrency, warm-up 옵션은 [재생 가이드](docs/replay.md)를 참조합니다.

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
Network 선택, 사설 CA, mount 권한과 GHCR 배포는 [Docker 가이드](docs/docker.md)에
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
구조와 검증 계약은 [artifact 형식](docs/artifact_format.md)을 참조합니다.

## 문서 안내

| 문서 | 내용 |
| --- | --- |
| [record.md](docs/record.md) | trace 설정, 기록 시점, 완료 확인, 실패 처리 |
| [replay.md](docs/replay.md) | 실행 순서, CLI 옵션, 결과 JSON, 오류 처리 |
| [workloads.md](docs/workloads.md) | Text, Image, Audio의 record/native/Docker 예시 |
| [docker.md](docs/docker.md) | replay image build, network, mount, GHCR |
| [artifact_format.md](docs/artifact_format.md) | manifest, Parquet shard, checksum 계약 |
| [benchmark_methodology.md](docs/benchmark_methodology.md) | timing과 결과 해석 |
| [audio.md](docs/audio.md) | Audio loader, ASR 경계, 개인정보 범위 |

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
