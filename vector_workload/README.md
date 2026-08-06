# Portable Vector Workload Preparation and Replay

`vector_workload/`는 GPU 서버에서 재사용 가능한 vector artifact를 준비하고, target
서버의 Milvus DISKANN에 이를 replay한다. Replay 경로는 RAGPerf의 vLLM, RAG pipeline,
monitoring module을 import하지 않으므로 target 서버에 GPU가 없어도 된다.

## Supported Workflows

| Workflow | Preparation Tool | Replay |
| --- | --- | --- |
| Text embedding | `record_workload.py text` | Single-vector Milvus search/insert |
| Audio ASR + text embedding | `record_workload.py audio-asr` | Single-vector Milvus search/insert |
| ColPali PDF image | `record_workload.py colpali` | Multi-vector Milvus search with MaxSim |
| Synthetic baseline | `generate_synthetic.py` | Optional single-vector validation with top-1 recall |

현재 standalone replayer의 기준 backend는 Milvus DISKANN이다.

## How It Works

Embedding 생성과 VectorDB 실행을 분리한다. Preparation server에서는 corpus/query vector와
request schedule을 portable artifact로 만들고, target server에서는 이 artifact를 Milvus에
직접 제출한다.

```text
corpus + queries
       ↓
embedding or synthetic generation
       ↓
Parquet artifact + manifest + checksums
       ↓
transfer to target server
       ↓
Milvus insert → index → replay → result JSON
```

이 구조는 embedding device와 model 실행 시간을 VectorDB 결과에서 제외하고, artifact에 정의된
동일한 logical workload를 여러 deployment에서 재사용하기 위한 것이다.

## Quick Start

모든 명령은 repository root에서 실행한다. 먼저 project-local environment를 준비한다.

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r vector_workload/requirements.txt
```

### Docker Smoke Test

Smoke runner는 Milvus 2.6.18 image를 내려받아 DISKANN을 활성화한 임시 container를 시작하고,
CPU에서 artifact 생성과 replay를 완료한 뒤 container, volume과 임시 artifact를 정리한다.

먼저 [Docker Engine](https://docs.docker.com/engine/install/)을 설치하고 현재 사용자가 daemon에
접근할 수 있는지 확인한다.

```bash
docker version
```

일반적인 Linux 설치에서는 Docker 설치와 daemon 설정에 `root` 또는 `sudo` 권한이 필요하다.
Smoke runner 자체는 `docker` command를 직접 실행하므로 현재 사용자에게 Docker socket 접근
권한이 있어야 한다. `docker` group은 root 수준의 권한을 부여하므로
[Docker post-install guide](https://docs.docker.com/engine/install/linux-postinstall/)의 보안 경고를
확인한다.

```bash
bash vector_workload/run_docker_smoke.sh
```

실제 workload는 [Record Guide](docs/RECORD.md)에 따라 artifact로 기록하고,
[Replay Guide](docs/REPLAY.md)에 따라 Milvus에서 실행한다. Milvus container를 직접 유지하거나
data volume을 보존하려면 [Milvus Docker Setup](docs/DOCKER_MILVUS.md)을 참고한다.

## Documentation

- [Record Guide](docs/RECORD.md): corpus/query embedding, schedule recording, and artifact validation
- [Replay Guide](docs/REPLAY.md): Milvus loading, DISKANN replay, options, and result output
- [Dataset Workflows](docs/DATASET_WORKFLOWS.md): dataset-specific input preparation and connection examples
- [Workload Scripts](scripts/README.md): count-based record/replay entrypoints with size estimates
- [Milvus Docker Setup](docs/DOCKER_MILVUS.md): validated Docker image, automatic smoke test,
  persistent Compose deployment, and cleanup
- [Artifact Format](docs/ARTIFACT_FORMAT.md): JSONL input, Parquet shards, manifest, schedule,
  checksums, and vector layout rules
- [Benchmark Methodology](docs/BENCHMARK_METHODOLOGY.md): measurement boundaries, fair comparison
  rules, required metadata, and result interpretation

`examples/`에는 작은 JSONL 입력 예제가 있다. Wikipedia/Natural Questions, arXiv PDF, ColPali,
Audio ASR, production-like workload와 큰 synthetic workload 예시는 [Dataset Workflows](docs/DATASET_WORKFLOWS.md)를
참고한다.
