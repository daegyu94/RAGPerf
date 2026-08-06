# Portable Vector Workload Preparation and Replay

`vector_workload/`는 GPU 서버에서 재사용 가능한 vector artifact를 준비하고, target
서버의 Milvus DISKANN에 이를 replay한다. Replay 경로는 RAGPerf의 vLLM, RAG pipeline,
monitoring module을 import하지 않으므로 target 서버에 GPU가 없어도 된다.

## Supported Workflows

| Workflow | Preparation Tool | Replay |
| --- | --- | --- |
| Text embedding | `export_vectors.py` | Single-vector Milvus search/insert |
| Audio ASR + text embedding | `prepare_workloads.py audio-asr`, then `export_vectors.py` | Single-vector Milvus search/insert |
| ColPali PDF image | `export_colpali.py` | Multi-vector Milvus search with MaxSim |
| Synthetic baseline | `generate_synthetic.py` | Optional single-vector validation with top-1 recall |

현재 standalone replayer의 기준 backend는 Milvus DISKANN이다. Milvus를 자동으로 배포하거나
host, filesystem, network, server-side storage metric을 수집하지는 않는다.

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

이 구조는 embedding device와 model 실행 시간을 VectorDB 결과에서 제외하고, 동일한 logical
workload를 여러 deployment에서 재사용하기 위한 것이다. 현재 replayer는 실제 application API
traffic을 기록하지 않으며, production traffic을 모델링하는 duration/arrival scheduler나
update/delete workload를 제공하지 않는다.

## Quick Start

모든 명령은 repository root에서 실행한다. 먼저 project-local environment를 준비한다.

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r vector_workload/requirements.txt
```

작은 text-embedding artifact를 생성하고 검증한다.

```bash
python vector_workload/export_vectors.py export \
  --corpus-file vector_workload/examples/corpus.jsonl \
  --query-file vector_workload/examples/queries.jsonl \
  --output-dir vector_workload/output/smoke-mixed \
  --smoke \
  --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  --device cuda:0 \
  --batch-size 8 \
  --dtype float32 \
  --chunk-size 256 \
  --chunk-overlap 32 \
  --initial-corpus-ratio 0.5 \
  --searches-per-insert 1 \
  --insert-event-size 1

python vector_workload/export_vectors.py verify \
  --artifact-dir vector_workload/output/smoke-mixed
```

실행 중인 Milvus 서버에 검증한 artifact를 replay한다.

```bash
python vector_workload/replay_milvus.py \
  --artifact-dir vector_workload/output/smoke-mixed \
  --uri http://localhost:19530 \
  --collection ragperf_smoke_mixed_001 \
  --result-file vector_workload/output/smoke-mixed/replay-result.json \
  --index-type DISKANN \
  --metric COSINE \
  --warmup-queries 0 \
  --concurrency 1
```

Preparation tool은 비어 있지 않은 output directory를 덮어쓰지 않으며, replayer는 기존 result
file이나 Milvus collection을 덮어쓰지 않는다. 실행마다 새 경로와 collection 이름을 사용한다.

## Documentation

- [Workflow Guide](docs/WORKFLOWS.md): dataset preparation, embedding export, synthetic generation,
  Milvus configuration, and replay commands
- [Audio ASR Workflow](docs/AUDIO_ASR.md): audio transcription, text embedding, and mixed replay
- [Artifact Format](docs/ARTIFACT_FORMAT.md): JSONL input, Parquet shards, manifest, schedule,
  checksums, and vector layout rules
- [Benchmark Methodology](docs/BENCHMARK_METHODOLOGY.md): measurement boundaries, fair comparison
  rules, required metadata, and result interpretation

`examples/`에는 작은 JSONL 입력 예제가 있다. Wikipedia/Natural Questions, arXiv PDF,
ColPali, Audio ASR, production-like workload와 큰 synthetic workload 예시는 위 문서를 참고한다.
