# 이식 가능한 vector workload record/replay

`vector_workload/`는 GPU 서버에서 재사용 가능한 vector artifact를 준비하고, target
서버의 Milvus DISKANN에 이를 replay한다. Replay 경로는 RAGPerf의 vLLM, RAG pipeline,
monitoring module을 import하지 않으므로 target 서버에 GPU가 없어도 된다.

## 지원 workflow

| Workflow | 준비 도구 | Replay |
| --- | --- | --- |
| Text embedding | `export_vectors.py` | Single-vector Milvus search/insert |
| ColPali PDF image | `export_colpali.py` | MaxSim을 사용하는 multi-vector Milvus search |
| Synthetic vector | `record_synthetic.py` | 선택적 top-1 recall을 포함한 single-vector Milvus search |

현재 standalone replayer의 기준 backend는 Milvus DISKANN이다. Milvus를 자동으로 배포하거나
host, filesystem, network, server-side storage metric을 수집하지는 않는다.

## 빠른 시작

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

Exporter는 기존 output directory를 덮어쓰지 않는다. Artifact와 replay result마다 새로운
directory를 사용한다.

## 문서

- [실행 가이드](docs/WORKFLOWS.md): dataset 준비, embedding export, synthetic record,
  Milvus 설정과 replay 명령
- [Artifact 형식](docs/ARTIFACT_FORMAT.md): JSONL 입력, Parquet shard, manifest, schedule,
  checksum과 vector layout 규칙
- [설계 문서](docs/DESIGN.md): 범위, benchmark 방법론, storage 결과 해석과 roadmap

`examples/`에는 작은 JSONL 입력 예제가 있다. Wikipedia/Natural Questions, arXiv PDF,
ColPali, production-like workload와 큰 synthetic workload 예시는 [실행 가이드](docs/WORKFLOWS.md)를
참고한다.
