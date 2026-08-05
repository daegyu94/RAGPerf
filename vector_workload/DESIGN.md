# Vector workload artifact와 GPU 없는 VectorDB I/O 벤치마킹

이 문서는 GPU를 사용할 수 있는 서버에서 workload를 준비하고, GPU를 바로 사용할 수
없는 PoC/B300 cluster에서 VectorDB와 그 storage backend의 I/O 성능을 평가하려는 경우의
RAGPerf 지원 범위와 권장 구현 방향을 설명한다.

RAGPerf 지원 범위 평가는 2026-08-05의 `49c9794` revision을 기준으로 하며, H100 artifact
생성 prototype은 `vector_workload/`에 추가되어 있다.

## 결론

목표 자체는 RAGPerf의 VectorDB abstraction을 확장하여 구현할 수 있다. 그러나 현재
CLI인 `src/run_new.py`를 그대로 사용해서 GPU 없는 환경에서 retrieval-only benchmark를
실행하는 것은 지원되지 않는다. 현재 query pipeline은 `generation: true`일 때만
실행되고, embedding·vLLM generation·GPU monitoring 경로가 GPU에 결합되어 있기 때문이다.

| 시나리오 | 현재 상태 | 설명 |
| --- | --- | --- |
| GPU 환경의 end-to-end RAG | 지원 | Embedding, retrieval, reranking, generation을 순차적인 batch pipeline으로 실행한다. |
| H100의 corpus/query artifact 생성 | Smoke test 지원 | JSONL을 chunking/embedding하고 Parquet, manifest와 checksum을 생성한다. TB 규모 streaming은 아직 지원하지 않는다. |
| Synthetic vector artifact 생성 | 중간 규모 검증 | Bounded-memory recorder로 100k vector artifact를 실제 mount에 생성하고 검증했다. |
| 기존 embedding을 이용한 insert/index | Milvus DISKANN 지원 | Portable Parquet artifact를 shard 단위로 적재하고 DISKANN index를 생성한다. |
| GPU 없는 환경의 retrieval-only 실행 | Standalone Milvus 지원 | 기존 `run_new.py`는 여전히 불가하지만 별도 replayer가 vLLM/GPU import 없이 실행된다. |
| VectorDB API를 직접 호출하는 microbenchmark | Milvus DISKANN 지원 | Insert/index 시간, latency percentile, QPS와 synthetic top-1 recall을 JSON으로 출력한다. |
| Storage-oriented macrobenchmark | 미지원 | 동시 사용자, arrival pattern, warm-up, duration, read/write mix와 update/delete workload가 구현되어 있지 않다. |
| Workload trace record/replay | Logical workload 지원 | Vector artifact와 query 순서/`delay_ms` replay는 지원한다. 실제 application API call trace는 미지원이다. |

따라서 H100에서 portable embedding artifact를 만들거나 CPU에서 synthetic artifact를
record한 뒤 GPU 없는 target cluster의 Milvus DISKANN에서 replay할 수 있다. 다른 VectorDB backend,
production traffic trace와 storage/server monitoring에는 아래의 나머지 기능이 필요하다.

## 권장 실험 구조

GPU가 필요한 embedding 생성과 GPU가 필요하지 않은 VectorDB 실행을 분리한다.

```text
[H100: workload 준비]
Corpus → chunking → corpus vectors ─┐
Queries → embedding → query vectors ├─ workload artifact + manifest
Request order/timing ────────────────┘
                         │
                         └────────── 이동 ──────────┐
                                                    ↓
                                  [PoC/B300: GPU 없이 실행]
                                  retrieval-only runner
                                             ↓
                                      VectorDB engine
                                             ↓
                                  filesystem / object storage
                                             ↓
                              latency, QPS, bandwidth, resource metrics
```

Workload artifact에는 최소한 다음 정보가 있어야 한다.

- Corpus vector, document ID와 필요한 payload
- Query vector와 query ID
- Vector dimension, dtype, normalization 여부와 embedding model revision
- Operation 종류와 순서, 요청 간격, batch size, `top_k`와 index/search parameter
- Dataset revision, random seed와 artifact checksum

Target cluster에서는 embedding, reranking과 generation을 실행하지 않고 미리 생성한 query
vector를 VectorDB에 직접 제출해야 한다. 이 방식이면 GPU 차이가 VectorDB I/O 결과에
섞이지 않고, 동일한 workload를 여러 storage backend에 반복할 수 있다.

## Storage backend 선택

RAGPerf의 `sys.vector_db.type`은 VectorDB engine을 선택한다. Filesystem이나 MinIO를
선택하는 공통 `storage_backend` 설정은 현재 없다. 따라서 무엇을 평가하려는지에 따라
VectorDB와 storage를 함께 선택해야 한다.

### Filesystem 비교: Milvus DISKANN을 우선 사용

Milvus는 embedded DB가 아니므로 QueryNode/IndexNode의 data path를 동일한 mount 아래로
지정해야 한다. Milvus 2.6.x의 공식 Linux build가 포함하는 Knowhere C++ DiskANN은
`LinuxAlignedFileReader`에서 `O_DIRECT`를 사용하지만, 이는 DiskANN graph/node sector
read에 대한 보장이다. 실제 이미지의 server version/build와 다른 파일 read는 syscall로
확인해야 한다.

이 결과는 raw filesystem benchmark가 아니라 **Milvus DISKANN over mounted storage**의 결과이다.
VectorDB의 index traversal, caching, serialization과 filesystem I/O가 모두 포함된다.
Filesystem 자체의 한계를 따로 확인하려면 fio 등의 storage benchmark 결과를 함께
수집해야 한다.

현재 monitoring system은 설정된 Milvus data path를 자동으로 추적하지 않는다. 실제
mount device를 `DiskMeter`에 연결하는 기능을 추가하기 전에는 RAGPerf의 disk metric이
대상 filesystem I/O를 정확히 나타낸다고 가정하면 안 된다. Remote filesystem에서는
client의 network metric과 storage server metric도 함께 수집해야 한다.

### Object storage 비교: 실제 배포 구조가 Milvus+MinIO일 때 사용

MinIO는 RAGPerf가 직접 읽고 쓰는 VectorDB가 아니다. Milvus가 MinIO를 storage로 사용하게
외부에서 배포한 뒤 RAGPerf는 Milvus endpoint에 query를 보내는 구조가 된다. 이 경우
측정값은 client, Milvus query/data/index node, network와 MinIO를 포함한 전체 stack의
end-to-end 결과이다.

RAGPerf에는 현재 MinIO endpoint, credential, bucket과 storage option을 관리하거나 MinIO
operation을 계측하는 기능이 없다. Milvus와 MinIO의 service-side latency, request 수,
전송 byte, queue와 network metric을 별도로 수집해야 storage 병목을 구분할 수 있다.

### 권장 비교 순서

1. Local NVMe를 data path로 마운트한 Milvus DISKANN으로 runner와 결과 계산을 검증한다.
2. 동일한 artifact와 index/search 설정으로 Milvus QueryNode/IndexNode의 data path를 3FS,
   pNFS 또는 local NVMe mount로 변경한다.
3. 실제 production 후보가 object storage 기반이면 같은 logical workload를
   Milvus+MinIO에 적재하고 전체 stack을 비교한다.
4. 모든 결과에 cold/warm cache 상태, dataset 크기, index 크기, `top_k`, batch size,
   concurrency, warm-up과 측정 시간을 기록한다.

Filesystem과 MinIO 결과를 직접 비교할 때는 VectorDB engine까지 달라질 수 있다. 이
경우 결과를 storage의 단독 성능 차이가 아니라 deployment architecture의 차이로
해석해야 한다.

## Microbenchmark와 macrobenchmark

현재 VectorDB adapter는 insert, index build와 search operation을 제공하므로 개별
operation의 elapsed time을 재는 microbenchmark의 기반으로 사용할 수 있다. 다만 현재
pipeline은 stage 경계 timestamp와 마지막 batch의 시간만 제한적으로 남기며, 공정한
microbenchmark에 필요한 per-operation distribution과 byte accounting은 제공하지 않는다.

RAGPerf에는 embedding부터 generation까지 실행하는 end-to-end pipeline이 있으므로 넓은
의미의 macrobenchmark 형태는 존재한다. 하지만 storage 관점의 realistic
macrobenchmark라고 보기에는 다음 기능이 부족하다.

- Open-loop 또는 closed-loop request scheduler
- Warm-up, 고정 duration과 반복 실행
- Configurable concurrency와 request arrival distribution
- Search, insert, update와 delete의 비율 및 시간에 따른 database 변화
- Query별 latency와 p50/p90/p95/p99, timeout/error rate와 achieved QPS
- Read/write byte, queue wait와 backend service time

현재 `BaseRetriever`는 VectorDB search에 `max_threads=1`을 전달하므로 query batch가 있어도
여러 client worker가 실제 동시 부하를 만드는 구조가 아니다. README가 설명하는
update/delete workload generator도 현재 코드에는 구현되어 있지 않다. 따라서 현 상태의
batch pipeline 결과를 concurrent production workload의 scalability 결과로 해석하면 안
된다.

## LMCache와 달리 trace가 필수인가

필수는 아니다. LMCache의 storage operation은 LLM 실행 중 KV cache 상태와 prefix reuse에
따라 동적으로 발생하므로, GPU가 있는 환경에서 `StorageManager` API를 record한 뒤 다른
cluster에서 replay하는 방식이 유용하다. VectorDB workload는 corpus vectors, query
vectors, operation parameter와 request schedule을 명시하면 storage operation을 다시
발생시킬 수 있다. 이 logical workload artifact가 충분히 결정적이면 API call을 별도로
record하지 않아도 된다.

다만 H100과 target cluster를 분리하고 동일한 production-like 순서와 간격을 보존하려면
workload manifest 또는 trace가 필요하다. 차이는 다음과 같다.

- 기본 권장 방식: vector와 request schedule을 portable workload artifact로 저장한다.
- 선택적 record/replay: 실제 RAG application의 VectorDB client call을 timestamp, operation,
  collection, vector/payload reference와 parameter 단위로 기록하고 replay한다.

실제 vector payload를 trace마다 중복 저장하기보다는 content-addressed artifact의 ID를
참조하는 편이 용량과 재사용성에 유리하다. Replay 결과의 latency는 record 때의 latency를
복사하는 것이 아니라 target cluster에서 새로 측정해야 한다.

## 구현 상태와 추가해야 할 기능

### P0: GPU 없는 재현 가능한 retrieval benchmark

1. **Retrieval-only runner: standalone Milvus DISKANN 구현**
   - `retrieval`을 `generation`과 독립적으로 실행한다.
   - vLLM, responser, reranker와 evaluator를 import하거나 초기화하지 않는다.
   - 미리 생성한 query vector를 직접 VectorDB adapter에 전달한다.
2. **Portable workload artifact: smoke와 100k synthetic record 구현**
   - `vector_workload/export_vectors.py`가 corpus/query vector와 metadata를 Parquet로
     저장하고 manifest에 model revision, dtype, dimension, normalization, checksum과
     request schedule을 기록한다.
   - TB 규모 본 실행을 위해 streaming input, incremental embedding과 bounded-memory shard
     writer를 추가해야 한다.
3. **CPU-only 실행과 monitoring**
   - Hard-coded `cuda:0` 대신 config의 device를 모든 embedding 경로에서 사용한다.
   - CPU에서는 적절한 dtype을 사용하고 CUDA cleanup API를 호출하지 않는다.
   - GPU가 없을 때 NVML initialization과 `GPUMeter`를 생략한다.
   - `/mnt/data1` 같은 hard-coded path를 제거하고 VectorDB `db_path`에서 monitor target을
     찾는다.
4. **Benchmark 결과 규격: standalone Milvus 구현**
   - Warm-up과 measurement phase를 분리한다.
   - Insert/index/replay 시간, batch/vector 수와 DB byte를 남긴다.
   - QPS와 p50/p90/p95/p99 latency를 machine-readable JSON으로 출력한다.

구현된 command와 100k 검증 결과의 단일 실행 가이드는 [README.md](README.md)에 있다.

### P1: Storage와 macro workload 계측

1. Concurrency, duration, arrival distribution과 random seed를 지원하는 scheduler를 추가한다.
2. Search/insert/update/delete mix와 database growth policy를 구현한다.
3. Milvus data path, local/remote filesystem, Milvus와 MinIO deployment metadata를 결과에 남긴다.
4. Client metric과 VectorDB/storage server metric을 같은 monotonic time 기준으로 연결한다.
5. Cache 상태와 index build/compaction background work를 명시적으로 기록한다.

### P2: 선택적 record/replay

실제 application traffic의 대표성이 필요할 때만 VectorDB API recorder와 replayer를
추가한다. 최소 record field는 timestamp, operation, collection, artifact object ID,
batch shape/dtype, search parameter와 correlation ID이다. 이 기능은 P0의 portable artifact와
scheduler를 재사용해야 하며 별도의 payload format을 중복 정의하지 않는다.

## 구현 후 목표 설정 예시

다음은 현재 동작하는 config가 아니라, 위 기능을 구현할 때 목표로 삼을 수 있는 예시다.

```yaml
mode: vector_io

workload:
  corpus_artifact: /path/to/corpus
  query_artifact: /path/to/queries
  operation_mix:
    search: 1.0
  top_k: 10
  batch_size: 32
  concurrency: 16
  warmup_seconds: 60
  duration_seconds: 600
  arrival: closed_loop
  seed: 42

sys:
  vector_db:
    type: milvus
    db_path: http://localhost:19530
    collection_name: ragperf_vectors
  monitoring:
    gpu: false
    vector_db_path: /mnt/target-filesystem/ragperf/milvus-data

rag:
  build_index:
    index_type: DISKANN
    metric_type: COSINE
```

이 설정이 구현되면 H100에서는 artifact를 한 번 생성하고, target cluster에서는
`db_path`, VectorDB deployment 또는 workload pressure만 바꿔 같은 실험을 반복할 수
있다.

## 현재 구현을 판단한 근거

- `src/run_new.py`: Text query pipeline이 `generation: true` 조건 안에 있으며 ingest
  embedding device가 `cuda:0`으로 고정되어 있다.
- `src/RAGPipeline/TextsRAGPipline.py`: Query batch마다 embedding, retrieval과 generation을
  순차 실행하며 retrieval-only mode가 없다.
- `src/encoder/sentenceTransformerEncoder.py`: `float16` model과 CUDA cleanup API를
  사용한다.
- `src/RAGPipeline/retriever/BaseRetriever.py`: VectorDB search에 `max_threads=1`을
  전달한다.
- `src/monitoring_sys/config_parser/msys_config_parser.py`: Import 시 NVML을 초기화하고
  `/mnt/data1`을 monitor 대상 계산에 사용하며, VectorDB process 탐지는 Milvus Docker
  Compose에 한정되어 있다.
- `src/vectordb/`: Milvus, Qdrant, Chroma와 Elasticsearch adapter가 insert/search
  API를 제공하지만 공통 workload scheduler나 result schema는 없다.
- `vector_workload/export_vectors.py`: RAGPerf runtime과 독립적으로 JSONL corpus/query를
  H100에서 embedding하고 Parquet shard, manifest와 checksum을 생성·검증한다.
