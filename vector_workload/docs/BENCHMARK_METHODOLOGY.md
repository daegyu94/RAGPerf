# Vector Workload Benchmark Methodology

이 문서는 `vector_workload/`로 VectorDB 또는 storage deployment를 비교할 때의 measurement
boundary, 실험 조건과 결과 해석 기준을 정의한다. 실행 명령은
[Dataset Workflows](DATASET_WORKFLOWS.md), artifact contract는 [Artifact Format](ARTIFACT_FORMAT.md)을
참고한다.

## Measurement Boundary

Standalone replayer는 portable artifact를 Milvus에 insert하고 index를 생성한 뒤 query
schedule을 실행한다. 결과 JSON에는 initial insert 처리량, index 생성 시간, replay QPS,
latency percentile과 expected neighbor가 있는 query의 top-1 recall이 포함된다.

이 결과는 raw storage benchmark가 아니다. Milvus client와 server, index traversal, cache,
serialization, network와 storage I/O가 포함된 end-to-end VectorDB 결과다. Filesystem 자체의
성능을 비교하려면 fio 같은 storage benchmark를 별도로 실행한다.

Synthetic artifact는 embedding model 없이 VectorDB 경로를 검증하는 deterministic baseline이다.
Embedding 품질이나 실제 application traffic의 대표성을 평가하지 않는다.

## Logical Replay Scope

Artifact는 corpus/query vector, operation 순서와 선택적인 `delay_ms`를 저장한다. Replayer는
이를 이용해 target deployment에서 같은 logical workload를 다시 실행하고 latency를 새로
측정한다.

현재 지원 범위는 search와 scheduled insert다. Open-loop 또는 closed-loop arrival model, 고정
duration, update/delete mix, timeout/error policy는 지원하지 않는다. 따라서 결과를 end-to-end
service scalability나 storage-oriented macrobenchmark로 해석하지 않는다.

`--concurrency`는 client worker 수를 제한하지만 production arrival distribution을 모델링하지
않는다. `--respect-delay`를 지정하지 않으면 artifact의 `delay_ms`를 무시하고 가능한 빠르게
query를 제출한다.

## Fair Comparison Rules

Backend 또는 deployment를 비교할 때 다음 조건을 지킨다.

1. 동일한 artifact와 checksum을 모든 target에서 재사용한다.
2. VectorDB engine을 고정하고 비교하려는 storage 또는 deployment 설정만 변경한다.
3. Index type과 parameter, metric, `top_k`, batch size, concurrency와 warm-up을 고정한다.
4. 매 replay에 새 collection과 result file을 사용한다.
5. Cold/warm cache 상태와 background index build 또는 compaction 여부를 기록한다.
6. 같은 host 조건과 반복 횟수를 사용하고 단일 실행 결과만으로 결론 내리지 않는다.

VectorDB engine까지 달라지면 결과를 storage 단독 성능 차이가 아니라 deployment architecture
차이로 해석한다.

## Result and Experiment Metadata

Replayer result JSON은 Milvus client/server version, endpoint, database와 collection, index type,
metric, vector layout, consistency level, initial load/index/replay 시간, query/insert 수,
concurrency, `top_k`, warm-up, QPS와 latency percentile을 기록한다. `--storage-path-note`를
지정하면 deployment path 식별 정보도 함께 기록한다.

Result JSON만으로는 전체 실험 조건을 복원할 수 없다. 각 결과와 함께 artifact의
`workload-manifest.yaml`과 `SHA256SUMS`를 보존하고, 다음 외부 조건을 별도 experiment log에
기록한다.

- Cache 상태, 반복 번호와 host 식별 정보
- Milvus server 설정과 index/search parameter 중 result에 없는 값
- Milvus data path 또는 object storage deployment 식별 정보
- Background index build/compaction 여부와 server-side metric 수집 구간

Client result만으로 storage 병목을 판정하지 않는다. Milvus QueryNode, DataNode와 IndexNode,
storage server, network의 metric을 같은 실험 구간에 별도로 수집한다.

## Filesystem Deployment

Milvus DISKANN의 data path와 index cache를 비교 대상 mount에 연결한다. 결과에는 index
traversal과 filesystem I/O가 함께 포함된다. RAGPerf monitoring이 실제 mount device를 추적하는지
확인하지 않았다면 client latency를 disk metric과 직접 연결하지 않는다.

DISKANN의 direct I/O 사용 여부는 Milvus version과 build에 따라 달라질 수 있다. Replayer 옵션만
보고 `O_DIRECT` 사용을 가정하지 말고, 필요한 경우 QueryNode의 index file open과 read syscall을
확인한다.

## Milvus with Object Storage

MinIO 같은 object storage는 replayer가 직접 호출하는 VectorDB가 아니다. Milvus가 object
storage를 사용하도록 배포하고 replayer는 동일하게 Milvus endpoint에 query를 제출한다.

측정값에는 Milvus service, network와 object storage가 모두 포함된다. Milvus와 object storage의
request latency, request count, transferred bytes, queue와 network metric을 함께 수집해야 병목을
구분할 수 있다.
