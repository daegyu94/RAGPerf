# Vector workload 실행 가이드

이 문서는 [README](../README.md)의 빠른 시작보다 큰 workload를 준비하고 Milvus DISKANN에서
replay하는 방법을 설명한다. 입력과 artifact의 의미는 [Artifact 형식](ARTIFACT_FORMAT.md)을
참고한다.

## 환경 준비

Embedding exporter와 ColPali exporter는 GPU 서버에서 실행한다. Target Milvus 서버에서
replay할 때 사용하는 package는 `pyarrow`, `PyYAML`, `numpy`, `pymilvus`다.

Project root에서 project-local virtual environment를 준비한다.

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r vector_workload/requirements.txt
```

첫 실행에서는 지정한 Sentence Transformers 또는 ColPali model을 Hugging Face Hub에서
다운로드할 수 있다. Embedding을 생성하기 전에 `nvidia-smi`로 사용할 GPU가 비어 있는지
확인한다.

`export_vectors.py export`의 기본 model은 `BAAI/bge-m3`다. `--smoke`는
`sentence-transformers/all-MiniLM-L6-v2`를 사용하며, `--model`을 지정하면 명시한 model을
사용한다.

## Wikipedia + Natural Questions

Preparer로 Hugging Face의 Wikipedia corpus와 Natural Questions query를 JSONL로 만든 뒤
text exporter와 replayer를 사용한다.

```bash
python vector_workload/prepare_workloads.py wikipedia-nq \
  --output-dir /MNTPNT/ragperf/wikipedia-nq/input \
  --corpus-count 100000 \
  --query-count 2000

python vector_workload/export_vectors.py export \
  --corpus-file /MNTPNT/ragperf/wikipedia-nq/input/corpus.jsonl \
  --query-file /MNTPNT/ragperf/wikipedia-nq/input/queries.jsonl \
  --output-dir /MNTPNT/ragperf/wikipedia-nq/artifact \
  --device cuda:0 \
  --initial-corpus-ratio 0.8 \
  --searches-per-insert 10 \
  --insert-event-size 100
```

## arXiv PDF text

Preparer는 PDF page text를 추출하고, exporter는 deterministic character chunking과
embedding을 수행한다.

```bash
python vector_workload/prepare_workloads.py arxiv-pdf-text \
  --pdf-dir /MNTPNT/ragperf/arxiv/pdfs \
  --query-file /MNTPNT/ragperf/arxiv/queries.jsonl \
  --output-dir /MNTPNT/ragperf/arxiv-text/input \
  --max-pdfs 10000

python vector_workload/export_vectors.py export \
  --corpus-file /MNTPNT/ragperf/arxiv-text/input/corpus.jsonl \
  --query-file /MNTPNT/ragperf/arxiv-text/input/queries.jsonl \
  --output-dir /MNTPNT/ragperf/arxiv-text/artifact \
  --device cuda:0 \
  --chunk-size 512 \
  --chunk-overlap 0 \
  --initial-corpus-ratio 0.8 \
  --searches-per-insert 10 \
  --insert-event-size 100
```

## arXiv PDF image + ColPali

ColPali exporter는 PDF page image와 query text를 multi-vector로 변환한다. Artifact에는
image 자체가 아니라 vector와 document/query grouping metadata가 저장된다. Replayer는
query token별 결과를 document 단위 MaxSim으로 합산한다.

```bash
python vector_workload/prepare_workloads.py arxiv-pdf-image \
  --pdf-dir /MNTPNT/ragperf/arxiv/pdfs \
  --query-file /MNTPNT/ragperf/arxiv/queries.jsonl \
  --output-dir /MNTPNT/ragperf/arxiv-image/input \
  --max-pdfs 10000

python vector_workload/export_colpali.py \
  --corpus-file /MNTPNT/ragperf/arxiv-image/input/corpus.jsonl \
  --query-file /MNTPNT/ragperf/arxiv-image/input/queries.jsonl \
  --output-dir /MNTPNT/ragperf/arxiv-image/artifact \
  --model vidore/colpali-v1.2 \
  --device cuda:0 \
  --batch-size 1 \
  --initial-corpus-ratio 0.8 \
  --searches-per-insert 10 \
  --insert-event-size 10

python vector_workload/replay_milvus.py \
  --artifact-dir /MNTPNT/ragperf/arxiv-image/artifact \
  --uri http://localhost:19530 \
  --collection ragperf_arxiv_colpali_001 \
  --result-file /MNTPNT/ragperf/arxiv-image/replay-result.json \
  --index-type DISKANN \
  --metric IP \
  --top-k 10 \
  --token-top-k 100 \
  --concurrency 1
```

`--insert-event-size`는 ColPali workload에서 event당 document 수다. 한 document의 token
vector는 같은 insert event에 함께 들어간다.

## Production-like text embedding

실제 corpus와 query에서는 `--smoke`를 생략한다. 기본값인 `BAAI/bge-m3`를 사용할 경우
대개 1024차원 vector가 생성되므로, target collection의 dimension을 manifest와 맞춘다.
서로 다른 dimension의 artifact를 하나의 collection에 섞지 않는다.

```bash
python vector_workload/export_vectors.py export \
  --corpus-file /MNTPNT/ragperf/custom/input/corpus.jsonl \
  --query-file /MNTPNT/ragperf/custom/input/queries.jsonl \
  --output-dir /MNTPNT/ragperf/vector-workload-bge-m3/artifact \
  --device cuda:0 \
  --batch-size 32 \
  --dtype float32
```

동일한 artifact를 여러 target에서 비교하려면 model/revision, chunking, normalization,
dtype을 고정한다.

## Synthetic vector record/replay

Embedding model 성능과 VectorDB I/O를 분리할 때 synthetic workload를 사용한다.
`record_synthetic.py`는 정규화된 vector를 bounded memory로 Parquet shard에 record한다.
각 query는 corpus vector에 작은 noise를 더해 만들며,
`metadata_json.expected_id`에 예상 top-1 ID를 기록한다. 이 workload는 VectorDB I/O 검증용이며
embedding 품질 평가용이 아니다.

```bash
RUN_DIR=/MNTPNT/ragperf/vector-workload-100k-001

python vector_workload/record_synthetic.py \
  --output-dir "$RUN_DIR/artifact" \
  --corpus-count 100000 \
  --query-count 2000 \
  --dimension 384 \
  --dtype float32 \
  --rows-per-shard 25000 \
  --query-noise 0.01 \
  --seed 42

python vector_workload/export_vectors.py verify \
  --artifact-dir "$RUN_DIR/artifact"

python vector_workload/replay_milvus.py \
  --artifact-dir "$RUN_DIR/artifact" \
  --uri http://localhost:19530 \
  --token root:Milvus \
  --collection ragperf_diskann_100k_001 \
  --result-file "$RUN_DIR/milvus/replay-result.json" \
  --storage-path-note /MNTPNT/ragperf/milvus-data \
  --insert-batch-size 10000 \
  --index-type DISKANN \
  --metric COSINE \
  --search-list 100 \
  --top-k 10 \
  --warmup-queries 100 \
  --concurrency 8
```

Replayer는 checksum과 Parquet row count를 검증한 뒤 collection 생성, initial insert, flush,
DISKANN index build, load, warm-up, measurement 순서로 실행한다. 결과 JSON에는 insert
처리량, index 시간, replay QPS, latency p50/p90/p95/p99와 synthetic top-1 recall이 포함된다.

기본값은 `delay_ms`를 무시하고 가능한 높은 부하를 만든다. 입력의 요청 간격을 재현하려면
`--respect-delay`를 추가한다.

## Milvus data path 준비

Milvus는 embedded database가 아니다. Standalone 또는 Cluster의 QueryNode와 IndexNode data
path를 대상 mount에 연결하고, QueryNode에서 disk index를 활성화한다.

```yaml
queryNode:
  enableDisk: true
common:
  DiskIndex:
    BeamWidthRatio: 4
```

`--storage-path-note`는 결과 JSON에 경로를 기록하는 메모일 뿐이며, replayer가 mount나
Milvus 설정을 변경하지는 않는다.

## 실험 체크리스트

- Artifact와 Milvus data path가 모두 의도한 mount 아래에 있는지 확인한다.
- Dataset size, dimension/dtype, model/revision, index parameter, `top_k`, concurrency,
  warm-up과 측정 시간을 고정한다.
- Cold/warm cache 상태를 결과와 함께 기록한다.
- Backend를 비교할 때 같은 artifact를 재사용하고, 매 replay마다 새 collection과 result
  file을 사용한다.
- Client metric만으로 storage 성능을 단정하지 말고 Milvus와 storage server metric을
  별도로 수집한다.

현재 standalone runner의 상세 범위와 해석상의 주의사항은 [설계 문서](DESIGN.md)에 정리되어
있다.
