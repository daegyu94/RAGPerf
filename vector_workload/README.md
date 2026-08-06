# Portable vector workload record/replay

이 디렉터리는 GPU server에서 corpus/query embedding을 생성하거나 CPU에서 synthetic
vector를 record하고, GPU가 없는 target server의 Milvus DISKANN에서 replay하는
독립 workflow를 제공한다. RAGPerf의 vLLM, pipeline과 monitoring system을 import하지 않는다.

전체 설계와 현재 RAGPerf의 지원 범위는 [DESIGN.md](DESIGN.md)를 참고한다.

## 디렉터리 구성

```text
vector_workload/
├── README.md
├── DESIGN.md
├── artifact_utils.py
├── export_colpali.py
├── export_vectors.py
├── prepare_workloads.py
├── record_synthetic.py
├── replay_milvus.py
├── requirements.txt
└── examples/
    ├── corpus.jsonl
    └── queries.jsonl
```

`output/`은 실행 시 생성되며 Git에는 포함하지 않는다.

## 입력 형식

Corpus와 query는 UTF-8 JSONL이다. 각 줄에는 `id`와 `text`가 필요하다. `metadata`는
선택 사항이다.

```json
{"id":"doc-001","text":"Document text","metadata":{"source":"example"}}
```

Query에는 해당 query를 제출하기 전 대기 시간을 나타내는 `delay_ms`를 선택적으로 넣을
수 있다. 생략하면 0이다.

```json
{"id":"query-001","text":"Question text","delay_ms":100}
```

Corpus의 각 `text`는 `--chunk-size`와 `--chunk-overlap`에 따라 deterministic character
window로 나뉜다. 이미 chunking된 입력은 `--chunk-size 0`으로 그대로 사용할 수 있다.

## 환경 준비

Project root에서 project-local virtual environment를 준비하고 필요한 package만 설치한다.
아래 명령의 `/MNTPNT`는 artifact와 Milvus data를 저장할 mount 경로로 바꾼다.

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r vector_workload/requirements.txt
```

첫 실행에서는 지정한 Sentence Transformers model을 Hugging Face Hub에서 내려받을 수
있다. 실행 전 `nvidia-smi`로 사용할 GPU가 비어 있는지 확인한다.

`export_vectors.py export`는 옵션 없이 실행하면 `BAAI/bge-m3`를 사용한다. 빠른
CI/smoke test에서만 `--smoke`를 추가하며, 이 모드는
`sentence-transformers/all-MiniLM-L6-v2`를 사용한다. 어느 모드에서든 `--model`을
지정하면 모델을 명시적으로 override할 수 있다.

## Embedding artifact smoke test

Project root에서 다음을 실행한다. Output directory는 비어 있어야 하며 exporter는 기존
artifact를 덮어쓰지 않는다.

```bash
source .venv/bin/activate

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
```

생성된 artifact를 다시 검증하려면 다음을 실행한다.

```bash
python vector_workload/export_vectors.py verify \
  --artifact-dir vector_workload/output/smoke-mixed

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

이 smoke artifact는 corpus 절반을 초기 적재한 뒤 DISKANN index를 만들고,
`search → insert → search → insert` 순서로 미리 embedding된 vector를 replay한다.
Embedding은 GPU server에서 끝나므로 target server에는 GPU가 필요하지 않다.

## Wikipedia + Natural Questions

Hugging Face의 Wikipedia corpus와 Natural Questions query를 JSONL로 준비한 다음 같은
mixed exporter/replayer를 사용한다.

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

다운로드한 `common-pile/arxiv_papers` PDF와 workload query JSONL을 준비한다. Preparer는
PDF page text를 추출하고 exporter가 deterministic chunking과 embedding을 수행한다.

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

생성된 artifact는 위 smoke 예시와 같은 `replay_milvus.py` 명령으로 GPU 없는 Milvus
server에서 initial insert, DISKANN index build와 mixed search/insert를 실행한다.

## arXiv PDF image + ColPali

PDF를 page image로 만들고 GPU server에서 ColPali document/query multi-vector를 모두
생성한다. Artifact에는 image 자체가 아니라 `document_id`, `sequence_id`와 vector가
저장된다. Replayer는 query token별 Milvus search 결과를 document 단위 MaxSim으로 합산한다.

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

`--insert-event-size`는 ColPali workload에서 event당 document 수이다. 한 document의 token
vector는 같은 insert event에 함께 들어간다. Target server에서는 ColPali, PyTorch 또는
GPU가 필요하지 않고 `pyarrow`, `PyYAML`, `numpy`, `pymilvus`만 사용한다.

## Production-like embedding workload

실제 corpus/query에서 text embedding을 생성할 때는 `--smoke` 없이 실행한다. 아래
명령은 기본값인 `BAAI/bge-m3`로 vector를 생성한다.

```bash
source .venv/bin/activate

python vector_workload/export_vectors.py export \
  --corpus-file /MNTPNT/ragperf/custom/input/corpus.jsonl \
  --query-file /MNTPNT/ragperf/custom/input/queries.jsonl \
  --output-dir /MNTPNT/ragperf/vector-workload-bge-m3/artifact \
  --device cuda:0 \
  --batch-size 32 \
  --dtype float32
```

`BAAI/bge-m3`는 일반적으로 1024차원 vector를 생성하므로, 이 artifact를 사용하는
Milvus collection은 manifest의 dimension과 동일하게 새로 만들어야 한다. 기존
384차원 MiniLM artifact와 production artifact를 같은 collection에 섞지 않는다.

## 실제 규모의 vector record/replay

Storage와 VectorDB 경로를 embedding model 성능과 분리해 검증할 때는 synthetic workload를
사용한다. `record_synthetic.py`는 정규화된 vector를 bounded memory로 Parquet shard에
record한다. 각 query는 corpus vector 하나에 작은 noise를 더해 만들며
`metadata_json.expected_id`에 예상 top-1 ID를 기록하므로 replay 후 recall도 확인할 수 있다.
이 workload는 VectorDB I/O 검증용이며 embedding 품질 평가용이 아니다.

다음 예시는 384-dimension float32 corpus vector 100,000개(순수 vector data 약 146 MiB)와
query 2,000개를 `/MNTPNT`에 기록하고, 같은 경로를 data path로 마운트한 Milvus
DISKANN에서 replay한다. `RUN_DIR`과 collection 이름은 새 값이어야 한다. Recorder와
replayer는 기존 artifact, collection 또는 result를 덮어쓰지 않는다.

Milvus는 embedded DB가 아니므로 먼저 Standalone/Cluster의 QueryNode와 IndexNode data path를
`/MNTPNT/ragperf/milvus-data`에 마운트하고 `queryNode.enableDisk: true`로 설정한다.

최소 DISKANN 설정은 다음과 같다.

```yaml
queryNode:
  enableDisk: true
common:
  DiskIndex:
    BeamWidthRatio: 4
```

```bash
source .venv/bin/activate

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
  --collection "ragperf_diskann_100k_001" \
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

Replayer는 먼저 `SHA256SUMS`와 Parquet row count를 검증하고 collection 생성, insert, flush,
DISKANN index build, load, warm-up, measurement 순서로 실행한다. 결과는
`$RUN_DIR/milvus/replay-result.json`에 저장되며 insert 처리량, index 시간, replay QPS,
latency p50/p90/p95/p99와 top-1 recall을 포함한다. 기본값은 `delay_ms`를 무시하고 최대
부하를 만들며, record된 요청 간격을 재현하려면 `--respect-delay`를 추가한다.

대규모 실행 전에 다음을 함께 기록한다.

- Artifact와 Milvus server의 data path가 모두 대상 mount 아래에 있는지 확인한다.
- Dataset size, dimension/dtype, index parameter, `top_k`, concurrency와 warm-up을 고정한다.
- Cold/warm cache 상태를 결과와 함께 명시하고, cache를 임의로 drop하지 않는다.
- 서로 다른 backend 비교에는 같은 artifact를 재사용하고 각 replay마다 새 collection과
  result file을 사용한다.

## 출력

```text
output/smoke-mixed/
├── corpus-00000.parquet
├── inserts-00000.parquet
├── queries-00000.parquet
├── schedule-00000.parquet
├── workload-manifest.yaml
└── SHA256SUMS
```

- Corpus shard: chunk ID, text, metadata와 embedding 또는 synthetic vector
- Query shard: query ID, text, metadata, vector와 `delay_ms`
- Insert shard: index build 후 schedule에 따라 추가할 pre-embedded corpus vector
- Schedule shard: 순서가 고정된 `search`/`insert` event와 event별 row count
- Manifest: 입력/generator, mode, model/revision, vector dimension/dtype, GPU와 처리량
- `SHA256SUMS`: 다운로드 후 전체 artifact 무결성 검증용 checksum

기본 `--rows-per-shard 100000`은 384-dimension float32 vector에서 vector data만 약
146 MiB이다. `record_synthetic.py`의 I/O 검증 workload는 이처럼 model-independent한
384차원을 사용한다. `export_vectors.py`로 실제 text embedding을 생성하는
production workload에서는 `BAAI/bge-m3`의 dimension과 artifact를 별도로 관리한다.

## 실제 workload 생성

Smoke test가 통과하면 example JSONL 대신 실제 corpus와 query JSONL을 지정한다. 동일한
artifact를 여러 storage backend에서 비교하려면 embedding model/revision, chunking,
normalization과 dtype을 변경하지 않는다.

Target cluster의 Milvus에서는 `replay_milvus.py`로 다음 단계를 실행할 수 있다.

1. `SHA256SUMS` 검증
2. Initial corpus Parquet를 Milvus collection에 insert
3. Target storage 위에서 DISKANN index build
4. Schedule 순서로 search와 pre-embedded corpus insert를 mixed replay
5. Search latency, QPS, storage/network/resource metric 수집

현재 standalone replayer의 기준 backend는 Milvus DISKANN이다.

## 현재 제한

- Exporter는 corpus/query와 생성된 vector 전체를 memory에 올린 뒤 shard를 작성한다.
- Exporter 입력은 local JSONL이며 Wikipedia/Natural Questions preparer만 Hugging Face
  streaming input을 지원한다.
- GitHub Release upload는 자동화하지 않는다.
- TB급 본 실행에는 streaming input, incremental embedding과 bounded-memory Parquet shard
  writer를 먼저 구현해야 한다.
- Replayer는 Milvus DISKANN search/insert mixed workload를 지원한다.
- Milvus server 배포와 `/MNTPNT` data path mount는 replayer가 자동으로 수행하지 않는다.
- Process, disk와 network monitoring은 별도 수집해야 한다.
