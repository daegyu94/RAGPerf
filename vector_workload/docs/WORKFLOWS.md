# Vector Workload Workflow Guide

이 문서는 dataset별 input을 준비하고 artifact로 기록하는 recipe를 제공한다. 공통 record 절차는
[Record Guide](RECORD.md), Milvus 실행은 [Replay Guide](REPLAY.md), 입력과 artifact의 의미는
[Artifact Format](ARTIFACT_FORMAT.md)을 참고한다.

먼저 [Record Guide](RECORD.md)의 project-local environment를 준비한다. Embedding exporter와
ColPali exporter는 GPU server에서 실행하며, 첫 실행에서는 지정한 model을 Hugging Face Hub에서
다운로드할 수 있다.

## Wikipedia + Natural Questions

Preparer로 Hugging Face의 Wikipedia corpus와 Natural Questions query를 JSONL로 만든 뒤
text exporter로 기록한다.

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

## arXiv PDF Text

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

## arXiv PDF Images + ColPali

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
```

`--insert-event-size`는 ColPali workload에서 event당 document 수다. 한 document의 token
vector는 같은 insert event에 함께 들어간다. Multi-vector 실행 옵션은
[Replay Guide](REPLAY.md)를 참고한다.

## Production-like Text Embedding

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

## Audio ASR + Text Embedding

Local audio를 ASR transcript corpus로 준비한 뒤 표준 text exporter와 mixed schedule을
사용할 수 있다. 지원 형식, provenance metadata와 전체 명령은
[Audio ASR Workflow](AUDIO_ASR.md)를 참고한다.

## Synthetic Workload Generation

Embedding model 성능과 VectorDB I/O를 분리하거나 artifact/replay 경로를 검증할 때 선택적으로
synthetic workload를 사용한다. `generate_synthetic.py`는 corpus vector를
`--rows-per-shard` 단위로 생성하고, query 생성에 필요한 anchor vector는 memory에 유지한다.
따라서 peak memory는 corpus 전체가 아니라 shard와 query 수에 비례한다.
각 query는 corpus vector에 작은 noise를 더해 만들며,
`metadata_json.expected_id`에 예상 top-1 ID를 기록한다. 이 workload는 VectorDB I/O 검증용이며
실제 dataset의 embedding 품질을 평가하지 않는다.

```bash
RUN_DIR=/MNTPNT/ragperf/vector-workload-100k-001

python vector_workload/generate_synthetic.py \
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
```

Synthetic artifact의 expected neighbor와 top-1 recall 실행 방법은
[Replay Guide](REPLAY.md)를 참고한다.

## Milvus Data Path Setup

Milvus는 embedded database가 아니다. Standalone 또는 Cluster의 QueryNode와 IndexNode data
path를 대상 mount에 연결하고, QueryNode에서 disk index를 활성화한다.

검증된 Milvus `2.6.18` Standalone container를 local volume으로 실행하는 방법은
[Milvus Docker Setup](DOCKER_MILVUS.md)을 참고한다.

```yaml
queryNode:
  enableDisk: true
common:
  DiskIndex:
    BeamWidthRatio: 4
```

Target data path를 result에 기록하는 방법은 [Replay Guide](REPLAY.md)의
`--storage-path-note` 설명을 참고한다.

## Experiment Checklist

Backend 비교 전에는 [Benchmark Methodology](BENCHMARK_METHODOLOGY.md)의 비교 규칙과 필수
metadata를 확인한다. 이 문서는 실행 절차만 다루며, measurement boundary와 결과 해석의 기준은
Benchmark Methodology를 따른다.
