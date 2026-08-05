# Portable vector workload record/replay

이 디렉터리는 GPU가 있는 H100 server에서 corpus/query embedding을 생성하거나 CPU에서
synthetic vector를 record하고, GPU가 없는 PoC/B300 cluster의 Milvus DISKANN에서 replay하는
독립 workflow를 제공한다. RAGPerf의 vLLM, pipeline과 monitoring system을 import하지 않는다.

전체 설계와 현재 RAGPerf의 지원 범위는 [DESIGN.md](DESIGN.md)를 참고한다.

## 디렉터리 구성

```text
vector_workload/
├── README.md
├── DESIGN.md
├── artifact_utils.py
├── export_vectors.py
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

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r vector_workload/requirements.txt
```

첫 실행에서는 지정한 Sentence Transformers model을 Hugging Face Hub에서 내려받을 수
있다. 실행 전 `nvidia-smi`로 사용할 GPU가 비어 있는지 확인한다.

## Embedding artifact smoke test

Project root에서 다음을 실행한다. Output directory는 비어 있어야 하며 exporter는 기존
artifact를 덮어쓰지 않는다.

```bash
source .venv/bin/activate

python vector_workload/export_vectors.py export \
  --corpus-file vector_workload/examples/corpus.jsonl \
  --query-file vector_workload/examples/queries.jsonl \
  --output-dir vector_workload/output/smoke-h100-pinned \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  --device cuda:0 \
  --batch-size 8 \
  --dtype float32 \
  --chunk-size 256 \
  --chunk-overlap 32
```

생성된 artifact를 다시 검증하려면 다음을 실행한다.

```bash
python vector_workload/export_vectors.py verify \
  --artifact-dir vector_workload/output/smoke-h100-pinned
```

## 실제 규모의 vector record/replay

Storage와 VectorDB 경로를 embedding model 성능과 분리해 검증할 때는 synthetic workload를
사용한다. `record_synthetic.py`는 정규화된 vector를 bounded memory로 Parquet shard에
record한다. 각 query는 corpus vector 하나에 작은 noise를 더해 만들며
`metadata_json.expected_id`에 예상 top-1 ID를 기록하므로 replay 후 recall도 확인할 수 있다.
이 workload는 VectorDB I/O 검증용이며 embedding 품질 평가용이 아니다.

다음 예시는 384-dimension float32 corpus vector 100,000개(순수 vector data 약 146 MiB)와
query 2,000개를 `/mnt/std-ssd`에 기록하고, `/mnt/std-ssd`를 data path로 마운트한 Milvus
DISKANN에서 replay한다. `RUN_DIR`과 collection 이름은 새 값이어야 한다. Recorder와
replayer는 기존 artifact, collection 또는 result를 덮어쓰지 않는다.

Milvus는 embedded DB가 아니므로 먼저 Standalone/Cluster의 QueryNode와 IndexNode data path를
`/mnt/std-ssd/ragperf/milvus-data`에 마운트하고 `queryNode.enableDisk: true`로 설정한다.
DISKANN의 on-disk index가 실제로 `O_DIRECT`를 사용하는지는 Milvus가 포함한 Knowhere/DiskANN
버전에 따라 다를 수 있으므로 replay 중 QueryNode syscall을 별도로 추적해야 한다.

최소 DISKANN 설정은 다음과 같다.

```yaml
queryNode:
  enableDisk: true
common:
  DiskIndex:
    BeamWidthRatio: 4
```

### Milvus 2.6 소스 기준 `O_DIRECT` 조사 결과

Milvus 2.6 소스는 Knowhere `v2.6.18`을 빌드 의존성으로 고정한다. Linux 빌드에서는
DiskANN build가 기본 활성화되고, Knowhere의 표준 C++ DiskANN 경로가
`LinuxAlignedFileReader`를 생성한다. 이 reader의 `open()`은
`O_DIRECT | O_RDONLY | O_LARGEFILE`로 index 파일을 열고, `io_prep_pread()`와
`io_submit()`/`io_getevents()`로 읽는다.

- [Milvus 2.6.18 Knowhere dependency](https://github.com/milvus-io/milvus/blob/v2.6.18/internal/core/thirdparty/knowhere/CMakeLists.txt)
- [Knowhere DiskANN Linux reader](https://github.com/zilliztech/knowhere/blob/v2.6.18/thirdparty/DiskANN/src/linux_aligned_file_reader.cpp)
- [Knowhere DiskANN reader alignment contract](https://github.com/zilliztech/knowhere/blob/v2.6.18/thirdparty/DiskANN/include/diskann/aligned_file_reader.h)
- [Milvus on-disk index configuration](https://milvus.io/docs/v2.4.x/disk_index.md)

따라서 **공식 Linux Milvus 2.6.x binary + `index_type=DISKANN`** 조합에서는 DiskANN의
graph/node sector read가 `O_DIRECT` 경로를 사용한다고 소스상 결론낼 수 있다. 하지만
다음은 같은 보장이 아니다.

- `pq_compressed.bin`, PQ pivot와 index metadata의 초기 load
- Milvus의 object storage, metadata, log와 일반 segment read
- 다른 Milvus 버전, non-Linux build 또는 `BUILD_DISK_ANN=OFF`로 빌드한 custom binary

즉 현재 adapter는 `O_DIRECT` 옵션을 Milvus에 전달하는 것이 아니라 `DISKANN` index를
요청하고, direct-I/O 여부는 서버가 포함한 Knowhere build가 결정한다. 실제 배포 이미지가
이 소스와 일치하는지는 QueryNode를 replayer보다 먼저 syscall trace해서 확인한다.

```bash
source .venv/bin/activate

RUN_DIR=/mnt/std-ssd/ragperf/vector-workload-100k-001

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
  --storage-path-note /mnt/std-ssd/ragperf/milvus-data \
  --insert-batch-size 10000 \
  --index-type DISKANN \
  --metric COSINE \
  --search-list 100 \
  --top-k 10 \
  --warmup-queries 100 \
  --concurrency 8
```

`O_DIRECT` 사용 여부는 replayer 출력으로 판정할 수 없다. 아래 trace를 **replayer 실행
전에** 별도 터미널에서 시작해 QueryNode의 index open과 search syscall을 함께 수집한다.

```bash
QUERYNODE_PID=<querynode-pid-or-host-pid-for-the-container>
sudo strace -ff -yy \
  -e trace=openat,openat2,close,read,pread64,io_submit,io_getevents,io_uring_enter \
  -p "$QUERYNODE_PID" \
  -o "$RUN_DIR/milvus/querynode.strace"

rg 'O_DIRECT|_disk\.index|\.bin' "$RUN_DIR/milvus"/*.strace
```

`O_DIRECT`는 `read()`가 아니라 해당 index file을 여는 `openat(..., O_DIRECT, ...)`에
나타난다. 이후 `pread64` 또는 `io_submit` 요청이 그 file descriptor를 사용한다. 결과는
DISKANN index file에 대한 direct-I/O 여부를 보여주며, Milvus의 metadata/log/object-store
read까지 모두 direct-I/O라는 뜻은 아니다.

`ptrace` 정책이나 컨테이너 보안 설정 때문에 실행 중인 QueryNode에 `strace -p`를 붙일 수
없는 환경에서는 Milvus 프로세스 자체를 처음부터 `strace`로 시작한다. 이 방법은 index
open 시점도 놓치지 않는다.

```bash
strace -ff -yy \
  -e trace=openat,openat2,pread64,io_submit,io_getevents,io_uring_enter \
  -o "$RUN_DIR/milvus/milvus.strace" \
  /path/to/milvus run standalone
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
output/smoke-h100-pinned/
├── corpus-00000.parquet
├── queries-00000.parquet
├── workload-manifest.yaml
└── SHA256SUMS
```

- Corpus shard: chunk ID, text, metadata와 embedding 또는 synthetic vector
- Query shard: query ID, text, metadata, vector와 `delay_ms`
- Manifest: 입력/generator, chunking, model/revision, vector dimension/dtype, GPU와 처리량
- `SHA256SUMS`: 다운로드 후 전체 artifact 무결성 검증용 checksum

기본 `--rows-per-shard 100000`은 384-dimension float32 vector에서 vector data만 약
146 MiB이다. Text와 Parquet overhead를 포함해도 GitHub Release의 asset당 2 GiB 제한보다
충분히 작게 유지하기 위한 보수적인 기본값이다.

## 검증된 smoke test

2026-08-05에 위 example input과 고정된 model revision으로 H100 smoke test를 실행했다.
사용한 snapshot은 `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`이다.

| 항목 | 결과 |
| --- | ---: |
| GPU | NVIDIA H100 80GB HBM3 |
| Corpus document/vector | 8 / 8 |
| Query vector | 4 |
| Vector dimension | 384 |
| Vector dtype | float32 |
| Normalization | L2 normalized |
| Parquet shard | 2 |
| 전체 smoke artifact | 23,700 bytes |

별도 verify 실행과 함께 vector shape `(8, 384)`, `(4, 384)`, L2 norm 약 1.0을 확인했다.
각 query의 cosine similarity가 가장 높은 corpus도 의도한 smoke-test document와 일치했다.

## 재생성된 100k vector artifact

2026-08-05에 위 설정으로 `/mnt/std-ssd/ragperf/vector-workload-milvus-20260805`에서
vector record와 checksum verify를 다시 실행했다. 이는 실제 replay에 재사용할 수 있는
artifact 검증 결과이며, Milvus replay는 endpoint가 실행 중인 target에서 별도로 수행해야
한다.

| 항목 | 결과 |
| --- | ---: |
| Corpus / query vector | 100,000 / 2,000 |
| Dimension / dtype | 384 / float32 |
| Artifact size | 145,223,078 bytes |
| Record time | 2.35 s |
| Milvus DISKANN result | 아래 실제 replay 결과 참고 |

## 실제 Milvus DISKANN replay 및 `O_DIRECT` 검증

2026-08-05에 위 artifact를 공식 Milvus `v2.6.18` standalone DEB binary로 실제
생성·insert·index build·load·search했다. QueryNode 설정은 `queryNode.enableDisk: true`였고,
Milvus local storage와 index cache는 모두 `/mnt/std-ssd` 아래에 두었다. 실행 결과는
`/mnt/std-ssd/ragperf/vector-workload-milvus-20260805/milvus/replay-result-02.json`에
보존되어 있다.

| 항목 | 결과 |
| --- | ---: |
| Collection | `ragperf_diskann_100k_20260805_02` |
| Server / client | Milvus 2.6.18 / PyMilvus 2.6.17 |
| Corpus / query | 100,000 / 2,000 |
| Index | DISKANN, COSINE, `search_list=100` |
| Insert | 21.42 s (4,669 rows/s) |
| Index build | 95.15 s |
| Replay | 12.66 s (157.98 queries/s) |
| Latency | p50 47.14 ms, p95 72.53 ms, p99 91.10 ms |
| Top-1 recall | 1.0 (2,000/2,000 expected-neighbor queries) |

Milvus binary를 처음부터 `strace`로 실행한 trace는
`/mnt/std-ssd/ragperf/vector-workload-milvus-20260805/milvus-run-20260805-02/`에
있다. `strace` 결과에서 `*_disk.index`를 다음처럼 열었다.

```text
openat(.../468167984654899642_1_468167984654659630_103/_disk.index,
       O_RDONLY|O_DIRECT) = 65
io_submit(... aio_lio_opcode=IOCB_CMD_PREAD,
          aio_fildes=65, aio_nbytes=4096, aio_offset=26599424) = 1
io_getevents(... res=4096, ...) = 1
```

두 DISKANN segment의 index file에 대해 `O_DIRECT` open 135회,
`io_submit` 135회, `io_getevents` 135회를 확인했다. 따라서 이 실제 실행에서는
**Milvus DISKANN vector search가 `O_DIRECT` direct-I/O 경로를 사용한다**고 보면 된다.

Artifact checksum 검증과 다섯 개 Parquet shard 생성은 성공했다. 이번 실행은
machine-readable 원본 결과를 `milvus/replay-result-02.json`에, 전체 Milvus process의
syscall 결과를 `milvus-run-20260805-02/milvus.strace.*`에 보관했다. 다음 실행에서는
각각 새 collection·result path·trace prefix를 사용한다.

## GitHub Release로 전달

검증된 artifact directory의 Parquet shard, `workload-manifest.yaml`과 `SHA256SUMS`를 같은
GitHub Release에 등록한다. GitHub Release는 release당 asset 1,000개, asset당 2 GiB
미만이므로 실제 대규모 실행에서는 shard 크기와 개수를 함께 계획해야 한다.

현재 exporter의 기본 shard는 smoke test와 중간 규모 검증에 맞춘 작은 크기이다. TB급
artifact를 1,000개 이내의 release asset으로 구성하려면 각 shard를 약 1~1.9 GiB로
키우고, 생성된 파일이 2 GiB 미만인지 upload 전에 확인해야 한다.

## 실제 workload 생성

Smoke test가 통과하면 example JSONL 대신 실제 corpus와 query JSONL을 지정한다. 동일한
artifact를 여러 storage backend에서 비교하려면 embedding model/revision, chunking,
normalization과 dtype을 변경하지 않는다.

Target cluster의 Milvus에서는 `replay_milvus.py`로 다음 단계를 실행할 수 있다.

1. `SHA256SUMS` 검증
2. Corpus Parquet를 Milvus collection에 insert
3. Target storage 위에서 DISKANN index build
4. Query Parquet의 row order로 search 실행하고 필요하면 `delay_ms` 재현
5. Search latency, QPS, storage/network/resource metric 수집

현재 standalone replayer의 기준 backend는 Milvus DISKANN이다.

## 현재 제한

- Exporter는 corpus/query와 생성된 vector 전체를 memory에 올린 뒤 shard를 작성한다.
- 입력은 local JSONL만 지원하며 Hugging Face dataset streaming은 지원하지 않는다.
- GitHub Release upload는 자동화하지 않는다.
- TB급 본 실행에는 streaming input, incremental embedding과 bounded-memory Parquet shard
  writer를 먼저 구현해야 한다.
- Replayer는 Milvus DISKANN search-only workload를 지원한다.
- Milvus server 배포와 `/mnt/std-ssd` data path mount는 replayer가 자동으로 수행하지 않는다.
- Milvus가 실제 `O_DIRECT`를 사용하는지는 adapter 결과만으로 판정하지 않고 QueryNode syscall
  trace로 확인해야 한다.
- Process, disk와 network monitoring은 별도 수집해야 한다.
