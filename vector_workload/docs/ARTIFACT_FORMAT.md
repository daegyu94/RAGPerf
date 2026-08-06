# Vector Workload Artifact Format

이 문서는 exporter와 replayer 사이에서 사용하는 portable artifact contract를 정의한다.
실행 명령은 [Workflow Guide](WORKFLOWS.md), 실험 조건과 결과 해석은
[Benchmark Methodology](BENCHMARK_METHODOLOGY.md)를 참고한다.

## Input JSONL

일반 text corpus와 query는 UTF-8 JSONL이다. 빈 줄은 무시한다.

Corpus record:

```json
{"id":"doc-001","text":"Document text","metadata":{"source":"example"}}
```

Query record:

```json
{"id":"query-001","text":"Question text","delay_ms":100}
```

| Field | Corpus | Query | Meaning |
| --- | --- | --- | --- |
| `id` | optional | optional | Unique identifier; omitted values receive deterministic generated IDs |
| `text` | required | required | Document or query text |
| `metadata` | optional | optional | JSON object preserved as `metadata_json` |
| `delay_ms` | ignored | optional | Delay before the query; defaults to `0` |

ColPali corpus 입력은 `text` 대신 `image_path`를 사용하며, 해당 local image file이 있어야
한다. ColPali query 입력은 위 query 형식을 따른다.

## Chunking and Embedding

Text export에서는 `--chunk-size`와 `--chunk-overlap`으로 corpus `text`를 deterministic
character window로 나눈다. 이미 chunking된 입력은 `--chunk-size 0`으로 사용한다. Query
text는 chunking하지 않는다.

Manifest에는 model, revision, device, dimension, dtype, normalization, seed와 embedding
처리량을 기록한다. `--smoke`는 `sentence-transformers/all-MiniLM-L6-v2`를 사용하며, 기본
text model은 `BAAI/bge-m3`다.

## Artifact Directory Layout

```text
artifact/
├── corpus-00000.parquet
├── inserts-00000.parquet       # optional
├── queries-00000.parquet
├── schedule-00000.parquet      # search-only artifact에서는 optional
├── workload-manifest.yaml
└── SHA256SUMS
```

모든 Parquet file은 Zstandard로 압축한다. 큰 입력은 `--rows-per-shard`에 따라 여러 shard로
나눈다.

### Corpus, Insert, and Query Shards

Single-vector shard에는 다음 column이 있다.

- `id`: string identifier
- `text`: 원본 text 또는 synthetic label
- `metadata_json`: JSON으로 encoding한 metadata
- `vector`: 고정 길이의 float16 또는 float32 list
- `delay_ms`: query shard에만 존재

`corpus`는 collection에 처음 적재할 row를 담는다. `inserts`는 schedule에 따라 나중에
적재할 pre-embedded row를 담는다. ColPali artifact에서는 하나의 document 또는 query가 여러
row로 표현되며, grouping metadata로 document/query identity와 token 순서를 보존한다.

### Schedule Shards

Schedule에는 `sequence`, `operation`, `count` column이 있다. Replayer는 row 순서대로 다음
operation을 소비한다.

- `search`: query 하나를 소비한다.
- `insert`: insert shard에서 `count`개의 row를 소비한다.

Text export는 `searches-per-insert`와 `insert-event-size`를 manifest에 기록한다. ColPali의
insert count는 해당 event의 document에 속한 모든 token vector를 기준으로 한다.

Query 순서는 Parquet row 순서다. `delay_ms`는 replayer에 `--respect-delay`를 지정한 경우에만
적용하며, 지정하지 않으면 가능한 빠르게 workload를 실행한다.

## Manifest and Checksums

`workload-manifest.yaml`의 schema version은 `1`이며 다음 정보를 기록한다.

- producer와 생성 시각
- 입력 종류와 record 수, file input을 사용한 경우 file 이름과 checksum
- Chunking 설정
- Embedding model과 vector 속성
- Workload operation, initial row, scheduled row와 timing model
- 모든 Parquet shard의 path, kind, row 수, byte 수와 SHA-256 checksum

`SHA256SUMS`에는 모든 Parquet shard와 manifest의 checksum이 들어 있다.
`export_vectors.py verify`는 checksum 목록과 manifest의 row/checksum metadata를 모두
검증한 뒤 성공한다.

## Vector Layout and Compatibility

Manifest의 `embedding.dimension`, `embedding.dtype`, `embedding.normalized`,
`embedding.vector_layout`은 target collection이 받아야 하는 vector를 설명한다. 현재 exporter는
모든 새 artifact에 `single_vector` 또는 `multi_vector`를 명시하며, replayer는 이전 schema 1
artifact에서 이 field가 없으면 `single_vector`로 해석한다.

- Text와 synthetic artifact는 `single_vector` semantics를 사용한다.
- ColPali artifact는 `multi_vector` semantics와 MaxSim scoring을 사용한다. Target replay는
  적절한 metric과 `--token-top-k`를 사용해야 한다.

Dimension, dtype, normalization assumption, model revision 또는 vector layout이 다른
artifact를 하나의 collection에 섞지 않는다. Target storage나 Milvus deployment를 비교할
때는 동일한 artifact를 재사용한다.

## Integrity and Current Limitations

Exporter는 비어 있지 않은 output directory를 덮어쓰지 않는다. Replayer는 collection을
생성하거나 replay하기 전에 artifact를 검증한다.

`export_vectors.py`는 현재 corpus, query와 생성된 vector를 memory에 올린 뒤 shard를 작성한다.
Wikipedia/Natural Questions preparer는 Hugging Face input streaming을 지원하지만, exporter
자체는 아직 TB 규모 streaming이나 incremental embedding을 제공하지 않는다.
