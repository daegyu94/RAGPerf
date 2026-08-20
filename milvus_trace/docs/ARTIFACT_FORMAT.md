# Artifact 형식

Milvus trace artifact는 한 번의 record run이 만든 self-contained directory bundle입니다.
이 문서에서 `artifact`는 이 trace artifact를 줄여 부르는 말이며, `.tar.zst`는 artifact를
전달하기 위한 archive입니다. Replayer는 directory 안의 `workload-manifest.yaml`에서
`format: ragperf-milvus-trace`를 확인한 뒤 checksum과 Parquet metadata를 검증합니다.

## 용어

| 용어 | 의미 |
| --- | --- |
| `trace` | Record session에서 발생한 Milvus 요청의 논리적 기록. bootstrap corpus와 timed event를 함께 포함합니다. |
| `trace artifact` | 한 번의 record 결과를 replay할 수 있도록 manifest, checksum, Parquet payload를 묶은 directory입니다. 문서에서 `artifact`라고 줄여 부르기도 합니다. |
| `bootstrap corpus` | Index 생성과 timed marker 이전에 source collection에 insert한 vector/scalar row입니다. Replay target의 초기 collection을 구성하는 데 사용하며 timed event 도착 간격에는 포함하지 않습니다. |
| `timed event` | Timed marker 이후의 search, query, timed insert 요청과 recorded relative offset입니다. |
| `artifact archive` | Trace artifact directory를 전달하기 위해 `tar.zst`로 패키징한 파일입니다. |

현재 형식에는 별도 schema version이 없습니다. Reader와 writer는 repository의 같은
format contract를 사용해야 하며, 호환되지 않는 변경에는 새로운 format identifier가
필요합니다.

## Directory 예시

```text
text-run-001/
├── workload-manifest.yaml
├── SHA256SUMS
├── corpus-00000.parquet
├── corpus-00001.parquet
├── searches-00000.parquet
├── scalar-queries-00000.parquet
└── events-00000.parquet
```

모든 workload에 모든 shard 종류가 생기지는 않습니다. 예를 들어 scalar `query`가 없는
Text workload에는 `scalar-queries-*`가 없고, timed insert가 없으면 `inserts-*`가
없습니다. `rows_per_shard`를 넘으면 suffix가 `00001`, `00002` 순서로 증가합니다.

## 파일 역할

| 파일 | 역할 |
| --- | --- |
| `workload-manifest.yaml` | Format, 완료 상태, collection/index metadata, event/file 목록 |
| `SHA256SUMS` | Manifest와 생성된 모든 Parquet shard의 SHA-256 |
| `corpus-*.parquet` | Index와 timed marker 이전의 bootstrap insert row |
| `inserts-*.parquet` | Timed insert payload |
| `searches-*.parquet` | Timed search vector |
| `scalar-queries-*.parquet` | Timed query filter와 parameter |
| `events-*.parquet` | Operation 순서, offset, parameter와 payload row reference |

Event row는 payload 전체를 복제하지 않고 `payload_refs_json`으로 payload shard의
`path`, `row_start`, `row_count`를 가리킵니다. Replayer는 event를 순서대로 읽고 필요한
shard만 load합니다.

## Manifest 주요 field

| Field | 의미 |
| --- | --- |
| `format` | `ragperf-milvus-trace` |
| `incomplete`, `incomplete_reason` | Queue/writer 오류로 정상 완료하지 못했는지와 첫 원인 |
| `timing_model` | `monotonic-open-loop-arrival` |
| `session_id` | Recorder process session identifier |
| `event_count` | Timed event 전체 수 |
| `operation_counts` | `insert`, `search`, `query`별 timed event 수 |
| `rows_per_shard`, `compression` | Writer 설정 |
| `collection` | Name, dimension, auto-ID, consistency level, vector field |
| `index` | Field/name, metric type, index type와 parameter |
| `vector` | Dimension과 `float32` dtype |
| `recorder.python` | Record에 사용한 Python version |
| `artifacts` | 각 shard의 kind, 상대 path, row 수, SHA-256 |

API token, Milvus response, source server URI/version은 manifest에 저장하지 않습니다.
Replay 대상 URI와 token은 실행할 때 별도로 지정합니다.

## Parquet schema

Vector는 typed `list<float32>` column으로 저장합니다. Scalar와 client parameter는
JSON string으로 직렬화해 workload마다 달라지는 dynamic field를 보존합니다.

### Corpus와 timed insert

| Column | Type | 의미 |
| --- | --- | --- |
| `vector` | `list<float32>` | Milvus에 전달한 한 row의 vector |
| `scalar_json` | `string` | `vector`를 제외한 insert row의 JSON |

### Search

| Column | Type | 의미 |
| --- | --- | --- |
| `vector` | `list<float32>` | Milvus search에 전달한 query vector |

Search의 `limit`, filter, output field, consistency level과 search parameter는 event의
`params_json`에 저장됩니다.

### Scalar query

| Column | Type | 의미 |
| --- | --- | --- |
| `request_json` | `string` | Filter, output field, limit 등 query request |

### Event

| Column | Type | 의미 |
| --- | --- | --- |
| `sequence` | `int64` | 0부터 증가하는 제출 순서 |
| `relative_timestamp_ns` | `int64` | Timed marker 이후 monotonic offset |
| `operation` | `string` | `insert`, `search`, `query` |
| `collection` | `string` | Record 당시 source collection |
| `batch_size` | `int64` | 참조 payload row 수 |
| `session_id` | `string` | Event를 만든 recorder session |
| `params_json` | `string` | Operation별 client parameter |
| `payload_refs_json` | `string` | Payload shard와 row 범위 목록 |

## 검증 계약

```bash
python -c "from milvus_trace.artifact import verify_artifact; print(verify_artifact('/path/to/artifact')['event_count'])"
```

`verify_artifact()`는 다음 조건을 모두 검사합니다.

1. Manifest가 mapping이고 format identifier가 일치합니다.
2. `incomplete`가 `false`입니다.
3. `SHA256SUMS` 형식이 유효하고 나열된 파일이 모두 존재합니다.
4. 나열된 파일의 SHA-256이 일치합니다.
5. Manifest에 나열된 Parquet shard의 실제 row 수와 개별 SHA-256이 일치합니다.

검증은 target collection을 만들기 전에 실행됩니다. `incomplete: true` artifact나 손상된
파일을 일부만 replay하는 option은 없습니다.

## 개인정보와 보안 경계

- Insert의 text와 scalar metadata는 artifact에 그대로 남습니다.
- Search vector는 남지만 그 vector를 만든 query 원문은 recorder가 받지 않으므로
  저장하지 않습니다.
- Query filter에는 document identifier나 조건식이 포함될 수 있습니다.
- File path 같은 model-specific scalar도 저장되지만 replayer가 해당 local file을 열지는
  않습니다.
- Artifact는 암호화되지 않습니다. 공유와 보관 시 dataset의 보안 정책을 적용합니다.

Audio의 구체적인 보존 범위는 [Audio 문서](AUDIO.md)를 참조합니다.

## Archive 패키징

`package_artifact`는 먼저 위 검증을 실행한 뒤 directory 이름을 보존하는 deterministic
`tar.zst`를 만듭니다. Parquet shard를 decode하거나 다시 작성하지 않습니다.

```bash
python -m milvus_trace.package_artifact \
  --artifact-dir /path/to/text-run-001 \
  --output /path/to/releases/text-run-001.tar.zst
```

Host에 `tar`와 `zstd`가 필요합니다. Output archive와 `.sha256`이 이미 있으면 덮어쓰지
않고 실패하며, output은 artifact directory 밖에 있어야 합니다. 생성 결과는 다음 두
파일입니다.

```text
text-run-001.tar.zst
text-run-001.tar.zst.sha256
```

압축을 푼 뒤에는 directory를 `--artifact-dir`로 지정해 일반 replay CLI를 그대로
사용합니다.
