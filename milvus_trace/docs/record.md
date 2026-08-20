# 기록(Record)

Recorder는 RAGPerf의 Milvus client wrapper에서 실제 client 호출 직전의 인자를
캡처합니다. 요청 thread는 disk write를 기다리지 않고 bounded queue에 기록을 넘기며,
background writer가 Parquet shard와 manifest를 만듭니다.

처음 실행한다면 먼저 [빠른 시작](../README.md#3-실제-workload-기록)을 따라 Text
artifact 하나를 만든 뒤 이 문서에서 세부 동작을 확인하는 순서를 권장합니다.

## 실행 전 확인

- RAGPerf 전체 의존성과 workload별 dataset/model dependency를 설치하고,
  `src/monitoring_sys/libmsys*.so`를 build합니다. `src/run_new.py` 실행에 필요합니다.
- Source Milvus는 실행 중인 standalone 또는 distributed endpoint여야 하며, collection 생성,
  insert, index 생성, search/query 권한이 필요합니다. RAGPerf가 서버를 시작하지 않으므로
  처음에는 standalone 서버 한 대를 사용하면 됩니다.
- Artifact를 저장할 충분한 disk 공간을 확보합니다. Vector와 scalar metadata가 Parquet에
  저장되므로 corpus가 커질수록 artifact도 커집니다.

Record host에서 repository root와 `src`를 import path에 추가합니다.

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

## Run마다 새로 정할 값

이 문서의 Text record 예시는 `config/milvus_trace_text.yaml`을 사용합니다. Image와 Audio는 각각
`config/milvus_trace_image.yaml`, `config/milvus_audio.yaml`(작은 확인은
`config/milvus_audio_smoke.yaml`)으로 바꿉니다. `src/run_new.py --config`에 전달한 파일이
이번 record의 실제 workload config입니다.

각 run에서 다음 두 config 값을 새로 정합니다.

1. `sys.vector_db.collection_name`: Source Milvus에 아직 없는 collection 이름
2. `sys.vector_db.trace.output_dir`: 이번 run의 trace artifact directory

Text run의 최소 예시는 다음과 같습니다. `${MNTPNT}`는 아래 환경 변수에서 확장됩니다.

```yaml
sys:
  vector_db:
    collection_name: ragperf_trace_text_run_001
    trace:
      output_dir: ${MNTPNT}/artifacts/text
```

예시 config를 직접 수정하거나 run 전용 복사본을 만들어 위 두 값만 바꿉니다. Recorder는
`trace.output_dir`에 파일이 하나라도 있으면 `FileExistsError`로 시작을 거부하므로,
`MNTPNT`도 run 전용 경로로 지정하고 이전 artifact directory를 재사용하지 않습니다.

## 공통 환경 변수

위 YAML config의 `${MNTPNT}`, `${MILVUS_URI}`, `${RAG_DEVICE}`, `${GENERATION_DEVICE}`를
실행 전에 shell 환경 변수로 설정합니다. `MSYS_CONFIG`는 YAML 변수라기보다
`--msys-config` CLI option에 전달하는 monitoring 설정 파일 경로입니다.

```bash
export MNTPNT=/path/to/ragperf-data/run-001
export MILVUS_URI=http://source-milvus.example:19530
export RAG_DEVICE=cuda:0
export GENERATION_DEVICE=cuda:1
export MSYS_CONFIG=config/monitor/example_config.yaml
```

RAGPerf는 `--config`로 읽은 YAML의 `${...}`를 시작 시점에 확장합니다. 설정되지 않은
변수가 남아 있으면 dataset이나 model을 load하기 전에
`config contains unset environment variable(s)` 오류로 종료합니다.

## Trace 설정

`sys.vector_db.trace`에서 recorder를 활성화합니다.

```yaml
sys:
  vector_db:
    type: milvus
    trace:
      enabled: true
      output_dir: ${MNTPNT}/artifacts/text
      max_queue_bytes: 268435456
      rows_per_shard: 65536
      compression: zstd
      on_overflow: invalidate
```

| 설정 | 기본값 | 의미 |
| --- | ---: | --- |
| `enabled` | `false` | `true`일 때 recorder 생성 |
| `output_dir` | 빈 문자열 | manifest와 Parquet shard를 기록할 directory. 활성화 시 필수 |
| `max_queue_bytes` | `268435456` | 요청 thread와 writer 사이 queue의 byte 상한 |
| `rows_per_shard` | `65536` | Parquet shard 하나의 최대 row 수 |
| `compression` | `zstd` | PyArrow에 전달할 Parquet compression codec |
| `on_overflow` | `invalidate` | 현재 유일한 지원값. Source 요청은 계속하고 artifact는 불완전 처리 |

기본값을 사용할 때는 `enabled`와 `output_dir`만 지정해도 됩니다. 알 수 없는 trace
설정이나 0 이하의 queue/shard 크기는 실행 초기에 거부됩니다.

## 기록 구간

Record는 collection을 준비하는 `bootstrap` 단계와 실제 요청 도착 간격을 남기는 `timed`
단계로 나뉩니다. `index create`와 `model load` 자체는 trace event가 아니라 두 단계의
경계를 정하는 작업입니다.

```text
Record 시작
  │
  ├─ Bootstrap (timed 아님)
  │    corpus insert ── index create ── model load
  │    └─ corpus-*.parquet
  │
  ├─ marker (relative offset = 0)
  │
  └─ Timed workload
       first query embedding/ASR ── search/query/insert ...
       ├─ events-*.parquet       (순서·offset·event parameter)
       └─ searches-*.parquet / inserts-*.parquet / scalar-queries-*.parquet (실제 payload)
```

- **Bootstrap corpus**: Index가 생성되지 않았고 timed marker도 설정되지 않은 동안의
  insert입니다. Replay target collection을 처음 채우는 데 사용되며 timed event 수와
  도착 간격에는 포함되지 않습니다. 보통 `corpus-*.parquet`에 저장됩니다.
- **Marker**: timed workload의 시각 기준점입니다. Text/Image는 model load 후 첫 query
  embedding 직전에, Audio는 query encoder load 후 첫 audio batch embedding 직전에
  설정합니다. Marker 자체가 Milvus 요청을 보내지는 않습니다.
- **첫 event**: Marker 뒤의 embedding/ASR 시간이 지나 첫 search가 호출되므로 첫 search의
  `relative_timestamp_ns`에는 그 시간이 포함됩니다. 이후 event 간격에는 retrieval,
  reranking, generation 등 다음 Milvus 호출 전의 upstream 시간이 포함됩니다.
- **Marker를 직접 설정하지 않은 integration**: 첫 search/query/timed insert를 recorder가
  만나는 시점에 marker를 자동으로 만들고, 그 event의 offset을 0으로 기록합니다.

Recorder가 재현하는 것은 Milvus 요청 payload, 제출 순서와 도착 간격입니다. GPU kernel,
model execution 자체, Milvus response/result는 trace artifact에 저장하지 않습니다.

## 저장하는 데이터

| Operation | Payload shard | Event metadata | 저장하지 않는 값 |
| --- | --- | --- | --- |
| Bootstrap insert | `corpus-*.parquet`의 vector와 JSON scalar/text | 없음 | insert parameter, Milvus response |
| Timed insert | `inserts-*.parquet`의 vector와 JSON scalar/text | `events-*.parquet`의 순서·offset·parameter | Milvus response |
| Search | `searches-*.parquet`의 query vector | `events-*.parquet`의 순서·offset·limit/filter/output/search parameter | query 원문, search result |
| Query | `scalar-queries-*.parquet`의 filter와 query parameter | `events-*.parquet`의 순서·offset | query result |

NumPy scalar/array처럼 `tolist()` 또는 `item()`으로 변환 가능한 값은 JSON 값으로
정규화합니다. JSON으로 표현할 수 없는 객체가 parameter에 들어오면 Milvus 요청을
보내기 전에 `TypeError`로 실패합니다.

## Workload 실행

| Workload | Config | 기본 artifact directory |
| --- | --- | --- |
| Text | `config/milvus_trace_text.yaml` | `$MNTPNT/artifacts/text` |
| Image | `config/milvus_trace_image.yaml` | `$MNTPNT/artifacts/image` |
| Audio | `config/milvus_audio.yaml` | `$MNTPNT/artifacts/audio` |

실제 명령, model/GPU 요구 사항, 작은 run을 만드는 설정은
[workload별 예시](workloads.md)를 참조합니다.

## 완료 확인

Manifest와 `SHA256SUMS`는 recorder가 정상적으로 닫힐 때 생성됩니다. Process를
`SIGKILL`로 종료하거나 host가 중단되면 이 파일이 없거나 불완전할 수 있습니다.

```bash
python -c "from milvus_trace.artifact import verify_artifact; m = verify_artifact('$MNTPNT/artifacts/text'); print('events:', m['event_count'], 'operations:', m['operation_counts'])"
```

성공하면 event 수와 operation별 수가 출력됩니다. 정상 artifact의 예시는 다음과
같습니다. 실제 file 종류와 shard 수는 workload에 따라 달라집니다.

```text
workload-manifest.yaml
SHA256SUMS
corpus-00000.parquet
events-00000.parquet
searches-00000.parquet
scalar-queries-00000.parquet   # query가 있을 때만 생성
inserts-00000.parquet          # timed insert가 있을 때만 생성
```

`verify_artifact()`는 format, `incomplete`, checksum, manifest에 기록된 shard row 수를
검사합니다. 출력 파일의 의미는 [artifact 형식](artifact_format.md)을 참조합니다.

## 주의 사항

- Queue overflow나 writer 오류가 발생해도 source Milvus workload는 계속 진행될 수 있습니다.
  Recorder는 `incomplete_reason`을 남기고 `incomplete: true` artifact를 만든 뒤
  `TraceRecordingError`를 발생시키며, `verify_artifact()`와 replayer는 이를 거부합니다.
  부분 artifact를 replay하지 말고, Queue overflow 후에는 `max_queue_bytes`를 늘린 뒤
  새 artifact directory와 source collection으로 다시 기록합니다.
- Artifact에는 insert한 text와 scalar metadata가 평문 JSON/Parquet로 저장될 수 있습니다.
  API token은 manifest에 저장하지 않지만 text, file path, document identifier 등 민감정보가
  포함될 수 있으므로 공유 전 payload 정책을 확인합니다. Audio 원본 bytes와 query transcript
  보존 범위는 [Audio 문서](audio.md)를 참조합니다.
