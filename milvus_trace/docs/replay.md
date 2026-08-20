# 재생(Replay)

Replayer는 [trace artifact](artifact_format.md#용어)의 bootstrap corpus와 timed Milvus
요청을 하나의 새 target collection에 재생합니다. Text, Image, Audio 모두 같은 CLI를 사용하며
record host의 dataset, model, GPU는 필요하지 않습니다.

처음 실행한다면 [빠른 시작](../README.md#4-artifact-재생)의 한 번짜리 replay를 먼저
완료한 뒤 이 문서에서 timing과 concurrency 옵션을 조정하는 순서를 권장합니다.

## 실행 전 확인

- `verify_artifact()`로 artifact가 `incomplete: false`인지, checksum과 shard row 수가 모두
  일치하는지 확인합니다.
- Target Milvus는 이미 실행 중이어야 합니다. Replayer는 client만 제공하며 server를 설치하거나
  시작하지 않습니다. 처음에는 record와 replay에 같은 standalone endpoint를 사용해도 됩니다.
- Staged remote replay의 표준 경로에서는 target을 replay VM의 Docker Compose standalone
  Milvus로 준비합니다. `prepare-replay`가 pre-staged image를 load하고, replay phase의
  runner가 server를 시작해 health check를 통과한 뒤 Python replayer를 실행합니다.
- Target에 collection/index 생성, insert, flush, load, search/query 권한이 있고 recorded vector
  dimension, metric, index type을 지원하는지 확인합니다. `--collection`은 아직 존재하지 않는
  새 이름이어야 합니다.
- 실행 방식에 따라 준비합니다. Python CLI는
  [`requirements-replay.lock`](../docker/requirements-replay.lock)을 설치하면 되며 RAGPerf
  pipeline, dataset/model, GPU와 monitoring system은 필요하지 않습니다. Docker replay는
  local Python package 없이 [Docker 문서](docker.md)의 image를 사용합니다.

## Artifact 검증

Milvus에 연결하지 않고 artifact만 확인하려면 다음 명령을 실행합니다.

```bash
python -c "from milvus_trace.artifact import verify_artifact; m = verify_artifact('/path/to/artifact'); print(m['format'], m['event_count'])"
```

## 기본 replay

```bash
export MNTPNT=/path/to/ragperf-data/run-001
export REPLAY_MILVUS_URI=http://target-milvus.example:19530
export MILVUS_TOKEN=root:Milvus

python -m milvus_trace.replay \
  --artifact-dir "$MNTPNT/artifacts/text" \
  --uri "$REPLAY_MILVUS_URI" \
  --token "$MILVUS_TOKEN" \
  --collection replay_text_run_001 \
  --result-file "$MNTPNT/results/text-run-001.json"
```

`--token`을 생략하면 `root:Milvus`를 사용합니다. 인증을 사용하지 않는 endpoint에서도
설치된 `pymilvus`와 server 설정에 맞는 token 값을 지정합니다.

CLI는 성공 시 별도의 summary를 stdout에 출력하지 않습니다. 결과를 확인하려면
`--result-file`을 지정한 뒤 다음처럼 읽습니다.

```bash
python -m json.tool "$MNTPNT/results/text-run-001.json"
```

## 실행 순서

Replayer는 한 번의 `run()`에서 다음 순서를 지킵니다.

1. `workload-manifest.yaml`, `SHA256SUMS`, shard checksum과 row 수를 검증합니다.
2. Recorded dimension, auto-ID, consistency level로 target collection을 생성합니다.
3. `corpus-*.parquet`을 `--bootstrap-batch-size` 단위로 insert하고 flush합니다.
4. Recorded index metadata가 있으면 index를 생성하고 collection을 load합니다.
5. `--warmup`을 지정했으면 첫 recorded search payload를 반복 제출합니다.
6. Timed event를 sequence 순서와 선택한 timing으로 open-loop 제출합니다.
7. 모든 in-flight 요청 완료를 기다린 뒤 result JSON을 기록합니다.

Replayer는 artifact event의 원래 collection 이름을 target `--collection`으로 치환합니다.
기존 target collection을 drop하거나 실패 시 자동 정리하지 않습니다.

## CLI 옵션

| 옵션 | 기본값 | 의미 |
| --- | ---: | --- |
| `--artifact-dir` | 필수 | 압축을 푼 artifact directory |
| `--uri` | 필수 | Target Milvus URI |
| `--collection` | 필수 | 새 target collection 이름 |
| `--token` | `root:Milvus` | Target 인증 token |
| `--timing` | `original` | `original` 또는 `none` |
| `--time-scale` | `1.0` | Recorded offset을 나눌 배속. 0보다 커야 함 |
| `--max-in-flight` | `1024` | 동시에 완료되지 않은 timed request 상한 |
| `--bootstrap-batch-size` | `1024` | Bootstrap corpus insert batch row 수 |
| `--warmup` | `0` | 첫 search payload를 timed replay 전에 반복할 횟수 |
| `--result-file` | 없음 | Result JSON 경로. Parent directory는 자동 생성 |

현재 설치된 코드의 전체 option은 다음 명령으로 확인합니다.

```bash
python -m milvus_trace.replay --help
```

## Timing과 배속

기본 `--timing original --time-scale 1`은 recorded marker 이후의 monotonic offset을
그대로 사용합니다. Bootstrap insert, index build, load, warm-up이 끝난 시점부터 timed
timeline이 새로 시작합니다.

2배 빠르게 재생하면 각 offset을 2로 나눕니다.

```bash
python -m milvus_trace.replay \
  --artifact-dir "$MNTPNT/artifacts/text" \
  --uri "$REPLAY_MILVUS_URI" \
  --collection replay_text_2x_run_001 \
  --time-scale 2 \
  --result-file "$MNTPNT/results/text-2x-run-001.json"
```

도착 간격을 무시하고 가능한 빠르게 제출하려면 `--timing none`을 사용합니다.

```bash
python -m milvus_trace.replay \
  --artifact-dir "$MNTPNT/artifacts/text" \
  --uri "$REPLAY_MILVUS_URI" \
  --collection replay_text_no_timing_run_001 \
  --timing none \
  --result-file "$MNTPNT/results/text-no-timing-run-001.json"
```

`time-scale`은 Milvus 요청 사이의 도착 간격만 바꿉니다. Record 당시의 GPU/model
계산량을 target에서 다시 실행하거나 비례 조정하지 않습니다. 결과 비교 방법은
[benchmark 방법론](benchmark_methodology.md)을 참조합니다.

## Concurrency와 overload

Timed request는 이전 요청 완료를 기다리지 않는 open-loop 방식으로 thread pool에
제출됩니다. 아직 완료되지 않은 future 수가 `--max-in-flight`에 도달하면 replayer는
대기하지 않고 즉시 다음 오류로 실패합니다.

```text
max-in-flight (N) reached at event SEQUENCE
```

이는 scheduler가 target 처리량에 맞춰 arrival pattern을 조용히 늦추는 것을 방지합니다.
오류가 나면 scheduler lag와 target latency를 확인하고, 실험 의도에 따라
`--max-in-flight`를 늘리거나 별도의 target collection으로 더 낮은 `--time-scale`에서
다시 실행합니다.

## Warm-up

`--warmup N`은 artifact에서 만나는 첫 `search` event의 payload와 parameter를 N회
동기적으로 제출합니다. Warm-up은 bootstrap/index/load 이후, timed timeline 시작 전에
실행됩니다.

Warm-up request는 `replay.events`, `operation_counts`, latency histogram에 포함되지
않고 `bootstrap.warmup_requests`에만 기록됩니다. Artifact에 search event가 없으면
요청을 제출하지 않고 0을 기록합니다.

## Result JSON 해석

| Field | 의미 |
| --- | --- |
| `artifact.format` | Artifact format identifier |
| `artifact.manifest_sha256` | 사용한 manifest의 identity |
| `target.*` | URI, collection, 감지 가능한 client/server version |
| `bootstrap.rows` | Bootstrap corpus insert row 수 |
| `bootstrap.insert_and_flush_ns` | Bootstrap insert와 flush의 전체 시간 |
| `bootstrap.index_and_load_ns` | Index 생성과 collection load 시간 |
| `bootstrap.warmup_requests` | 실제 제출한 warm-up 수 |
| `replay.events` | 제출한 timed event 수 |
| `replay.elapsed_ns` | Timed timeline 시작부터 모든 request 완료까지의 시간 |
| `replay.throughput_ops_per_second` | `events / elapsed` |
| `replay.operation_counts` | Operation별 제출 수 |
| `replay.latency` | Operation별 client call latency histogram |
| `replay.scheduler_lag` | 목표 제출 시각 대비 실제 제출 지연 histogram |
| `replay.maximum_in_flight` | 관측된 최대 미완료 request 수 |
| `replay.failures` | Timed client call 오류 문자열 |
| `replay.timing`, `time_scale` | 이번 실행의 timing 설정 |

Latency와 scheduler lag는 nanosecond 단위이며 `min_ns`, `mean_ns`, `p50_ns`,
`p95_ns`, `p99_ns`, `max_ns`를 포함합니다. Histogram은 개별 sample을 보존하지 않는
bounded summary입니다.

`operation_counts`는 완료 성공 수가 아니라 제출 수입니다. Timed client call이 하나라도
실패하면 result file을 먼저 기록한 뒤 `ReplayError`로 process를 실패시키므로
`replay.failures`와 함께 해석합니다.

## 실패 후 처리

Artifact 검증, collection 생성, bootstrap, index, warm-up 또는 timed replay 중 오류가
나면 target collection이 부분적으로 남을 수 있습니다. Replayer는 자동 rollback이나
drop을 수행하지 않습니다.

1. 오류와 result file이 생성되었는지 확인합니다.
2. Target collection 상태를 확인합니다.
3. 사용자가 명시적으로 부분 collection을 정리하거나 새 이름을 선택합니다.
4. 같은 artifact와 수정한 option으로 다시 실행합니다.

| 오류 | 확인할 항목 |
| --- | --- |
| `target collection already exists` | 새 collection 이름 사용 또는 사용자가 기존 대상 정리 |
| `artifact does not contain a vector dimension` | 새 source collection으로 artifact를 다시 record했는지 확인 |
| `checksum mismatch` | Artifact 복사/압축 해제 손상 여부 확인 |
| `max-in-flight ... reached` | Target 처리량, 배속, in-flight 상한 확인 |
| Milvus index 오류 | Target server가 recorded index/metric을 지원하는지 확인 |

Docker 실행 문제는 [Docker 문서](docker.md), workload별 명령은
[workload 예시](workloads.md)를 참조합니다.
