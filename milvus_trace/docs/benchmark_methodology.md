# Benchmark 방법론

Milvus trace replay의 목적은 RAG workload가 만든 storage request payload와 arrival
pattern을 분리해 target Milvus의 처리 특성을 측정하는 것입니다. End-to-end RAG
benchmark나 model 성능 측정을 대체하지 않습니다.

## 측정 경계

```text
Record timeline

model load ── upstream work ── Milvus event ── upstream work ── Milvus event
                              (insert/search/query)
               └──────────── recorded arrival gaps ────────────┘

Replay timeline

              insert/search/query ─── insert/search/query
              └── 같은 payload와 time-scaled arrival offset
```

Timed marker는 workload model load가 끝난 뒤 첫 query embedding 직전에 설정됩니다.
따라서 첫 Milvus event의 offset에는 첫 query embedding 또는 Audio ASR 시간이 포함됩니다.
이후 offset에는 다음 Milvus 호출 전까지 발생한 retrieval, reranking, generation 등
upstream 지연이 포함됩니다.

Replay가 실행하는 것은 저장된 Milvus request뿐입니다. GPU computation을 다시
실행하거나 그 계산의 resource 사용량을 target host에서 재현하지 않습니다.

## Replay되는 작업

위 그림의 `Milvus event`에는 다음 작업이 포함될 수 있습니다. `search`만 replay하는
것이 아닙니다.

| 구분 | 동작 | 결과에 기록되는 위치 |
| --- | --- | --- |
| Bootstrap | Recorded corpus를 target에 `insert`하고 `flush` | `bootstrap.insert_and_flush_ns` |
| Timed event | `insert`, vector `search`, scalar/filter `query`를 기록된 순서와 offset으로 제출 | `replay.operation_counts`와 `replay.latency`의 operation별 항목 |
| Warm-up | 옵션을 지정하면 첫 번째 `search` payload만 timed replay 전에 반복 | `bootstrap.warmup_requests`. Timed 결과에는 제외 |
| 미지원 | `delete`는 record/replay operation으로 지원하지 않음 | 해당 event가 있으면 `unsupported operation` 오류 |

Bootstrap의 corpus `insert`, `flush`, index 생성, collection load는 target을 준비하는
단계입니다. 이 작업들은 timed event의 arrival 간격이나 operation별 timed latency에
포함되지 않으므로, timed replay 결과와 따로 비교합니다.

Timed `query`는 자연어 질의가 아니라 Milvus `query()` 호출입니다. Recorded filter와
parameter가 함께 저장되며 replay 시 같은 요청으로 제출됩니다. 현재 timed operation은
`insert`, `search`, `query` 세 종류이므로, `delete`나 다른 Milvus API를 benchmark에
포함하려면 recorder/replayer 구현을 먼저 확장해야 합니다.

## Setup과 timed workload

Corpus embedding과 source corpus insert는 record setup으로 분류합니다. Replay는 이미
저장된 corpus vector를 target에 insert하므로 embedding 시간은 측정하지 않고 다음 값을
별도로 보고합니다.

| 구간 | Result field | 포함하는 작업 |
| --- | --- | --- |
| Bootstrap | `bootstrap.insert_and_flush_ns` | Recorded corpus의 target insert와 flush |
| Index setup | `bootstrap.index_and_load_ns` | Recorded index 생성과 collection load |
| Warm-up | `bootstrap.warmup_requests` | 선택한 첫 search payload 반복. Latency 결과에서 제외 |
| Timed replay | `replay.elapsed_ns` | 첫 event schedule 시작부터 모든 event 완료까지 |

Target 간 비교에서는 bootstrap/index 결과와 timed replay 결과를 합치지 말고 각각
비교합니다.

## Open-loop arrival

Replayer는 각 event의 목표 시각에 이전 request 완료 여부와 관계없이 제출합니다. 이는
target latency가 높아졌을 때 client가 자동으로 arrival rate를 낮추는 closed-loop 효과를
피하기 위한 방식입니다.

Target이 따라가지 못하면 in-flight request와 scheduler lag가 증가합니다.
`--max-in-flight`에 도달하면 arrival를 늦춰 계속하지 않고 replay를 실패시킵니다. 이
경우 결과를 정상 완료 run과 같은 표본으로 비교하지 않습니다.

## Time scale 해석

`--time-scale N`은 각 recorded relative offset을 N으로 나눕니다.

| 설정 | 의미 |
| --- | --- |
| `--time-scale 1` | 원래 arrival offset |
| `--time-scale 2` | Offset 절반, 약 2배 빠른 arrival |
| `--time-scale 0.5` | Offset 2배, 약 절반 속도의 arrival |
| `--timing none` | Offset 무시, client가 가능한 빠르게 제출 |

이 조정은 storage request 사이의 간격만 변경합니다. Record 당시 GPU/model 작업이
더 빠른 시스템에서 실제로 만들 arrival pattern과 동일하다고 단정할 수 없습니다.
배속 실험에서는 이 제한을 결과와 함께 명시합니다.

## Result를 함께 봐야 하는 이유

Throughput 하나만으로 target 상태를 판단하지 않습니다.

| 지표 | 해석 |
| --- | --- |
| `throughput_ops_per_second` | Timed event 수를 전체 replay elapsed로 나눈 값 |
| `latency.*` | Client가 관측한 operation call 시간 |
| `scheduler_lag` | 목표 제출 시각보다 실제 제출이 늦은 정도 |
| `maximum_in_flight` | Target 처리 중 겹친 request의 최대 수 |
| `failures` | Timed client call 오류. 비어 있어야 정상 run |

Latency가 높고 lag도 커지면 target 처리량뿐 아니라 replay host CPU scheduling, Python
thread pool, network, `max-in-flight`를 함께 확인합니다. Latency가 낮아도 lag가 크면
client host가 목표 arrival를 만들지 못했을 수 있습니다.

`operation_counts`는 성공 완료 수가 아니라 제출 수입니다. `failures`가 있으면 process가
실패하므로 성공한 run의 throughput/percentile과 직접 비교하지 않습니다.

## 데이터를 한꺼번에 메모리에 올리지 않는 방식

Record와 replay는 전체 dataset, 요청(event), latency sample을 하나의 Python list에
쌓아두지 않습니다. 대신 정해진 크기의 작은 단위로 읽고 쓰면서 처리합니다.

- Record: 요청은 `max_queue_bytes`까지 담을 수 있는 임시 대기열(queue)에 보관하고,
  payload는 `rows_per_shard` 행 단위의 Parquet 파일로 나눠 저장합니다.
- Replay: 요청 기록(event)은 batch 단위로 읽고, payload는 한 번에 하나의 Parquet
  shard만 메모리에 올려 처리합니다.
- Latency: latency sample을 전부 저장하지 않고, latency 구간별 개수(histogram)로
  요약합니다.
- 동시성: 동시에 처리 중인 request 수는 `max-in-flight`를 넘지 않습니다.

따라서 artifact 전체가 커져도 그 크기만큼 RAM을 계속 사용하는 구조는 아닙니다. 다만
프로세스가 실제로 사용하는 메모리(RSS)가 정확히 일정한 상한 이하로 고정된다는 보장은
없습니다. 현재 replay reader는 Parquet shard 하나를 메모리에 올리므로 peak memory는
`rows_per_shard`, vector dimension, scalar metadata 크기에 따라 달라집니다. Record
queue와 replay 동시 요청 수도 각각 `max_queue_bytes`와 `max-in-flight`의 영향을
받습니다.

## 재현 가능한 비교를 위한 체크리스트

성능을 비교할 때는 다음 조건과 결과를 함께 기록합니다.

1. 같은 artifact와 checksum을 사용합니다.
2. Target마다 새 collection을 만들고, 이전 run의 collection과 cache 상태를 기록합니다.
3. Milvus server/client version, index type/parameter, hardware 구성을 남깁니다.
4. `timing`, `time_scale`, `max-in-flight`, `bootstrap-batch-size`, `warmup`을
   함께 기록합니다.
5. latency와 함께 result의 scheduler lag, maximum in-flight, failures를 보고합니다.
6. Replay를 container에서 실행한다면 network mode와 replay host의 CPU/memory resource도
   동일하게 유지합니다.

이 정보를 남겨야 latency 차이가 Milvus 자체 때문인지 replay host와 실행 조건 때문인지
구분할 수 있습니다.
