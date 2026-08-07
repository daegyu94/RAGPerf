# Benchmark 방법론

Milvus trace replay의 목적은 RAG workload가 만든 storage request payload와 arrival
pattern을 분리해 target Milvus의 처리 특성을 측정하는 것입니다. End-to-end RAG
benchmark나 model 성능 측정을 대체하지 않습니다.

## 측정 경계

```text
Record timeline

model load ── query embedding ── search ── generation ── next embedding ── search
               └────────────── recorded arrival gaps ────────────────────────┘

Replay timeline

              search ──────────────── search
              └── 같은 payload와 time-scaled arrival offset
```

Timed marker는 workload model load가 끝난 뒤 첫 query embedding 직전에 설정됩니다.
따라서 첫 Milvus event의 offset에는 첫 query embedding 또는 Audio ASR 시간이 포함됩니다.
이후 offset에는 다음 Milvus 호출 전까지 발생한 retrieval, reranking, generation 등
upstream 지연이 포함됩니다.

Replay가 실행하는 것은 저장된 Milvus request뿐입니다. GPU computation을 다시
실행하거나 그 계산의 resource 사용량을 target host에서 재현하지 않습니다.

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

## Streaming과 memory

Record와 replay는 전체 dataset, event, latency sample을 하나의 Python list에 모으지
않습니다.

- Recorder는 byte 상한이 있는 queue와 row 상한이 있는 Parquet shard를 사용합니다.
- Replayer는 event를 batch iterator로 읽고 payload shard를 하나씩 cache합니다.
- Latency는 개별 sample 대신 bounded logarithmic histogram에 집계합니다.
- Timed concurrency는 `max-in-flight`로 제한합니다.

이 설계는 artifact 크기와 무관한 순차 처리를 목표로 합니다. RSS를 고정된 정확한
상한으로 보장한다는 의미는 아니며, 현재 payload reader는 한 Parquet shard를 memory에
load하므로 `rows_per_shard`, vector dimension과 scalar 크기가 peak memory에 영향을
줍니다.

## 재현 가능한 비교를 위한 체크리스트

- 같은 artifact와 checksum identity를 사용합니다.
- Target마다 새 collection을 만들고 이전 run의 cache/collection 상태를 기록합니다.
- Milvus server/client version, index type/parameter와 hardware 구성을 남깁니다.
- `timing`, `time_scale`, `max-in-flight`, `bootstrap-batch-size`, `warmup`을 함께
  기록합니다.
- Result의 scheduler lag, maximum in-flight, failures를 latency와 함께 보고합니다.
- Container replay라면 network mode와 replay host resource도 동일하게 유지합니다.
