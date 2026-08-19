# Vector workload 기록

DISKANN filesystem 비교에는 두 종류의 trace를 구분합니다.

- 실제 Text/Image/Audio RAG의 요청 도착 패턴은 기존 [workload 예시](WORKLOADS.md)에
  따라 record합니다.
- 0.5TB, 1TB처럼 corpus 용량을 통제하는 실험은 이 문서의 deterministic synthetic
  vector recorder를 사용합니다.

둘은 같은 artifact 형식과 replayer를 사용하지만 결과의 의미가 다릅니다. Synthetic
trace는 embedding model 품질이나 실제 문서 분포를 대표하지 않으며, dataset 크기와
DISKANN storage I/O를 독립적으로 조절하기 위한 capacity workload입니다.

## Preset 정의

[`vector-workloads.yaml`](../configs/recorder/vector-workloads.yaml)은 768차원
`float32` vector를 기준으로 다음 strict decimal target을 제공합니다.

| Preset | Row 수 | Logical vector bytes |
| --- | ---: | ---: |
| `0.5tb` | 162,760,416 | 499,999,997,952 |
| `1tb` | 325,520,833 | 999,999,998,976 |
| `2tb` | 651,041,666 | 1,999,999,997,952 |
| `4tb` | 1,302,083,333 | 3,999,999,998,976 |

계산식은 `floor(target_bytes / (dimension * 4))`입니다. 여기서 TB는 `10^12` byte이고,
target을 한 row라도 넘지 않습니다. Scalar field를 넣지 않아 logical size가 vector
payload만 나타내도록 했습니다.

이 값은 Parquet archive 크기나 Milvus가 실제 filesystem에 쓴 byte가 아닙니다.
Parquet compression, MinIO object, segment, DISKANN index와 local cache 때문에 실제
사용량은 달라집니다. 실험 결과에서 logical size와 `du -sb`를 함께 기록합니다.

## 먼저 plan 확인

Artifact를 쓰지 않고 row 수와 예상량만 확인합니다.

```bash
source .venv/bin/activate
python -m milvus_trace.benchmarks.recorder.record_vector_workload \
  --preset 0.5tb \
  --plan-only
```

`--dimension`, `--rows`, `--query-count`, `--query-qps`, `--batch-size` 등은 CLI로
override할 수 있습니다. 비교할 artifact들은 dimension, seed, query 설정과 batch
size를 동일하게 유지합니다.

## GPU record

다음 명령은 CUDA에서 corpus/query vector를 batch 단위로 생성하고 CPU로 옮겨 기존
bounded queue와 Parquet shard writer에 전달합니다. CUDA에서는 CuPy를 사용하고,
CPU를 명시하면 Torch를 사용합니다.

```bash
source .venv/bin/activate
python -m milvus_trace.benchmarks.recorder.record_vector_workload \
  --preset 0.5tb \
  --device cuda:0 \
  --artifact-dir /data/ragperf-artifacts/vector/0.5tb
```

Corpus와 query는 별도 seed를 사용하므로 dataset 크기가 달라도 query vector는
동일합니다. Manifest에는 generator framework/version, GPU name, device, 두 seed,
logical byte가 기록됩니다. Recorder queue가 producer를 따라가지 못하면 artifact를
완료본으로 사용하지 말고 `producer_delay_ms`, `batch_size`, `max_queue_bytes`를 조정해
새 directory에 다시 기록합니다.

TB 단위 artifact는 corpus logical size와 비슷한 controller storage, archive를 위한
추가 공간, replay node의 추출 공간, target DISKANN mount 공간이 각각 필요합니다.
한 filesystem에 모두 배치하면 workload read와 target write가 섞이므로 trace archive는
DISKANN target과 다른 controller/replay-local storage에 둡니다.

## 실제 disk target으로 보정

같은 logical artifact를 모든 backend에 replay하는 비교가 기본입니다. 특정 disk budget
아래에서 최대 dataset을 만들 목적이라면 먼저 작은 calibration artifact를 한 backend에
replay하고 다음 factor를 구합니다.

```text
storage_overhead_factor = measured du -sb / manifest logical_vector_bytes
```

그 factor로 plan과 record를 수행하면 logical rows를 줄여 estimated disk byte가 target을
넘지 않게 계산합니다.

```bash
python -m milvus_trace.benchmarks.recorder.record_vector_workload \
  --preset 1tb \
  --storage-overhead-factor 1.25 \
  --plan-only
```

Factor는 Milvus version, index parameter와 filesystem에 따라 달라질 수 있습니다.
Filesystem 비교에서는 backend마다 다른 factor로 서로 다른 artifact를 만들지 말고, 한
artifact identity를 유지해야 합니다.

## Checked-in GPU smoke trace

[`examples/traces/gpu-smoke`](../examples/traces/gpu-smoke)는 현재 host의
`NVIDIA RTX PRO 4000 Blackwell`에서 CuPy 14.1.1로 생성한 실제 예시입니다.

- corpus: 256 × 32차원 `float32`
- timed search: 8개, 20 QPS schedule
- index metadata: `DISKANN` + `COSINE`
- generator seed: corpus `20260819`, query `20260820`

검증과 archive 위치는 다음과 같습니다.

```bash
python -c "from milvus_trace.artifact import verify_artifact; print(verify_artifact('milvus_trace/examples/traces/gpu-smoke')['event_count'])"

(cd milvus_trace/examples/archives/vector && \
  sha256sum -c gpu-smoke.tar.zst.sha256)
```

일반 artifact도 같은 layout으로 package합니다.

```bash
python -m milvus_trace.package_artifact \
  --artifact-dir /data/ragperf-artifacts/vector/0.5tb \
  --output /data/ragperf-trace-archives/vector/0.5tb.tar.zst
```

Archive 형식과 checksum 계약은 [Artifact 형식](ARTIFACT_FORMAT.md)이 단일 출처입니다.
