# DISKANN filesystem experiments

이 디렉터리는 같은 Milvus trace artifact를 `xfs`, `3fs`, `pnfs` filesystem에 바꿔
replay하는 실험 matrix와 진행 순서를 정의합니다. Staging과 replay node 준비는
[Staged remote replay](../../docs/staged_remote_replay.md), workload 생성은
[Vector workloads](../../docs/vector_workloads.md)를 따릅니다.
각 topology는 `replay_diskann_root`를 기본 경로로 사용합니다. 예제 topology의 기본값은
`/mnt/nvme/milvus-data`이며, 실행 시 `--diskann-root PATH`로 모든 backend의
DiskANN mount 경로를 한 번에 덮어쓸 수 있습니다. backend별로 다른 경로가 필요하면
각 topology 파일의 `replay_diskann_root`를 따로 지정합니다. `expected_fstype`는
backend에 맞게 검증합니다.

## Matrix preset

[`diskann-experiments.yaml`](../../configs/evaluation/diskann-experiments.yaml)에 세
단계가 있습니다.

| Preset | Workload | Time scale | 목적 |
| --- | --- | --- | --- |
| `smoke` | checked-in GPU smoke | 1 | topology, mount, offline runtime, result 회수 |
| `capacity` | 0.5/1/2/4TB | 1 | dataset 크기별 index/replay 특성 |
| `arrival` | 0.5/1TB | 0.5/1/2/4 | open-loop arrival 민감도 |

반복 순서는 workload → backend → time scale → repeat입니다. Repeat count는 preset에
저장하지 않으며, 모든 실행에서 `--repeats N`으로 선택합니다. 각 case는 새 run directory와
새 Milvus data directory를 사용합니다.

## 실행 옵션

Matrix를 실행하기 전에 이번 run에서 사용할 값을 지정합니다. `--repeats`는 필수이며
preset의 기본값이 없습니다. `--diskann-root`는 생략하면 topology의
`replay_diskann_root`를 사용하고, 지정하면 모든 backend에 같은 remote mount 경로를
적용합니다.

```bash
export REPEATS="${REPEATS:?set REPEATS to a positive integer}"
export DISKANN_ROOT="${DISKANN_ROOT:-/mnt/nvme/milvus-data}"
```

주요 선택 옵션은 다음과 같습니다.

- `--preset smoke|capacity|arrival`: matrix 모양을 선택합니다.
- `--repeats N`: 이 실행의 반복 횟수를 선택합니다.
- `--diskann-root PATH`: 모든 topology의 DiskANN mount 경로를 덮어씁니다.
- `--topology NAME=PATH`: backend별 topology 파일을 교체합니다. 반복 지정할 수 있습니다.
- `--workloads`, `--backends`, `--time-scales`: preset의 부분집합을 선택합니다.

## 0. 사전 용량 계산

0.5TB plan과 controller/replay node의 free space를 먼저 확인합니다.

```bash
source .venv/bin/activate
python -m milvus_trace.benchmarks.recorder.record_vector_workload \
  --preset 0.5tb --plan-only
```

최소 calibration artifact를 replay해 `run-metadata.txt`의 `du -sb`와 manifest의
`logical_vector_bytes` 비율을 기록합니다. 이 factor는 필요한 mount 용량과 예상 시간을
정하는 용도이며, backend별 artifact 크기를 다르게 만드는 데 사용하지 않습니다.

## 1. Smoke matrix

세 topology template의 host/path를 실제 값으로 바꾼 뒤 먼저 dry-run합니다.

```bash
bash milvus_trace/benchmarks/evaluation/run_diskann_experiments.sh \
  --preset smoke \
  --run-tag smoke-20260819 \
  --repeats "$REPEATS" \
  --diskann-root "$DISKANN_ROOT" \
  --skip-prepare \
  --dry-run
```

`--skip-prepare`를 제거하면 backend별 offline runtime과 smoke archive를 stage한 뒤 실제
replay를 실행합니다. Topology 파일을 repository 밖에서 관리한다면 반복 option으로
명시합니다.

```bash
bash milvus_trace/benchmarks/evaluation/run_diskann_experiments.sh \
  --preset smoke \
  --run-tag smoke-20260819 \
  --repeats "$REPEATS" \
  --diskann-root "$DISKANN_ROOT" \
  --topology xfs=/secure/topologies/xfs.yaml \
  --topology 3fs=/secure/topologies/3fs.yaml \
  --topology pnfs=/secure/topologies/pnfs.yaml
```

Runner는 `milvus_trace/outputs/diskann-experiments/<run-tag>`에 `matrix-plan.yaml`,
case marker, `matrix-summary.yaml`을 남깁니다. 성공한 case marker가 있으면 재실행 시
건너뜁니다. 의도적으로 교체할 때만 `--overwrite-output`을 사용합니다.

## 2. Capacity matrix

각 preset artifact를 [Vector workloads](../../docs/vector_workloads.md)의 명령으로
record/package하고 topology의 `controller_trace_root/vector`에 둡니다.

```text
vector/
├── 0.5tb.tar.zst
├── 0.5tb.tar.zst.sha256
├── 1tb.tar.zst
├── 1tb.tar.zst.sha256
└── ...
```

0.5TB 한 크기로 end-to-end 시간과 공간을 먼저 측정합니다.

```bash
bash milvus_trace/benchmarks/evaluation/run_diskann_experiments.sh \
  --preset capacity \
  --workloads 0.5tb \
  --repeats "$REPEATS" \
  --diskann-root "$DISKANN_ROOT" \
  --run-tag capacity-preflight-20260819
```

preflight 결과가 안정되면 원하는 반복 횟수로 full capacity matrix를 실행합니다.

```bash
bash milvus_trace/benchmarks/evaluation/run_diskann_experiments.sh \
  --preset capacity \
  --repeats "$REPEATS" \
  --diskann-root "$DISKANN_ROOT" \
  --run-tag capacity-20260819
```

2TB/4TB가 mount capacity나 experiment window를 넘으면 `--workloads 0.5tb,1tb`로
명시하고 제외 이유를 matrix plan과 결과 노트에 남깁니다.

## 3. Arrival matrix

Capacity 결과가 안정된 뒤 같은 0.5TB/1TB artifact에 time scale만 바꿉니다.

```bash
bash milvus_trace/benchmarks/evaluation/run_diskann_experiments.sh \
  --preset arrival \
  --repeats "$REPEATS" \
  --diskann-root "$DISKANN_ROOT" \
  --run-tag arrival-20260819
```

`time_scale=2`는 recorded query offset을 절반으로 줄입니다. Model/GPU compute를 더 빠른
system처럼 재현한다는 의미는 아니며 storage request arrival만 바뀝니다.

## 결과 기록

각 case의 matrix plan, command, remote exit code와 replay 결과는 runner의 state root와
topology의 `controller_output_root`에 저장합니다. 성공한 case만 결과 비교에 사용하고,
실패한 case는 원인과 함께 별도로 남깁니다. Primary metric과 bootstrap/timed replay
경계는 [Benchmark 방법론](../../docs/benchmark_methodology.md)을 따릅니다.
`run-metadata.txt`에는 선택한 mount path와 filesystem identity를 보관합니다.
