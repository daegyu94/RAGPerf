# DISKANN filesystem experiments

이 디렉터리는 같은 Milvus trace artifact를 `xfs`, `3fs`, `pnfs` filesystem에 바꿔
replay하는 실험 matrix와 진행 순서를 정의합니다. Staging과 VM 준비는
[Staged remote replay](../../docs/staged_remote_replay.md), workload 생성은
[Vector workloads](../../docs/vector_workloads.md)를 따릅니다.
모든 topology는 같은 `/mnt/nvme/milvus-data` data path를 사용하고
`expected_fstype`만 backend에 맞게 검증합니다.

## Matrix preset

[`diskann-experiments.yaml`](../../configs/evaluation/diskann-experiments.yaml)에 세
단계가 있습니다.

| Preset | Workload | Time scale | Repeat | 목적 |
| --- | --- | --- | ---: | --- |
| `smoke` | checked-in GPU smoke | 1 | 1 | topology, mount, offline runtime, result 회수 |
| `capacity` | 0.5/1/2/4TB | 1 | 3 | dataset 크기별 index/replay 특성 |
| `arrival` | 0.5/1TB | 0.5/1/2/4 | 3 | open-loop arrival 민감도 |

기본 반복 순서는 workload → backend → time scale → repeat입니다. 따라서 한 workload의
세 filesystem을 연속 비교한 뒤 다음 크기로 이동합니다. 각 case는 새 run directory와
새 Milvus data directory를 사용합니다.

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
  --repeats 1 \
  --run-tag capacity-preflight-20260819
```

성공한 뒤 full capacity matrix를 3회 반복합니다.

```bash
bash milvus_trace/benchmarks/evaluation/run_diskann_experiments.sh \
  --preset capacity \
  --run-tag capacity-20260819
```

2TB/4TB가 mount capacity나 experiment window를 넘으면 `--workloads 0.5tb,1tb`로
명시하고 제외 이유를 matrix plan과 결과 노트에 남깁니다.

## 3. Arrival matrix

Capacity 결과가 안정된 뒤 같은 0.5TB/1TB artifact에 time scale만 바꿉니다.

```bash
bash milvus_trace/benchmarks/evaluation/run_diskann_experiments.sh \
  --preset arrival \
  --run-tag arrival-20260819
```

`time_scale=2`는 recorded query offset을 절반으로 줄입니다. Model/GPU compute를 더 빠른
system처럼 재현한다는 의미는 아니며 storage request arrival만 바뀝니다.

## 비교 통제 항목

- 모든 backend에서 archive SHA-256과 manifest SHA-256이 동일해야 합니다.
- 동일한 Milvus/client version, CPU/memory limit, index/search parameter를 사용합니다.
- `replay_meta_root`는 같은 local storage class에 두고 DISKANN mount와 분리합니다.
- 각 case는 새 run directory를 사용하고 Compose를 내린 뒤 다음 case를 시작합니다.
- Script는 host page cache를 drop하지 않습니다. cold/warm cache 정책은 사전에 정하고
  세 backend에 같은 운영 절차를 적용합니다.
- 3FS/pNFS의 node 수, replication, stripe, mount option과 network link를 결과에 남깁니다.
- 실패 case는 정상 표본에 포함하지 않고 `remote_exit_code`, Milvus failure log,
  scheduler lag와 request failures를 함께 확인합니다.

Primary metric과 bootstrap/timed replay 경계는 기존
[Benchmark 방법론](../../docs/benchmark_methodology.md)을 그대로 사용합니다. 추가로
`run-metadata.txt`의 mount identity와 disk byte를 case metadata와 함께 보관합니다.
