# Milvus trace benchmarks

이 디렉터리는 `lmcache-tracebench`와 같은 recorder → staged replayer → evaluation
흐름으로 Milvus/DISKANN 실험 진입점을 정리합니다. Artifact 형식과 일반 replay CLI는
상위 [`milvus_trace`](../README.md)가 단일 출처입니다.

## Directory

```text
benchmarks/
├── recorder/
│   └── record_vector_workload.py       # GPU synthetic vector artifact
├── replayer/
│   ├── build_offline_bundle.sh         # wheelhouse + OCI image bundle
│   ├── staged_remote_replay.sh         # controller → isolated VM
│   └── run_diskann_replay.sh           # one verified filesystem run
└── evaluation/
    ├── run_diskann_experiments.py      # workload/backend/repeat matrix
    ├── run_diskann_experiments.sh      # project venv entrypoint
    └── README.md                       # experiment plan
```

## 시작 위치

| 목적 | 문서 |
| --- | --- |
| 0.5TB, 1TB 등 vector artifact 생성 | [Vector workloads](../docs/VECTOR_WORKLOADS.md) |
| 외부망 없는 replay VM 준비와 단일 실행 | [Staged remote replay](../docs/STAGED_REMOTE_REPLAY.md) |
| xfs/3FS/pNFS matrix와 측정 순서 | [Evaluation plan](evaluation/README.md) |

모든 Python 명령은 project virtual environment에서 실행합니다. Shell wrapper가 있는
matrix runner는 환경을 자동으로 활성화합니다.
