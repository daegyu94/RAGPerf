# Staged remote DISKANN replay

Staged remote replay는 외부망이 되는 controller에서 artifact, `milvus_trace` source,
Python wheel과 Milvus OCI image를 준비해 SSH로 격리 VM에 전달하는 방식입니다. Replay
VM은 GitHub, Hugging Face, PyPI, Docker Hub에 접속하지 않습니다.

```text
controller                                isolated replay VM

trace.tar.zst ───────────────┐          ┌─ apt Python/Docker
wheelhouse + OCI images ─────┼─ SSH ───>├─ pip --no-index / docker load
milvus_trace source ─────────┘          ├─ target Milvus + replayer
retrieved results <─────────────────────┤
                                       └─ /mnt/nvme/milvus-data
```

일반 replay option과 result schema는 [재생 가이드](REPLAY.md), 측정 경계는
[Benchmark 방법론](BENCHMARK_METHODOLOGY.md)이 단일 출처입니다. 이 문서는 격리 VM
staging과 DISKANN mount 교체만 설명합니다.

## 1. Replay VM의 apt 준비

Ubuntu/Debian VM에는 내부 apt mirror에서 다음 도구만 설치합니다.

```bash
sudo apt-get update
sudo apt-get install -y \
  bash coreutils findutils util-linux tar zstd rsync openssh-server \
  python3 python3-venv docker.io docker-compose-v2
```

배포판에 따라 Compose v2 package 이름은 다를 수 있습니다. `docker compose version`이
성공해야 하며 replay account는 Docker daemon과 DISKANN data directory에 접근할 수 있어야
합니다. Python package나 container image를 VM에서 내려받지 않습니다.

XFS, 3FS, pNFS 실험은 모두 `/mnt/nvme/milvus-data`를 DISKANN data path로
사용합니다. Backend를 바꿀 때는 이 경로를 받치는 filesystem을 바꾸고, script는
`findmnt -T`로 해당 경로의 실제 mount target, source, FSTYPE을 검사합니다.

```text
/mnt/nvme/milvus-data -> xfs
/mnt/nvme/milvus-data -> fuse.3fs (환경에 따라 fuse3fs/fuse)
/mnt/nvme/milvus-data -> nfs4 또는 nfs
```

## 2. Controller offline bundle

Controller에는 project virtual environment와 세 Milvus standalone image가 있어야
합니다. Image가 없으면 controller에서만 먼저 pull합니다.

```bash
docker pull quay.io/coreos/etcd:v3.5.5
docker pull minio/minio:RELEASE.2023-03-20T20-16-18Z
docker pull milvusdb/milvus:v2.4.15
```

Replay VM의 Python version/architecture와 맞는 wheelhouse를 만듭니다.

```bash
source .venv/bin/activate
bash milvus_trace/benchmarks/replayer/build_offline_bundle.sh \
  --output-dir /data/ragperf-offline-runtime \
  --include-milvus-images
```

다른 Python/platform용 cross-download는 `--python-version`, `--platform`, `--abi`를
함께 지정합니다. Bundle의 `SHA256SUMS`는 VM에서 install 전에 검증됩니다.
`prepare-replay`는 다음 명령만 사용합니다.

```text
python -m venv
pip install --no-index --find-links <wheelhouse>
docker load --input <milvus-images.tar>
```

Compose는 `--pull never`로 실행되므로 image가 빠지면 외부 pull 대신 즉시 실패합니다.

## 3. Topology

[`example.yaml`](../configs/replayer/staged-remote/example.yaml)을 backend별 실제 파일로
복사합니다. Repository에는 필드 구성을 보여주는 `xfs.yaml`, `3fs.yaml`, `pnfs.yaml`
template도 있습니다. 예시의 host와 `/absolute/path`는 반드시 바꿉니다.

| Key | 역할 |
| --- | --- |
| `controller_repo_root` | controller의 RAGPerf checkout; VM에는 `milvus_trace`만 전송 |
| `controller_trace_root` | category별 `.tar.zst` archive root |
| `controller_runtime_root` | wheelhouse/image bundle |
| `controller_output_root` | 회수한 run directory root |
| `replay_repo_root`, `replay_venv_root` | staged source와 offline venv |
| `replay_trace_root`, `replay_output_root` | 추출 trace와 remote result |
| `replay_meta_root` | etcd metadata용 local VM storage |
| `replay_diskann_root` | 실험 대상 filesystem 위의 DISKANN data directory |
| `storage_backend`, `expected_fstype` | result label과 mount 검증값 |
| `replay_milvus_uri` | VM에서 접근할 target Milvus URI |

`lmcache-tracebench`의 `weka01` profile과 같은 jump-user 접속은 다음처럼 지정합니다.

```yaml
replay_host: weka01
replay_jump_user: user
replay_user: daegyu
replay_port: 22
```

이 경우 SSH는 `user@weka01`로 접속하고 remote command와 rsync는
`sudo -n -u daegyu`로 실행합니다.

MinIO object와 Milvus local/index data는 `<replay_diskann_root>/<run-name>`에 두고,
etcd만 `<replay_meta_root>/<run-name>`에 둡니다. Docker image layer는 Docker root에
남으므로 measured DISKANN data와 섞이지 않습니다.

## 4. 단계별 실행

VM prerequisite를 먼저 검사합니다.

```bash
bash milvus_trace/benchmarks/replayer/staged_remote_replay.sh \
  check-prerequisites \
  --topology milvus_trace/configs/replayer/staged-remote/xfs.yaml
```

GPU smoke archive와 offline runtime을 각각 stage합니다.

```bash
bash milvus_trace/benchmarks/replayer/staged_remote_replay.sh \
  prepare-trace \
  --topology milvus_trace/configs/replayer/staged-remote/xfs.yaml \
  --asset vector/gpu-smoke.tar.zst

bash milvus_trace/benchmarks/replayer/staged_remote_replay.sh \
  prepare-replay \
  --topology milvus_trace/configs/replayer/staged-remote/xfs.yaml
```

단일 replay는 topology placeholder로만 path를 전달합니다.

```bash
bash milvus_trace/benchmarks/replayer/staged_remote_replay.sh replay \
  --topology milvus_trace/configs/replayer/staged-remote/xfs.yaml \
  --run-name gpu-smoke-xfs-r1 -- \
  bash @REPO_ROOT@/milvus_trace/benchmarks/replayer/run_diskann_replay.sh \
    --artifact-dir @TRACE_ROOT@/vector/gpu-smoke \
    --backend @STORAGE_BACKEND@ \
    --expected-fstype @EXPECTED_FSTYPE@ \
    --diskann-root @DISKANN_ROOT@ \
    --meta-root @META_ROOT@ \
    --output-dir @OUTPUT_ROOT@ \
    --run-name @RUN_NAME@ \
    --uri @MILVUS_URI@
```

`all` phase는 `prepare-trace`, `prepare-replay`, `replay`를 순서대로 실행합니다.
단일 runner는 TCP port 개방 뒤에도 Milvus client health check가 성공할 때까지 기다립니다.
`--dry-run`은 SSH/transfer/replay를 수행하지 않고 치환된 명령을 출력합니다.

## 5. Result와 재실행

Remote command의 성공/실패와 무관하게 output을 controller로 회수하고
`remote_exit_code`를 추가합니다. 한 DISKANN run의 주요 파일은 다음과 같습니다.

```text
<controller_output_root>/<run-name>/
├── replay-result.json
├── run-metadata.txt
└── remote_exit_code
```

`run-metadata.txt`에는 mount source/FSTYPE, Milvus version, 실행 전후 시각,
`du -sb`, Compose 상태가 있습니다. 새 run은 새 `run-name`을 사용합니다. 같은 이름을
의도적으로 재실행할 때만 `--overwrite-output`을 지정하면 그 run directory만
교체합니다.

`reset`은 controller를 건드리지 않습니다. `diskann` target은
`/mnt/nvme/milvus-data` directory 자체를 삭제하지 않고 `find -xdev`로 내용만
지웁니다. 실제 topology를 검토하고 dry-run한 뒤 benchmark 전용 directory에서만
사용합니다.

```bash
bash milvus_trace/benchmarks/replayer/staged_remote_replay.sh reset \
  --topology milvus_trace/configs/replayer/staged-remote/xfs.yaml \
  --target trace --target output --target diskann \
  --dry-run
```
