# Milvus Docker Setup

이 문서는 RAGPerf vector workload를 replay하기 위한 Milvus Standalone을 Docker로 실행하는
방법을 설명한다. 아래 조합은 실제 smoke test로 검증했다.

| Component | Validated value |
| --- | --- |
| Milvus image | `milvusdb/milvus:v2.6.18` |
| Python client | `pymilvus 2.6.17` |
| Deployment | Standalone with embedded etcd and local storage |
| Index and metric | DISKANN and COSINE |
| Host requirement | Docker Engine에 접근 가능한 사용자 |

## Automated Smoke Test

Project-local environment와 dependency를 준비하고 repository root에서 실행한다.

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r vector_workload/requirements.txt
docker version
bash vector_workload/run_docker_smoke.sh
```

Runner는 다음 작업을 자동으로 수행한다.

1. 빈 localhost port와 임시 Docker volume을 할당한다.
2. DISKANN이 활성화된 Milvus `2.6.18` container를 시작하고 health check를 기다린다.
3. CPU smoke artifact를 생성하고 checksum과 Parquet shard를 검증한다.
4. Initial insert, DISKANN index build, query와 scheduled insert를 실행한다.
5. Server version, index type, row/query/insert 수를 검증한다.
6. 성공 여부와 관계없이 container, volume과 임시 artifact를 정리한다.

성공하면 마지막에 다음 형식의 메시지가 출력된다.

```text
Docker smoke test passed: Milvus 2.6.18, 8 rows, 4 queries
```

Runner는 host의 기존 `19530` port와 충돌하지 않도록 임시 port를 사용한다. Test 결과는 검증
직후 정리되므로 실제 workload replay에는 아래의 persistent deployment를 사용한다.

## Persistent Milvus Deployment

Repository에 포함된 [embedded etcd 설정](../docker/embed-etcd.yaml)과
[DISKANN 설정](../docker/milvus-user.yaml)을 [Compose file](../docker/compose.yaml)과 함께
사용한다. Named volume은 container를 다시 만들어도 Milvus collection과 index를 유지한다.

```bash
docker compose \
  --project-name ragperf-vector \
  --file vector_workload/docker/compose.yaml \
  up --detach --wait
```

`127.0.0.1` binding은 Milvus port를 외부 network에 공개하지 않는다. Remote client가 필요하면
host firewall, 인증과 network policy를 먼저 구성한 뒤 binding을 명시적으로 변경한다.

Health 상태와 log를 확인한다.

```bash
docker compose \
  --project-name ragperf-vector \
  --file vector_workload/docker/compose.yaml \
  ps

docker compose \
  --project-name ragperf-vector \
  --file vector_workload/docker/compose.yaml \
  logs --tail 100 milvus

curl --fail http://127.0.0.1:9091/healthz
```

`healthy`가 확인되면 [Record Guide](RECORD.md)에서 만든 artifact를
[Replay Guide](REPLAY.md)에 따라 `http://127.0.0.1:19530`으로 제출한다.

## Stop or Clean Up

Container를 내리더라도 named volume의 Milvus data는 보존된다. 같은 Compose command로 다시
시작하면 기존 data를 사용한다.

```bash
docker compose \
  --project-name ragperf-vector \
  --file vector_workload/docker/compose.yaml \
  down
```

Milvus data까지 영구 삭제하려면 `--volumes`를 추가한다. 이 작업은 복구할 수 없으므로 더 이상
필요하지 않은 collection과 index인지 먼저 확인한다.

```bash
docker compose \
  --project-name ragperf-vector \
  --file vector_workload/docker/compose.yaml \
  down --volumes
```

일반적인 Linux 환경에서 Docker 설치와 daemon 설정에는 `root` 또는 `sudo` 권한이 필요할 수
있다. 이후 명령은 현재 사용자가 Docker socket에 접근할 수 있으면 `sudo` 없이 실행할 수 있다.
`docker` group membership은 root 수준 권한을 제공하므로 host 보안 정책에 맞게 설정한다.
