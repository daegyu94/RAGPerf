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

## Persistent Workload Export and Replay

Repository에 포함된 [embedded etcd 설정](../docker/embed-etcd.yaml)과
[DISKANN 설정](../docker/milvus-user.yaml)을 [Compose file](../docker/compose.yaml)과 함께
사용한다. Named volume은 container를 다시 만들어도 Milvus collection과 index를 유지한다.

### 1. Start Milvus

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

### 2. Export the Workload Artifact

이 프로젝트에서 workload record는 application API traffic을 capture하는 과정이 아니라,
corpus와 query를 portable artifact로 export하는 단계다. Input과 output path만 필수이며 나머지는
기본값을 사용할 수 있다.

```bash
RUN_DIR=/MNTPNT/ragperf/workload-001

python vector_workload/export_vectors.py export \
  --corpus-file /DATASET/corpus.jsonl \
  --query-file /DATASET/queries.jsonl \
  --output-dir "$RUN_DIR/artifact"

python vector_workload/export_vectors.py verify \
  --artifact-dir "$RUN_DIR/artifact"
```

기본 embedding device는 `cuda:0`이다. GPU가 없는 preparation host에서는 `--device cpu`를
추가한다. Mixed search/insert workload가 필요하면 initial corpus 일부만 먼저 적재하도록
`--initial-corpus-ratio`를 `1.0`보다 작게 지정한다.

### 3. Replay on Milvus

Milvus가 `healthy`이면 검증한 artifact를 replay한다. Collection과 result path는 실행마다 새
값을 사용한다.

```bash
python vector_workload/replay_milvus.py \
  --artifact-dir "$RUN_DIR/artifact" \
  --uri http://127.0.0.1:19530 \
  --collection ragperf_workload_001 \
  --result-file "$RUN_DIR/replay-result.json"
```

Replayer의 기본값은 `DISKANN`, `COSINE`, warm-up query 100개와 concurrency 8이다. Artifact에
query가 100개보다 적다면 `--warmup-queries 0` 또는 더 작은 값을 지정한다. 전체 workload 옵션은
[Workflow Guide](WORKFLOWS.md)를 참고한다.

### 4. Stop or Clean Up

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
