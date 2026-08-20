# Docker image 빌드와 실행

## Image의 역할

`milvus_trace/docker/Dockerfile`은 replay 전용 image를 만듭니다. 이 image에는
`pymilvus`, `pyarrow`, `numpy`, `PyYAML`, lock file의 transitive dependency와 replay
code만 들어 있습니다. Milvus
server, RAGPerf pipeline, dataset loader, `torch`, embedding/ASR model, CUDA runtime은
포함하지 않습니다.

record는 source RAG workload의 model load와 embedding/generation timing을 관찰해야
하므로 RAGPerf가 설치된 record host에서 실행합니다. record artifact를 복사한 뒤
replay image로 target Milvus에 재생하는 구조입니다.

이 image는 Milvus server를 시작하지 않습니다. 실행 전에 standalone Milvus endpoint를
별도로 준비하고, `--uri`가 replayer container에서 접근 가능한 주소를 가리키게 합니다.
첫 테스트에서는 standalone 서버 하나를 source와 target에 함께 사용할 수 있으며,
replay에는 source collection과 다른 새 collection 이름을 사용합니다.

repository root에서 `./milvus_trace/scripts/milvus-standalone.sh start`를 실행하면 이 repository가
제공하는 standalone helper로 테스트용 Milvus를 준비할 수 있습니다. helper가 시작하는
Milvus server와 replay image는 별개의 Docker image입니다.

```text
RAGPerf record host ── artifact directory/tar.zst ──> replay host
       │                                                   │
       └── source Milvus                         Docker replayer ──> target Milvus
```

## Local image 빌드

repository root에서 실행합니다.

```bash
export REPLAYER_IMAGE=ragperf-milvus-replayer:local
docker build \
  -f milvus_trace/docker/Dockerfile \
  -t "$REPLAYER_IMAGE" .
```

사설 CA가 필요한 network에서는 certificate를 BuildKit secret으로 전달합니다.
certificate는 dependency stage에서만 사용하며 final image에는 복사되지 않습니다.

```bash
export CA_CERT=/path/to/ca.crt
docker build \
  --secret id=corp_ca,src="$CA_CERT" \
  -f milvus_trace/docker/Dockerfile \
  -t "$REPLAYER_IMAGE" .
```

## Image 확인

```bash
docker run --rm "$REPLAYER_IMAGE" --help
docker run --rm --entrypoint python "$REPLAYER_IMAGE" -c \
  "import importlib.util as u; assert u.find_spec('torch') is None"
```

두 번째 명령은 replay image에 `torch`가 설치되지 않았는지 확인합니다.

## Network 선택

Milvus와 replayer가 같은 user-defined Docker network에 있으면 service name을 URI로
사용합니다.

```bash
export DOCKER_NETWORK=milvus-network
export REPLAY_MILVUS_URI=http://milvus:19530
export MILVUS_TOKEN=root:Milvus
```

외부 cluster endpoint를 사용한다면 container에서 접근 가능한 DNS 이름과 port를
`REPLAY_MILVUS_URI`에 지정합니다. Linux host의 localhost Milvus에 연결하기 위해
host network를 사용할 수 있지만, 이는 platform별 동작이 다르므로 기본 예시로
가정하지 않습니다.

## Artifact mount와 replay

artifact는 read-only로, result directory는 writable로 mount합니다.

```bash
export MNTPNT=/path/to/ragperf-data
export WORKLOAD=text
mkdir -p "$MNTPNT/results"
docker run --rm --network "$DOCKER_NETWORK" \
  -v "$MNTPNT/artifacts/$WORKLOAD:/artifact:ro" \
  -v "$MNTPNT/results:/output" \
  "$REPLAYER_IMAGE" \
  --artifact-dir /artifact \
  --uri "$REPLAY_MILVUS_URI" \
  --token "$MILVUS_TOKEN" \
  --collection "replay_${WORKLOAD}" \
  --result-file "/output/${WORKLOAD}.json"
```

`WORKLOAD`에는 `text`, `image`, `audio`를 사용할 수 있습니다. workload별 완전한
명령은 [workload 예시](workloads.md)를 참조하십시오.

## tar.zst release asset 사용

archive를 container에 직접 전달하지 않고 먼저 검증 가능한 directory로 풉니다.

```bash
cd "$MNTPNT/release"
sha256sum -c rag-run-001.tar.zst.sha256
mkdir -p "$MNTPNT/release/unpacked"
tar --zstd -xf "$MNTPNT/release/rag-run-001.tar.zst" \
  -C "$MNTPNT/release/unpacked"
```

압축 해제된 artifact directory를 앞의 `/artifact:ro` 위치에 mount합니다.

## GHCR publish

`.github/workflows/milvus-replayer-image.yml`은 다음 경우 image를 build/push합니다.

- `milvus-replayer-v*` tag push
- GitHub Actions의 manual `workflow_dispatch`

publish된 image 이름은 workflow의 `images` 값으로 결정됩니다. fork나 다른 organization에서
사용할 때는 해당 값을 자신의 GHCR namespace로 변경합니다.

## 흔한 오류

- `target collection already exists`: 새 `--collection` 이름을 사용하거나 사용자가
  명시적으로 기존 collection을 정리합니다. replayer가 임의로 삭제하지 않습니다.
- `connection refused`/DNS 오류: `REPLAY_MILVUS_URI`가 container network에서
  접근 가능한지 확인합니다.
- `artifact checksum mismatch`: 복사 또는 압축 해제 과정이 손상되었습니다. 원본
  archive의 `.sha256`을 먼저 확인합니다.
- permission 오류: artifact mount는 readable, result mount는 UID 10001 사용자가
  쓸 수 있어야 합니다.
