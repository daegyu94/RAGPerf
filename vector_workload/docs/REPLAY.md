# Replay a Vector Workload

Replay는 검증된 artifact의 initial corpus를 Milvus에 적재하고 DISKANN index를 만든 뒤, artifact에
기록된 search/insert schedule을 실행해 result JSON을 생성한다. Embedding 생성 시간은 replay
측정에 포함되지 않는다.

```text
verified artifact
       ↓
initial insert → DISKANN build → load → warm-up → schedule replay
       ↓
result JSON
```

Artifact 생성 방법은 [Record Guide](RECORD.md), local Milvus 실행 방법은
[Milvus Docker Setup](DOCKER_MILVUS.md)을 참고한다.

## Requirements

- `export_vectors.py verify`를 통과한 완전한 artifact directory
- 접근 가능한 Milvus endpoint와 DISKANN이 활성화된 QueryNode
- Project-local `.venv`에 설치된 `pyarrow`, `PyYAML`, `numpy`, `pymilvus`

Docker Compose 예시를 사용했다면 기본 endpoint는 `http://127.0.0.1:19530`이다.

## Run Replay

Artifact를 다시 검증한 뒤 새 collection과 result path로 실행한다.

```bash
RUN_DIR=/MNTPNT/ragperf/workload-001

python vector_workload/export_vectors.py verify \
  --artifact-dir "$RUN_DIR/artifact"

python vector_workload/replay_milvus.py \
  --artifact-dir "$RUN_DIR/artifact" \
  --uri http://127.0.0.1:19530 \
  --collection ragperf_workload_001 \
  --result-file "$RUN_DIR/replay-result.json"
```

Replayer는 기존 result file이나 Milvus collection을 덮어쓰지 않는다. 반복 실행에는 새 이름을
사용한다. `default`는 collection 이름으로 사용할 수 없다.

## Important Options

| Option | Default | Purpose |
| --- | --- | --- |
| `--index-type` | `DISKANN` | 생성할 vector index |
| `--metric` | `COSINE` | Search distance metric |
| `--search-list` | `100` | DISKANN search candidate 범위 |
| `--top-k` | `10` | Query당 반환할 결과 수 |
| `--warmup-queries` | `100` | 측정 전에 실행할 query 수 |
| `--concurrency` | `8` | 동시에 실행할 search 수 |
| `--insert-batch-size` | `10000` | Initial insert batch row 수 |
| `--consistency-level` | `Strong` | Scheduled insert visibility 기준 |

Artifact의 query가 100개보다 적으면 warm-up은 전체 query 수로 제한된다. Smoke 검증에서 warm-up을
제외하려면 `--warmup-queries 0`을 지정한다. Query의 `delay_ms`를 적용하려면
`--respect-delay`를 추가한다. 기본 동작은 delay를 무시하고 가능한 높은 부하를 만든다.

ColPali multi-vector artifact는 일반적으로 `--metric IP`와 충분한 `--token-top-k`를 사용한다.
동일한 비교 실험에서는 metric, search list, top-k, warm-up과 concurrency를 고정한다.

## Replay Sequence

Replayer는 다음 순서로 동작한다.

1. Checksum, manifest와 Parquet row count를 검증한다.
2. 새 collection을 만들고 initial corpus를 batch insert한 뒤 flush한다.
3. DISKANN index를 생성하고 collection을 load한다.
4. 지정한 수의 query를 warm-up으로 실행한다.
5. Schedule 순서대로 search와 scheduled insert를 실행한다.
6. Result JSON을 기록한다.

Schedule이 없는 artifact는 query shard 순서대로 search-only workload를 실행한다. Mixed
schedule에서는 insert 이전에 진행 중인 search를 완료하므로 이후 search가 새 row를 볼 수 있다.

## Result JSON

Result에는 다음 항목이 포함된다.

- Milvus client/server version, endpoint, database와 collection
- Row 수, vector layout, index type과 metric
- Initial insert 시간과 처리량
- Index build 시간
- Warm-up 및 measured replay 시간
- Query 수, scheduled insert event/row 수와 QPS
- Latency min, mean, p50, p90, p95, p99, max
- Synthetic expected neighbor가 있으면 top-1 recall

결과 해석과 backend 비교 규칙은 [Benchmark Methodology](BENCHMARK_METHODOLOGY.md)를 따른다.
`--storage-path-note`는 결과에 target path를 기록하는 metadata이며 mount나 Milvus 설정을
변경하지 않는다.
