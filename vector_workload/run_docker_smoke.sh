#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
expected_venv="${repo_root}/.venv"

if [[ "${VIRTUAL_ENV:-}" != "${expected_venv}" ]]; then
    echo "error: activate the project-local .venv before running this script" >&2
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    echo "error: Docker Engine is unavailable to the current user" >&2
    exit 1
fi

run_id="$(date -u +%Y%m%d%H%M%S)-$$"
container_name="ragperf-milvus-smoke-${run_id}"
volume_name="${container_name}-data"
tmp_root="$(realpath "${TMPDIR:-/tmp}")"
run_dir="$(mktemp -d "${tmp_root}/ragperf-vector-smoke.XXXXXX")"
case "${run_dir}" in
    "${tmp_root}"/ragperf-vector-smoke.*) ;;
    *)
        echo "error: unexpected temporary directory: ${run_dir}" >&2
        exit 1
        ;;
esac

container_created=0
volume_created=0
cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [[ ${container_created} -eq 1 ]]; then
        docker rm -f "${container_name}" >/dev/null 2>&1 || true
    fi
    if [[ ${volume_created} -eq 1 ]]; then
        docker volume rm "${volume_name}" >/dev/null 2>&1 || true
    fi
    if [[ -d "${run_dir}" && "${run_dir}" == "${tmp_root}"/ragperf-vector-smoke.* ]]; then
        rm -rf -- "${run_dir}"
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

docker volume create "${volume_name}" >/dev/null
volume_created=1
docker run -d \
    --name "${container_name}" \
    --security-opt seccomp=unconfined \
    -e ETCD_USE_EMBED=true \
    -e ETCD_DATA_DIR=/var/lib/milvus/etcd \
    -e ETCD_CONFIG_PATH=/milvus/configs/embedEtcd.yaml \
    -e COMMON_STORAGETYPE=local \
    -e DEPLOY_MODE=STANDALONE \
    -v "${volume_name}:/var/lib/milvus" \
    -v "${script_dir}/docker/embed-etcd.yaml:/milvus/configs/embedEtcd.yaml:ro" \
    -v "${script_dir}/docker/milvus-user.yaml:/milvus/configs/user.yaml:ro" \
    -p 127.0.0.1::19530 \
    --health-cmd "curl -f http://localhost:9091/healthz" \
    --health-interval 5s \
    --health-start-period 60s \
    --health-timeout 10s \
    --health-retries 12 \
    milvusdb/milvus:v2.6.18 \
    milvus run standalone >/dev/null
container_created=1

for _ in $(seq 1 60); do
    state="$(docker inspect --format '{{.State.Status}}' "${container_name}")"
    health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{end}}' "${container_name}")"
    if [[ "${state}" == "running" && "${health}" == "healthy" ]]; then
        break
    fi
    if [[ "${state}" == "exited" || "${state}" == "dead" ]]; then
        docker logs "${container_name}" >&2
        echo "error: Milvus exited before becoming healthy" >&2
        exit 1
    fi
    sleep 2
done

if [[ "${health}" != "healthy" ]]; then
    docker logs "${container_name}" >&2
    echo "error: Milvus did not become healthy within 120 seconds" >&2
    exit 1
fi

artifact_dir="${run_dir}/artifact"
result_file="${run_dir}/result.json"
collection="ragperf_docker_smoke_${run_id//-/_}"
host_port="$(docker inspect --format '{{(index (index .NetworkSettings.Ports "19530/tcp") 0).HostPort}}' "${container_name}")"
milvus_uri="http://127.0.0.1:${host_port}"

python "${script_dir}/export_vectors.py" export \
    --corpus-file "${script_dir}/examples/corpus.jsonl" \
    --query-file "${script_dir}/examples/queries.jsonl" \
    --output-dir "${artifact_dir}" \
    --smoke \
    --device cpu \
    --initial-corpus-ratio 0.5
python "${script_dir}/export_vectors.py" verify --artifact-dir "${artifact_dir}"
python "${script_dir}/replay_milvus.py" \
    --artifact-dir "${artifact_dir}" \
    --uri "${milvus_uri}" \
    --collection "${collection}" \
    --result-file "${result_file}" \
    --warmup-queries 0 \
    --concurrency 1

python -c '
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
assert result["database"]["server_version"] == "2.6.18"
assert result["database"]["index_type"] == "DISKANN"
assert result["database"]["rows"] == 8
assert result["replay"]["queries"] == 4
assert result["replay"]["inserted_rows"] == 4
database = result["database"]
replay = result["replay"]
print(
    "Docker smoke test passed: "
    "Milvus {}, {} rows, {} queries".format(
        database["server_version"], database["rows"], replay["queries"]
    )
)
' "${result_file}"
