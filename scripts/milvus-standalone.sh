#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/milvus_trace/docker/milvus-standalone-compose.yml"
COMPOSE_PROJECT="${MILVUS_COMPOSE_PROJECT:-ragperf-milvus}"
MILVUS_DATA_DIR="${MILVUS_DATA_DIR:-$REPO_ROOT/.milvus/volumes}"
MILVUS_VERSION="${MILVUS_VERSION:-v2.4.15}"
MILVUS_HOST_PORT="${MILVUS_HOST_PORT:-19530}"
MILVUS_WEBUI_PORT="${MILVUS_WEBUI_PORT:-9091}"
MILVUS_WAIT_SECONDS="${MILVUS_WAIT_SECONDS:-120}"

export COMPOSE_PROJECT MILVUS_DATA_DIR MILVUS_VERSION MILVUS_HOST_PORT MILVUS_WEBUI_PORT

usage() {
    cat <<EOF
Usage: $(basename "$0") <command>

Commands:
  start   Download required images and start standalone Milvus.
  stop    Stop containers and keep Milvus data.
  down    Remove containers and keep Milvus data.
  status  Show container status.
  logs    Follow standalone Milvus logs.

Environment overrides:
  MILVUS_VERSION       Milvus image tag (default: $MILVUS_VERSION)
  MILVUS_DATA_DIR      Persistent data directory (default: $MILVUS_DATA_DIR)
  MILVUS_HOST_PORT     Host gRPC port (default: $MILVUS_HOST_PORT)
  MILVUS_WEBUI_PORT    Host WebUI port (default: $MILVUS_WEBUI_PORT)
  MILVUS_COMPOSE_PROJECT
                       Docker Compose project name (default: $COMPOSE_PROJECT)

The Docker image cache is managed by Docker. Inspect its location with:
  docker info --format '{{.DockerRootDir}}'
EOF
}

require_docker() {
    if ! command -v docker >/dev/null 2>&1; then
        echo "Docker CLI was not found. Install Docker Engine/Desktop first." >&2
        exit 1
    fi
    if ! docker compose version >/dev/null 2>&1; then
        echo "Docker Compose v2 is required (docker compose)." >&2
        exit 1
    fi
    if ! docker info >/dev/null 2>&1; then
        echo "Docker daemon is not running or the current user cannot access it." >&2
        exit 1
    fi
}

compose() {
    docker compose \
        --project-name "$COMPOSE_PROJECT" \
        --file "$COMPOSE_FILE" \
        "$@"
}

print_locations() {
    local docker_root
    docker_root="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)"
    echo "Milvus version: $MILVUS_VERSION"
    echo "Persistent data: $MILVUS_DATA_DIR"
    if [[ -n "$docker_root" ]]; then
        echo "Docker image/container storage: $docker_root"
    fi
    echo "Milvus endpoint: http://localhost:$MILVUS_HOST_PORT"
    echo "Milvus WebUI: http://localhost:$MILVUS_WEBUI_PORT/webui/"
}

wait_for_port() {
    local deadline=$((SECONDS + MILVUS_WAIT_SECONDS))
    echo "Waiting for Milvus on localhost:$MILVUS_HOST_PORT ..."
    while (( SECONDS < deadline )); do
        if (echo >/dev/tcp/127.0.0.1/"$MILVUS_HOST_PORT") >/dev/null 2>&1; then
            echo "Milvus is accepting connections."
            return 0
        fi
        sleep 2
    done

    echo "Milvus did not open port $MILVUS_HOST_PORT within ${MILVUS_WAIT_SECONDS}s." >&2
    compose ps >&2 || true
    compose logs --tail=100 standalone >&2 || true
    return 1
}

main() {
    local command="${1:-}"

    if [[ "$command" == "-h" || "$command" == "--help" || -z "$command" ]]; then
        usage
        [[ -n "$command" ]] || exit 1
        exit 0
    fi

    require_docker

    case "$command" in
        start)
            mkdir -p "$MILVUS_DATA_DIR/etcd" "$MILVUS_DATA_DIR/minio" "$MILVUS_DATA_DIR/milvus"
            print_locations
            compose up -d
            wait_for_port
            compose ps
            ;;
        stop)
            compose stop
            ;;
        down)
            compose down
            ;;
        status)
            compose ps
            print_locations
            ;;
        logs)
            compose logs -f standalone
            ;;
        *)
            echo "Unknown command: $command" >&2
            usage >&2
            exit 2
            ;;
    esac
}

main "$@"
