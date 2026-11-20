#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-auto}"
if [[ "$TARGET" == "auto" ]]; then
    # Recommend conservatively using GPU0 (the default inference device).
    MEMORY="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
    MEMORY="${MEMORY//[[:space:]]/}"
    if [[ "$MEMORY" =~ ^[0-9]+$ ]] && (( MEMORY >= 6144 )); then
        TARGET=gpu
    else
        TARGET=local
    fi
fi
case "$TARGET" in
    local|cpu|gpu|all) ;;
    *) echo "用法: bash scripts/build_docker.sh [auto|local|gpu|all]" >&2; exit 1 ;;
esac
docker build -f "$PROJECT_ROOT/Dockerfile" -t freevideoai:web "$PROJECT_ROOT"
if [[ "$TARGET" == gpu || "$TARGET" == all ]]; then
    docker build -f "$PROJECT_ROOT/docker/Dockerfile.gpu" -t freevideoai:gpu "$PROJECT_ROOT"
    echo "启动：docker compose -f compose.yaml -f compose.gpu.yaml up -d"
else
    echo "启动：docker compose up -d（在设置页配置远程数字人 API）"
fi
