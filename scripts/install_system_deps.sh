#!/usr/bin/env bash
# Use temporary apt sources; do not rewrite the server's system configuration.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export FREEVIDEOAI_MIRROR_CONFIG="${FREEVIDEOAI_MIRROR_CONFIG:-$PROJECT_ROOT/config/install-mirrors.env}"
. "$PROJECT_ROOT/scripts/install_sources.sh"
if (( EUID != 0 )); then
    echo "请使用 sudo bash scripts/install_system_deps.sh" >&2
    exit 1
fi
. /etc/os-release
APT_OPTIONS=()
if [[ -n "${FREEVIDEOAI_APT_MIRROR:-}" ]]; then
    SOURCES_FILE="$(mktemp)"
    trap 'rm -f "$SOURCES_FILE"' EXIT
    case "$ID" in
        ubuntu)
            printf 'deb %s/ubuntu %s main restricted universe multiverse\n' \
                "$FREEVIDEOAI_APT_MIRROR" "$VERSION_CODENAME" \
                "$FREEVIDEOAI_APT_MIRROR" "$VERSION_CODENAME-updates" \
                "$FREEVIDEOAI_APT_MIRROR" "$VERSION_CODENAME-security" > "$SOURCES_FILE"
            ;;
        debian)
            printf 'deb %s/debian %s main\n' \
                "$FREEVIDEOAI_APT_MIRROR" "$VERSION_CODENAME" \
                "$FREEVIDEOAI_APT_MIRROR" "$VERSION_CODENAME-updates" > "$SOURCES_FILE"
            printf 'deb %s/debian-security %s-security main\n' \
                "$FREEVIDEOAI_APT_MIRROR" "$VERSION_CODENAME" >> "$SOURCES_FILE"
            ;;
        *) echo "仅自动配置 Ubuntu / Debian 系统依赖。" >&2; exit 1 ;;
    esac
    chmod 644 "$SOURCES_FILE"
    APT_OPTIONS=(-o "Dir::Etc::sourcelist=$SOURCES_FILE" -o "Dir::Etc::sourceparts=-")
fi
apt-get "${APT_OPTIONS[@]}" update
apt-get "${APT_OPTIONS[@]}" install -y curl ca-certificates git build-essential \
    ffmpeg libsndfile1 libgl1 libglib2.0-0 fonts-noto-cjk python3-pip
