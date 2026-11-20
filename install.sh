#!/usr/bin/env bash
# Linux entry point. uv manages Python; application dependencies are not needed to preview.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! command -v uv >/dev/null 2>&1; then
    if [[ -x "${HOME}/.local/bin/uv" ]]; then
        export PATH="${HOME}/.local/bin:${PATH}"
    else
        command -v curl >/dev/null 2>&1 || {
            echo "请先安装 curl：sudo apt-get install -y curl ca-certificates" >&2
            exit 1
        }
        UV_INSTALLER="$(mktemp)"
        trap 'rm -f "$UV_INSTALLER"' EXIT
        curl --fail --location --retry 3 https://astral.sh/uv/install.sh -o "$UV_INSTALLER"
        UV_NO_MODIFY_PATH=1 UV_INSTALL_DIR="${HOME}/.local/bin" sh "$UV_INSTALLER"
        rm -f "$UV_INSTALLER"
        trap - EXIT
        export PATH="${HOME}/.local/bin:${PATH}"
    fi
fi
export FREEVIDEOAI_UV="$(command -v uv)"
exec uv run --no-project --python 3.11 --with filelock "$PROJECT_ROOT/scripts/install.py" "$@"
