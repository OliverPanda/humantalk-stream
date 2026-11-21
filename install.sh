#!/usr/bin/env bash
# Linux entry point. uv manages Python; application dependencies are not needed to preview.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FREEVIDEOAI_MIRROR_CONFIG="${FREEVIDEOAI_MIRROR_CONFIG:-$PROJECT_ROOT/config/install-mirrors.env}"
. "$PROJECT_ROOT/scripts/install_sources.sh"
if ! command -v uv >/dev/null 2>&1; then
    if [[ -x "${HOME}/.local/bin/uv" ]]; then
        export PATH="${HOME}/.local/bin:${PATH}"
    elif [[ -x "${HOME}/.local/share/freevideoai-uv/bin/uv" ]]; then
        export PATH="${HOME}/.local/share/freevideoai-uv/bin:${PATH}"
    elif command -v python3 >/dev/null 2>&1 && python3 -m pip --version >/dev/null 2>&1; then
        # uv's PyPI wheel contains its executable, so no GitHub release is needed.
        UV_BOOTSTRAP="${HOME}/.local/share/freevideoai-uv"
        python3 -m pip install --upgrade --target "$UV_BOOTSTRAP" uv
        UV_BINARY="$(PYTHONPATH="$UV_BOOTSTRAP" python3 -c 'import uv; print(uv.find_uv_bin())')"
        export PATH="$(dirname "$UV_BINARY"):$PATH"
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
