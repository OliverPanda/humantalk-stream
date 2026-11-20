#!/bin/sh
set -eu
mkdir -p /app/config /app/workspace_data /app/output
if [ ! -f /app/config/user_config.yaml ]; then
    cp "${FREEVIDEOAI_DEFAULT_CONFIG:-/opt/freevideoai/default-user.cpu.yaml}" /app/config/user_config.yaml
fi
touch "${FREEVIDEOAI_ENV_FILE}"
chmod 600 "${FREEVIDEOAI_ENV_FILE}"
exec "$@"
