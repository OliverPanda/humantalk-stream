#!/bin/sh
# Source this file; simple KEY=value data, never execute the configuration file.
if [ "${FREEVIDEOAI_DOWNLOAD_SOURCE:-china}" != official ]; then
    while IFS= read -r source_line || [ -n "$source_line" ]; do
        source_line="$(printf '%s' "$source_line" | tr -d '\r')"
        case "$source_line" in ''|\#*) continue ;; esac
        source_key="${source_line%%=*}"
        source_value="${source_line#*=}"
        case "$source_key" in *[!a-zA-Z0-9_]*|'') echo "Invalid mirror key" >&2; return 1 ;; esac
        # Docker build arguments without a value may be present but empty.
        if [ -z "$(printenv "$source_key" 2>/dev/null || true)" ]; then
            export "$source_key=$source_value"
        fi
    done < "$FREEVIDEOAI_MIRROR_CONFIG"
    unset source_line source_key source_value
fi
