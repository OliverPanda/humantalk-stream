#!/usr/bin/env bash
# Compatibility entry point: hardware-aware native installation (Linux first).
# Install system packages listed in README.md before running this script.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$PROJECT_ROOT/install.sh" --install "$@"
