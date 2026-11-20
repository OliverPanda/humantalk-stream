"""Shared cross-process locking and atomic writes for user configuration."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import yaml
from filelock import FileLock


def config_lock(destination: Path) -> FileLock:
    """Lock the stable sidecar across the entire read/modify/write transaction."""
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(destination) + ".lock")


def write_config(destination: Path, data: dict) -> None:
    """Atomically replace YAML; the caller must hold config_lock(destination)."""
    fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            yaml.safe_dump(
                data, stream, allow_unicode=True,
                default_flow_style=False, sort_keys=False,
            )
        if destination.exists():
            shutil.copy2(destination, destination.with_suffix(".yaml.bak"))
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
