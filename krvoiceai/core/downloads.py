"""Download settings shared by bootstrap, native installers and Docker.

Only standard-library imports: this module must work before dependencies exist.
"""
from __future__ import annotations

import os
from pathlib import Path


def download_environment() -> dict[str, str]:
    """Return defaults overlaid by the caller's environment, without mutating it."""
    env = dict(os.environ)
    if env.get("FREEVIDEOAI_DOWNLOAD_SOURCE", "china") == "official":
        return env
    config = Path(env.get(
        "FREEVIDEOAI_MIRROR_CONFIG",
        str(Path(__file__).resolve().parents[2] / "config/install-mirrors.env"),
    ))
    with config.open(encoding="utf-8-sig") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if not key.isidentifier():
                raise ValueError(f"无效的下载配置字段: {key}")
            if not env.get(key):
                env[key] = value
    return env


def torch_index(cuda: str) -> str:
    base = download_environment().get(
        "FREEVIDEOAI_TORCH_INDEX", "https://download.pytorch.org/whl")
    return f"{base.rstrip('/')}/{cuda}"


def github_url(url: str) -> str:
    base = download_environment().get("FREEVIDEOAI_GITHUB_MIRROR")
    if base and url.startswith("https://github.com/"):
        return base.rstrip("/") + "/" + url.removeprefix("https://github.com/")
    return url
