"""Single native installation process shared by Web API requests."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

from . import installation
from .hardware import query_gpus

_lock = threading.Lock()
_process: subprocess.Popen | None = None
_log: Path | None = None


def native_available() -> bool:
    root = Path(__file__).resolve().parents[2]
    if Path("/.dockerenv").exists() or not (root / "scripts/install.py").is_file():
        return False
    try:
        installation.uv_command()
        return True
    except RuntimeError:
        return False


def status() -> dict:
    with _lock:
        code = _process.poll() if _process else None
        state = "idle" if _process is None else (
            "running" if code is None else "success" if code == 0 else "failed")
        text = ""
        if _log and _log.is_file():
            with _log.open("rb") as stream:
                stream.seek(max(0, _log.stat().st_size - 16000))
                text = stream.read().decode("utf-8", errors="replace")
        return {"state": state, "exit_code": code, "log": text}


def start(profile: str, activate: bool = False) -> dict:
    global _process, _log
    if not native_available():
        raise RuntimeError("当前为容器部署或缺少 uv；请在宿主机使用安装脚本或构建对应镜像。")
    root = Path(__file__).resolve().parents[2]
    profile = "pro" if profile == "auto" else profile
    plan = installation.make_plan(root, profile, query_gpus())
    installation.preflight(plan)
    with _lock:
        if _process is not None and _process.poll() is None:
            raise RuntimeError("已有安装任务正在运行。")
        _log = Path(plan["runtime"]) / "install.log"
        command = [sys.executable, "-u", "-m", "krvoiceai.core.installation",
                   "--profile", profile, "--install"]
        if activate:
            command.append("--activate")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        with _log.open("wb") as stream:
            _process = subprocess.Popen(command, cwd=root, env=env,
                                        stdout=stream, stderr=subprocess.STDOUT)
    return status()
