"""OliverPanda/MuseTalk-API 的文件上传接口适配。"""
from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import uuid
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from filelock import FileLock


class MuseTalkAPIClient:
    def __init__(self, config):
        self.base = str(config.get("avatar.api_base", "")).rstrip("/")
        self.key = str(config.get("avatar.api_key", "") or "")
        self.timeout = float(config.get("avatar.timeout", 1800))
        if not self.base or urlparse(self.base).scheme not in ("http", "https"):
            raise ValueError("请在数字人模型页填写 MuseTalk-API 的 http(s) 服务根地址")

    @property
    def headers(self):
        return {"Authorization": f"Bearer {self.key}"} if self.key else {}

    def health(self) -> dict:
        response = httpx.get(f"{self.base}/health", headers=self.headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("status") != "healthy" or not data.get("models_loaded"):
            raise RuntimeError("MuseTalk-API 模型尚未加载完成")
        if not data.get("gpu_available"):
            raise RuntimeError("MuseTalk-API 未检测到可用 GPU")
        return data

    @staticmethod
    def _save(path: Path, state: dict):
        pending = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            pending.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
            os.replace(pending, path)
        finally:
            pending.unlink(missing_ok=True)

    def generate(self, source: Path, audio: Path, output: Path, fps: int = 25) -> Path:
        for path in (source, audio):
            if not path.is_file():
                raise RuntimeError(f"MuseTalk-API 输入素材不存在: {path}")
        output.parent.mkdir(parents=True, exist_ok=True)
        state_path = output.with_name(output.name + ".musetalk.json")
        digest = hashlib.sha256(f"{self.base}|{fps}".encode())
        for path in (source, audio):
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        fingerprint = digest.hexdigest()
        with FileLock(str(state_path) + ".lock"):
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if state.get("fingerprint") != fingerprint:
                    raise RuntimeError(f"MuseTalk 输入或地址已改变，请先核查原任务: {state_path}")
            else:
                state = {
                    "fingerprint": fingerprint,
                    "output_name": "freevideoai_" + uuid.uuid4().hex,
                    "submitted": False,
                }
            if output.is_file() and output.stat().st_size:
                return output
            with httpx.Client(
                timeout=httpx.Timeout(self.timeout, connect=15),
                headers=self.headers,
            ) as client:
                if not state.get("download_url"):
                    if state.get("submitted"):
                        raise RuntimeError(
                            "MuseTalk 提交结果不明，停止自动重提以避免重复推理；"
                            f"请在远端 results 查找 {state['output_name']}.mp4，任务记录: {state_path}"
                        )
                    state["submitted"] = True
                    self._save(state_path, state)
                    with source.open("rb") as image, audio.open("rb") as speech:
                        response = client.post(
                            f"{self.base}/generate",
                            files={
                                "source": (source.name, image, mimetypes.guess_type(source.name)[0] or "application/octet-stream"),
                                "audio": (audio.name, speech, mimetypes.guess_type(audio.name)[0] or "audio/wav"),
                            },
                            data={
                                "fps": str(fps),
                                "enhance": "false",
                                "output_name": state["output_name"],
                            },
                        )
                    response.raise_for_status()
                    data = response.json()
                    if data.get("status") != "success" or not data.get("download_url"):
                        raise RuntimeError("MuseTalk-API 未返回成功状态和下载地址")
                    # 服务返回 /download/...；保留反向代理 Base URL 的路径前缀。
                    target = urljoin(self.base + "/", data["download_url"].lstrip("/"))
                    if urlparse(target).netloc != urlparse(self.base).netloc:
                        raise RuntimeError("MuseTalk 下载地址必须与服务地址同站")
                    state["download_url"] = target
                    self._save(state_path, state)
                partial = output.with_name(output.name + ".part")
                try:
                    with client.stream("GET", state["download_url"]) as download:
                        download.raise_for_status()
                        with partial.open("wb") as stream:
                            for chunk in download.iter_bytes():
                                stream.write(chunk)
                    if not partial.is_file() or not partial.stat().st_size:
                        raise RuntimeError("MuseTalk 视频下载为空")
                    os.replace(partial, output)
                finally:
                    partial.unlink(missing_ok=True)
        return output
