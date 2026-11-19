"""通过 OpenAI Python SDK 调用兼容 /v1/videos 的视频中转站。"""
from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import time
import uuid
from pathlib import Path

from filelock import FileLock
from openai import OpenAI, OpenAIError
from openai.types.video import Video


class VideoRelayError(RuntimeError):
    pass


class VideoRelayClient:
    def __init__(self, config=None, logger=None):
        get = config.get if config is not None else (lambda k, d=None: d)
        self.logger = logger
        self.base = os.environ.get("VIDEO_API_BASE", "").rstrip("/")
        self.api_key = os.environ.get("VIDEO_API_KEY", "")
        self.model = os.environ.get("VIDEO_MODEL", "")
        self.resolution = str(get("broll_auto.resolution", "720p")).lower()
        self.poll_interval = float(get("broll_auto.poll_interval", 10))
        self.poll_timeout = float(get("broll_auto.poll_timeout", 600))

    def available(self) -> tuple[bool, str]:
        if not self.base:
            return False, "未配置 VIDEO_API_BASE"
        if not self.api_key:
            return False, "未配置 VIDEO_API_KEY"
        if not self.model:
            return False, "未配置 VIDEO_MODEL"
        return True, "ok"

    def generate_clip(
        self,
        prompt: str,
        output_path: Path,
        duration: int = 5,
        ratio: str = "9:16",
        reference_image: Path | None = None,
        generate_audio: bool = False,
    ) -> Path:
        ok, reason = self.available()
        if not ok:
            raise VideoRelayError(reason)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "seconds": int(duration),
            "aspect_ratio": ratio,
            "resolution": self.resolution,
        }
        image_bytes = b""
        if reference_image is not None:
            image = Path(reference_image)
            if not image.is_file():
                raise VideoRelayError(f"参考图不存在: {image}")
            mime = mimetypes.guess_type(str(image))[0] or "image/jpeg"
            image_bytes = image.read_bytes()
            data_uri = f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"
            payload["images"] = [data_uri]
            if "@Image1" not in prompt:
                payload["prompt"] = f"参考 @Image1 中的主体，{prompt}"
        digest = hashlib.sha256()
        digest.update(json.dumps(
            {k: v for k, v in payload.items() if k != "images"},
            sort_keys=True, ensure_ascii=False,
        ).encode("utf-8"))
        digest.update(image_bytes)
        fingerprint = digest.hexdigest()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        state_path = output_path.with_name(output_path.name + ".task.json")
        lock_path = output_path.with_name(output_path.name + ".lock")
        with FileLock(str(lock_path)):
            if state_path.exists():
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                except (ValueError, OSError) as exc:
                    raise VideoRelayError(
                        f"视频任务记录损坏，停止重新提交以避免重复计费: {state_path}"
                    ) from exc
                if (state.get("fingerprint") != fingerprint
                        or state.get("base_url") != self.base):
                    raise VideoRelayError(
                        f"此输出路径已有其他视频任务，停止重新提交以避免重复计费: {state_path}"
                    )
            else:
                state = {
                    "fingerprint": fingerprint,
                    "base_url": self.base,
                    "request_id": uuid.uuid4().hex,
                    "task_id": None,
                    "submit_attempted": False,
                }
                self._save_task_state(state_path, state)
            if output_path.is_file() and output_path.stat().st_size:
                return output_path

            # 创建请求不可自动重试：并非所有兼容中转站都保证幂等。
            client = OpenAI(api_key=self.api_key, base_url=self.base, max_retries=0)
            task_id = state.get("task_id")
            if not task_id and state.get("submit_attempted"):
                try:
                    recovered = client.get(
                        f"/task-requests/{state['request_id']}",
                        cast_to=dict[str, object],
                        options={"timeout": 15},
                    )
                except OpenAIError as exc:
                    raise VideoRelayError(
                        f"视频创建结果不明，已停止重新提交以避免重复计费；"
                        f"请用请求号 {state['request_id']} 向中转站核查"
                    ) from exc
                task_id = recovered.get("id") or recovered.get("task_id")
                if task_id:
                    state["task_id"] = task_id
                    self._save_task_state(state_path, state)
                elif recovered.get("created") is False:
                    state["submit_attempted"] = False
                    self._save_task_state(state_path, state)
                else:
                    raise VideoRelayError(
                        f"视频创建结果不明，已停止重新提交以避免重复计费；"
                        f"请用请求号 {state['request_id']} 向中转站核查"
                    )
            if not task_id:
                state["submit_attempted"] = True
                self._save_task_state(state_path, state)
                try:
                    # 中转站使用 JSON 的 images/aspect_ratio/resolution 字段；
                    # SDK 的 videos.create 固定发送 multipart。
                    task = client.post(
                        "/videos", cast_to=Video, body=payload,
                        options={
                            "headers": {"Idempotency-Key": state["request_id"]},
                            "timeout": 120,
                        },
                    )
                except OpenAIError as exc:
                    raise VideoRelayError(f"视频任务创建失败: {exc}") from exc
                task_id = getattr(task, "id", None)
                if not task_id:
                    raise VideoRelayError("视频任务创建响应缺少 id")
                state["task_id"] = task_id
                self._save_task_state(state_path, state)
                if self.logger:
                    self.logger.info(f"视频中转任务已提交 id={task_id} model={self.model}")
            elif self.logger:
                self.logger.info(f"续查视频中转任务 id={task_id}")
            self._poll_and_download(client, task_id, output_path)
            return output_path

    @staticmethod
    def _save_task_state(path: Path, state: dict) -> None:
        tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def _poll_and_download(self, client: OpenAI, task_id: str, output_path: Path) -> None:
        deadline = time.monotonic() + self.poll_timeout
        while time.monotonic() < deadline:
            time.sleep(self.poll_interval)
            try:
                task = client.videos.retrieve(task_id, timeout=60)
            except OpenAIError as exc:
                status = getattr(exc, "status_code", None)
                if status == 429 or (isinstance(status, int) and status >= 500):
                    continue
                raise VideoRelayError(f"视频任务查询失败: {exc}") from exc
            status = getattr(task, "status", None)
            if not status:
                raise VideoRelayError("视频任务查询响应缺少 status")
            if status == "completed":
                break
            if status in ("failed", "cancelled", "expired"):
                error = getattr(task, "error", None)
                message = getattr(error, "message", None) or str(error or "")
                raise VideoRelayError(f"视频任务 {status}: {message[:300]}")
        else:
            raise VideoRelayError(f"视频任务超时（>{self.poll_timeout:.0f}s）")
        partial_path = output_path.with_name(output_path.name + ".part")
        try:
            with client.with_streaming_response.videos.download_content(
                task_id, timeout=300,
            ) as response:
                response.stream_to_file(partial_path)
            if not partial_path.is_file() or not partial_path.stat().st_size:
                raise VideoRelayError("视频下载后为空")
            os.replace(partial_path, output_path)
        except OpenAIError as exc:
            raise VideoRelayError(f"视频下载失败: {exc}") from exc
        finally:
            partial_path.unlink(missing_ok=True)
