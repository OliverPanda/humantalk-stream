"""OmniHuman 客户端：经 CV 兼容中转站上传素材、提交任务及轮询结果。

中转站须提供 Bearer 鉴权的 CVSubmitTask/CVGetResult，以及返回公网 URL
的 multipart /uploads 接口。OMNIHUMAN_MODEL 可选 v15 或 v1。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

VERSION = "2022-08-31"

REQ_KEYS = {
    "v15": {
        "create_role": "jimeng_realman_avatar_picture_create_role_omni_v15",
        "video": "jimeng_realman_avatar_picture_omni_v15",
    },
    "v1": {
        "create_role": "jimeng_realman_avatar_picture_create_role_omni",
        "video": "jimeng_realman_avatar_picture_omni_v2",
    },
}


class OmniHumanError(RuntimeError):
    pass


class OmniHumanClient:
    def __init__(self, config=None, logger=None):
        self.config = config
        self.logger = logger
        self.relay_base = os.environ.get("OMNIHUMAN_API_BASE", "").rstrip("/")
        self.relay_key = os.environ.get("OMNIHUMAN_API_KEY", "")
        self.upload_url = os.environ.get("OMNIHUMAN_UPLOAD_URL", "")
        get = config.get if config is not None else (lambda k, d=None: d)
        self.version = os.environ.get("OMNIHUMAN_MODEL") or str(get("avatar.omnihuman.version", "v15"))
        self.output_resolution = int(get("avatar.omnihuman.output_resolution", 1080))
        self.prompt = str(get("avatar.omnihuman.prompt", ""))
        self.poll_interval = float(get("avatar.omnihuman.poll_interval", 10))
        self.poll_timeout = float(get("avatar.omnihuman.poll_timeout", 900))

    def _log(self, msg: str) -> None:
        if self.logger:
            self.logger.info(msg)

    def available(self) -> tuple[bool, str]:
        if not self.relay_base:
            return False, "未配置 OMNIHUMAN_API_BASE"
        if not self.relay_key:
            return False, "未配置 OMNIHUMAN_API_KEY"
        if urlparse(self.relay_base).scheme not in ("http", "https"):
            return False, "OMNIHUMAN_API_BASE 须为 http(s) 地址"
        if self.version not in REQ_KEYS:
            return False, "OMNIHUMAN_MODEL 仅支持 v15 或 v1"
        return True, "ok"

    def _signed_request(self, action: str, body: dict) -> dict:
        """保留既有方法名供调用方使用，所有请求都走中转站。"""
        query = f"Action={action}&Version={VERSION}"
        resp = requests.post(
            f"{self.relay_base}/?{query}",
            headers={"Authorization": f"Bearer {self.relay_key}"},
            json=body, timeout=60,
        )
        try:
            data = resp.json()
        except ValueError:
            raise OmniHumanError(
                f"OmniHuman 中转站非 JSON 响应 HTTP {resp.status_code}: {resp.text[:200]}")
        if resp.status_code >= 400 or data.get("code", 10000) != 10000:
            raise OmniHumanError(
                f"OmniHuman 中转站错误 HTTP {resp.status_code} code={data.get('code')}: "
                f"{str(data.get('message', ''))[:300]}")
        return data

    # ---------- 任务提交与轮询 ----------

    def _submit(self, req_key: str, extra: dict) -> str:
        body = {"req_key": req_key, **extra}
        data = self._signed_request("CVSubmitTask", body)
        task_id = (data.get("data") or {}).get("task_id")
        if not task_id:
            raise OmniHumanError(f"提交任务未返回 task_id: {data}")
        return task_id

    def _poll(self, req_key: str, task_id: str) -> dict:
        deadline = time.time() + self.poll_timeout
        while time.time() < deadline:
            time.sleep(self.poll_interval)
            data = self._signed_request(
                "CVGetResult", {"req_key": req_key, "task_id": task_id})
            d = data.get("data") or {}
            status = d.get("status")
            if status == "done":
                return d
            if status in ("not_found", "expired"):
                raise OmniHumanError(f"任务 {status}（task_id={task_id}）")
            self._log(f"即梦数字人任务 {task_id} 状态={status}，继续等待...")
        raise OmniHumanError(f"任务轮询超时（>{self.poll_timeout:.0f}s）")

    # ---------- 本地素材经中转站上传 ----------
    def _ensure_url(self, path_or_url) -> str:
        s = str(path_or_url)
        if s.startswith("http://") or s.startswith("https://"):
            return s
        path = Path(s)
        if not path.is_file():
            raise OmniHumanError(f"素材不存在: {path}")
        upload_url = self.upload_url or f"{self.relay_base}/uploads"
        with path.open("rb") as f:
            resp = requests.post(
                upload_url,
                headers={"Authorization": f"Bearer {self.relay_key}"},
                files={"file": (path.name, f)}, timeout=120,
            )
        if resp.status_code >= 400:
            raise OmniHumanError(f"OmniHuman 素材上传失败 HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            data = resp.json()
        except ValueError:
            raise OmniHumanError("OmniHuman 素材上传未返回 JSON")
        url = data.get("url") or (data.get("data") or {}).get("url")
        if not url or urlparse(url).scheme not in ("http", "https"):
            raise OmniHumanError("OmniHuman 中转站上传响应缺少公网 url")
        return url

    # ---------- 对外主流程 ----------

    def generate(
        self,
        image: Path,
        audio: Path,
        output_path: Path,
        resume_task_id: str | None = None,
        on_video_task=None,
        on_video_submit_start=None,
    ) -> Path:
        """主体识别 → 视频生成 → 下载。阻塞直至完成（云端排队+生成通常数分钟）。

        resume_task_id: 传入未完成的视频任务 id 时，跳过提交直接续轮询——
        轮询超时后的重试必须走这里：重新提交会撞并发上限（code=50430），
        且原任务照常计费，等于双倍花钱。
        on_video_task: 视频任务提交成功后回调 task_id（调用方持久化以便续跑）。
        on_video_submit_start: 提交视频任务前回调（调用方记录结果不明时的防重标记）。
        """
        ok, reason = self.available()
        if not ok:
            raise OmniHumanError(reason)
        keys = REQ_KEYS[self.version]

        if resume_task_id:
            self._log(
                f"复用未完成的即梦任务 {resume_task_id}，续轮询（不重复提交/计费）...")
            video_task = resume_task_id
        else:
            image_url = self._ensure_url(image)
            audio_url = self._ensure_url(audio)

            # 1. 主体识别（校验图片含人物/类人主体，避免白花视频生成的钱）
            self._log(f"即梦数字人[{self.version}] 主体识别中...")
            role_task = self._submit(keys["create_role"], {"image_url": image_url})
            role_result = self._poll(keys["create_role"], role_task)
            resp_data = role_result.get("resp_data")
            if resp_data:
                try:
                    status = int(json.loads(resp_data).get("status", 1))
                    if status != 1:
                        raise OmniHumanError(
                            "照片未检测到人物主体，请换一张清晰的正面人像照")
                except (ValueError, TypeError):
                    pass  # resp_data 结构变化时不拦截，交给视频生成步骤裁决

            # 2. 视频生成
            body: dict = {"image_url": image_url, "audio_url": audio_url}
            if self.version == "v15":
                body["output_resolution"] = self.output_resolution
                body["pe_fast_mode"] = self.output_resolution <= 720  # 官方建议
                if self.prompt:
                    body["prompt"] = self.prompt[:300]
            self._log("即梦数字人视频生成中（1元/秒计费，按生成时长）...")
            if on_video_submit_start:
                on_video_submit_start()
            video_task = self._submit(keys["video"], body)
        if on_video_task:
            on_video_task(video_task)
        video_result = self._poll(keys["video"], video_task)
        video_url = video_result.get("video_url")
        if not video_url:
            raise OmniHumanError(f"任务完成但无 video_url: {video_result}")

        # 3. 下载（URL 有效期仅 1 小时，立即落盘）
        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = output_path.with_name(output_path.name + ".part")
        try:
            with requests.get(video_url, stream=True, timeout=600) as dl:
                dl.raise_for_status()
                with open(partial_path, "wb") as f:
                    for chunk in dl.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            if not partial_path.exists() or partial_path.stat().st_size == 0:
                raise OmniHumanError("视频下载后为空")
            os.replace(partial_path, output_path)
        finally:
            partial_path.unlink(missing_ok=True)
        self._log(f"即梦数字人完成: {output_path.name} "
                  f"({output_path.stat().st_size // 1024}KB)")
        return output_path
