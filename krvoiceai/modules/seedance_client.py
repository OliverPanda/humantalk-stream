"""视频空镜客户端：统一使用 OpenAI 风格中转站。"""
from __future__ import annotations

from pathlib import Path

from .video_relay_client import VideoRelayClient, VideoRelayError

SeedanceError = VideoRelayError


class SeedanceClient:
    """保留既有调用入口，Seedance 和 MiniMax 均由 VIDEO_MODEL 选择。"""

    def __init__(self, config=None, logger=None):
        self.relay = VideoRelayClient(config, logger)

    def available(self) -> tuple[bool, str]:
        return self.relay.available()

    def generate_clip(
        self,
        prompt: str,
        output_path: Path,
        duration: int = 5,
        ratio: str = "9:16",
        reference_image: Path | None = None,
        generate_audio: bool = False,
    ) -> Path:
        return self.relay.generate_clip(
            prompt, output_path, duration, ratio, reference_image, generate_audio
        )
