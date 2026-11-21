"""faster-whisper 共享运行时

提供进程内 WhisperModel 单例缓存（按 model_size+device+compute_type 键控），
供字幕识别（subtitle_engine）与音色注册转写（tts_engine）复用，
避免每次调用重新加载模型（加载 small 模型约 2-4s，此前每次识别都会重复付出）。

GPU 支持（Windows）：
- 主 venv 是 torch-CPU，本身不带 CUDA 运行时；ctranslate2 的 CUDA 后端需要
  cublas64_12.dll + cudnn*_9.dll。本机 VoxCPM venv 的 torch-cu124 自带全套
  DLL，通过 asr.whisper.cuda_dll_dirs 配置注入 PATH 即可复用（实测可用）。
- device=cuda 加载/计算失败（DLL 缺失、显存不足等）时自动回退 cpu+int8，
  绝不破坏现有 CPU 路径。加载后用 0.5s 静音做 warmup 强制触发 cublas/cudnn
  加载，把"加载成功但计算才报错"的坑提前暴露在可回退的位置。
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, resolve_path
from .logger import get_logger

logger = get_logger().bind(module="asr_utils")

# (本地模型目录, device, compute_type) -> WhisperModel
_MODEL_CACHE: dict[tuple, Any] = {}
_CACHE_LOCK = threading.RLock()
_DLL_DIRS_INJECTED: set[str] = set()


def _inject_cuda_dll_dirs(dll_dirs) -> None:
    """把 CUDA 运行时 DLL 目录注入 PATH（幂等）

    ctranslate2 用系统默认搜索路径加载 cublas/cudnn，os.add_dll_directory
    单独用不生效，必须前置到 PATH。
    """
    for raw in dll_dirs or []:
        if not raw:
            continue
        p = Path(raw)
        if not p.is_absolute():
            p = (Path(PROJECT_ROOT) / raw).resolve()
        d = str(p)
        if d in _DLL_DIRS_INJECTED:
            continue
        if not p.is_dir():
            logger.warning(f"cuda_dll_dirs 目录不存在，跳过: {d}")
            continue
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(d)
        except Exception:
            pass
        _DLL_DIRS_INJECTED.add(d)
        logger.info(f"已注入 CUDA DLL 目录到 PATH: {d}")


def _warmup(model, device: str) -> None:
    """用 0.5s 静音强制跑一次推理，提前触发 cublas/cudnn 实际加载"""
    if device == "cpu":
        return
    import numpy as np
    segments, _ = model.transcribe(
        np.zeros(8000, dtype=np.float32), language="zh", beam_size=1,
    )
    list(segments)  # transcribe 是惰性生成器，必须消费才真正计算


def get_whisper_model(
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    download_root: str | None = None,
    cuda_dll_dirs: list | None = None,
):
    """获取进程内缓存的 WhisperModel 单例

    首次加载后缓存，同参数再次调用直接复用（省去每次 2-4s 的模型加载）。
    GPU 加载/warmup 失败自动降级 cpu+int8 并记录原因。
    """
    from .pro_models import require_model

    # small 是配置标签，实际始终读取本地 CTranslate2 目录。
    local_dir = resolve_path(
        "./workspace_data/runtime/models/faster-whisper-small"
        if model_size == "small" else model_size
    )
    require_model("whisper", local_dir)
    from faster_whisper import WhisperModel

    key = (str(local_dir), device, compute_type)
    with _CACHE_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is not None:
            return model
        if device != "cpu":
            _inject_cuda_dll_dirs(cuda_dll_dirs)
        try:
            model = WhisperModel(
                str(local_dir),
                device=device,
                compute_type=compute_type,
                local_files_only=True,
            )
            _warmup(model, device)
        except Exception as e:
            if device != "cpu":
                logger.warning(
                    f"faster-whisper device={device} compute={compute_type} "
                    f"不可用（{e}），回退 cpu+int8"
                )
                return get_whisper_model(model_size, "cpu", "int8", download_root)
            raise
        _MODEL_CACHE[key] = model
        logger.info(
            f"faster-whisper 模型已加载并缓存 model={model_size} "
            f"device={device} compute={compute_type}"
        )
        return model


def get_whisper_model_from_cfg(whisper_cfg: dict | None):
    """从 asr.whisper 配置块构造/复用 WhisperModel 单例"""
    cfg = whisper_cfg or {}
    return get_whisper_model(
        model_size=cfg.get("model_dir") or cfg.get("model_size", "small"),
        device=cfg.get("device", "cpu"),
        compute_type=cfg.get("compute_type", "int8"),
        download_root=cfg.get("download_root") or None,
        cuda_dll_dirs=cfg.get("cuda_dll_dirs") or None,
    )


def transcribe_ref_text(
    audio_path: str,
    whisper_cfg: dict | None = None,
    language: str = "zh",
) -> str:
    """转写参考音频为纯文本（用于 VoxCPM 零样本克隆的 prompt_text）

    Args:
        audio_path: 音频文件路径
        whisper_cfg: asr.whisper 配置块
        language: 语言代码

    Returns:
        转写文本（各 segment 拼接）；失败抛异常由调用方处理
    """
    cfg = whisper_cfg or {}
    model = get_whisper_model_from_cfg(cfg)
    segments, _info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=int(cfg.get("beam_size", 5)),
        vad_filter=bool(cfg.get("vad_filter", False)),
    )
    text = "".join((seg.text or "").strip() for seg in segments)
    return text.strip()
