"""NVIDIA 硬件探测与 Pro 固定模型状态。安装要求 GPU0 至少 18GB。"""
from __future__ import annotations

import subprocess
from typing import Any

TIER_ORDER = {"cloud": 0, "lite": 1, "pro": 2}

# 分档阈值（GB）
PRO_MIN_VRAM_GB = 18
LITE_MIN_VRAM_GB = 4


class VramGateError(RuntimeError):
    """显存/档位不足时抛出，携带人话说明。"""


def query_gpus() -> list[dict[str, Any]]:
    """经 nvidia-smi 查询显卡列表；无 NVIDIA 环境返回空列表。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return []
        gpus = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({
                    "name": parts[0],
                    "total_mb": int(parts[1]),
                    "free_mb": int(parts[2]),
                })
        return gpus
    except Exception:
        return []


def detect_tier(gpus: list[dict[str, Any]] | None = None) -> str:
    """按推理默认使用的 GPU0 定档；不以另一张大卡替代实际运行设备。"""
    if gpus is None:
        gpus = query_gpus()
    if not gpus:
        return "cloud"
    max_total_gb = gpus[0]["total_mb"] / 1024
    if max_total_gb >= PRO_MIN_VRAM_GB:
        return "pro"
    if max_total_gb >= LITE_MIN_VRAM_GB:
        return "lite"
    return "cloud"


def get_profile(config=None) -> dict[str, Any]:
    """完整硬件画像：检测结果 + 生效档位（含手动覆盖）。"""
    gpus = query_gpus()
    detected = detect_tier(gpus)
    manual = None
    if config is not None:
        manual = config.get("hardware.tier", "auto")
    if manual in ("cloud", "lite", "pro"):
        effective, source = manual, "manual"
    else:
        effective, source = detected, "auto"
    return {
        "gpus": gpus,
        "detected_tier": detected,
        "tier": effective,
        "tier_source": source,
    }


def tier_allows(current_tier: str, required_tier: str) -> bool:
    return TIER_ORDER.get(current_tier, 0) >= TIER_ORDER.get(required_tier, 0)


def ensure_vram(min_gb: float, feature: str, config=None) -> None:
    """重模型启动前的硬护栏：显存不足直接拒绝（人话报错），绝不试跑。

    nvidia-smi 不可用视为无卡，同样拒绝——云端 provider 不经过本函数。
    """
    gpus = query_gpus()
    if not gpus:
        raise VramGateError(
            f"「{feature}」需要 ≥{min_gb:g}GB 显存的 NVIDIA 显卡，"
            f"当前未检测到可用独显。可改用云端引擎（模型配置页）。"
        )
    max_total_gb = gpus[0]["total_mb"] / 1024
    if max_total_gb < min_gb:
        biggest = gpus[0]
        raise VramGateError(
            f"「{feature}」需要 ≥{min_gb:g}GB 显存，"
            f"本机 {biggest['name']} 只有 {max_total_gb:.0f}GB——已拦截，不会尝试运行。"
            f"请换 ≥{min_gb:g}GB 显卡的机器，或改用云端引擎。"
        )


# ============ 模型目录（供"模型管理"页与按档下载） ============
# disk_gb 为估算安装体积（权重+环境）；min_vram_gb 为运行门槛；
# tier 与实际显存共同决定解锁；安装方案由 installation 模块执行。

def get_model_catalog(config=None, profile: dict | None = None) -> list[dict[str, Any]]:
    """Pro 固定模型清单；安装状态只查本地，完整加载由安装器验证。"""
    from .config import Config, resolve_path
    from .pro_models import model_status
    config = config if config is not None else Config.load()
    profile = profile if profile is not None else get_profile(config)
    gpus = profile.get("gpus", [])
    max_gb = gpus[0]["total_mb"] / 1024 if gpus else 0
    catalog = []
    for model, name, kind, minimum, disk, key in (
        ("voxcpm", "VoxCPM-0.5B 声音克隆", "tts", 4, 7, "tts.voxcpm.model_dir"),
        ("musetalk", "MuseTalk 1.5 数字人", "avatar", 5, 13, "avatar.musetalk_local.repo_dir"),
        ("whisper", "Whisper small 本地字幕", "asr", 0, 1, "asr.whisper.model_dir"),
    ):
        folder = resolve_path(config.get(key) or "./workspace_data/runtime/models/faster-whisper-small")
        if model == "musetalk":
            folder = folder / "models"
        status = model_status(model, folder)
        installed = status["ready"]
        if model != "whisper":
            executable_key = ("tts.voxcpm.venv_python" if model == "voxcpm"
                              else "avatar.musetalk_local.env_python")
            executable = config.get(executable_key, "")
            installed = installed and bool(executable and resolve_path(executable).is_file())
        catalog.append({
            "id": model, "name": name, "kind": kind,
            "tier": "cloud" if model == "whisper" else "lite",
            "min_vram_gb": minimum, "disk_gb": disk,
            "weights_gb": round(status["bytes"] / 1024**3, 2),
            "installed": installed, "allowed": max_gb >= minimum,
            "desc": "Pro 固定链路，本地运行，权重通过 ModelScope 下载",
            "install": "bash install.sh --profile pro --install --activate",
            "installation_check": "files",
        })
    return catalog
