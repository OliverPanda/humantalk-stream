"""硬件分档（GPU Profile）与模型目录

启动时检测 NVIDIA 显卡与显存，把机器映射到能力档位，供各引擎做护栏：

- cloud: 无可用独显（或手动指定）——只允许云端 provider 与 CPU 兜底
- lite:  4-16GB 消费卡（如 RTX 2060 6GB）——现役栈 VoxCPM + MuseTalk
- pro:   >=16GB 大显存卡（如 RTX 4090 24GB）——额外解锁 LatentSync/EchoMimic 本地档

档位是"允许的上限"，不改变用户显式配置的 provider；重模型引擎在干活前调用
ensure_vram() 做硬拦截，防止在小卡上误跑大模型（显存不够直接拒绝并给出人话原因，
而不是 OOM 崩掉半条流水线）。

手动覆盖：config `hardware.tier: auto|cloud|lite|pro`（auto=按检测结果）。
"""
from __future__ import annotations

import subprocess
from typing import Any

TIER_ORDER = {"cloud": 0, "lite": 1, "pro": 2}

# 分档阈值（GB）
PRO_MIN_VRAM_GB = 16
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

def _model_installed(config, model: str) -> bool:
    """Check configured interpreter and required weights, independent of launch directory."""
    from .config import Config, resolve_path
    from .installation import MUSETALK_FILES
    cfg = config if config is not None else Config.load()
    if model == "voxcpm":
        executable = cfg.get("tts.voxcpm.venv_python", "")
        weights = cfg.get("tts.voxcpm.model_dir", "")
        if not executable or not weights:
            return False
        folder = resolve_path(weights)
        return resolve_path(executable).is_file() and all(
            (folder / name).is_file() and (folder / name).stat().st_size > 0
            for name in ("config.json", "pytorch_model.bin", "audiovae.pth", "tokenizer.json"))
    prefix = {"musetalk": "avatar.musetalk_local",
              "latentsync": "avatar.latentsync_local",
              "echomimic_v3": "avatar.echomimic_local"}[model]
    executable, repo = cfg.get(prefix + ".env_python", ""), cfg.get(prefix + ".repo_dir", "")
    if not executable or not repo or not resolve_path(executable).is_file():
        return False
    folder = resolve_path(repo)
    if model == "musetalk":
        return all((folder / "models" / name).is_file()
                   and (folder / "models" / name).stat().st_size > 0 for name in MUSETALK_FILES)
    # These optional third-party models use different checkpoint layouts.
    return folder.is_dir() and any(
        path.is_file() for pattern in ("**/*.pth", "**/*.safetensors", "**/*.ckpt")
        for path in folder.glob(pattern))


def get_model_catalog(config=None, profile: dict | None = None) -> list[dict[str, Any]]:
    """已知模型目录 + 本机安装状态。路径探测轻量（只 stat 目录）。

    profile 可传入已算好的 get_profile 结果，避免重复跑 nvidia-smi。
    """
    if config is None:
        from .config import Config
        config = Config.load()
    catalog = [
        {
            "id": "voxcpm",
            "name": "VoxCPM-0.5B 音色克隆",
            "kind": "tts",
            "tier": "lite",
            "min_vram_gb": 4,
            "disk_gb": 7,
            "installed": _model_installed(config, "voxcpm"),
            "desc": "零样本音色克隆 TTS（学音色+语气+说话习惯），本机 GPU 运行",
            "install": "Linux: bash install.sh --profile voice --install\n"
                       "Windows: powershell -File install.ps1 --profile voice --install",
        },
        {
            "id": "musetalk",
            "name": "MuseTalk 1.5 数字人（嘴型）",
            "kind": "avatar",
            "tier": "lite",
            "min_vram_gb": 5,
            "disk_gb": 13,
            "installed": _model_installed(config, "musetalk"),
            "desc": "只重绘嘴部区域；配合说话视频参考效果最好。MIT 许可可商用",
            "install": "Linux: bash install.sh --profile standard --install\n"
                       "Windows: powershell -File install.ps1 --profile standard --install\n"
                       "自动安装推荐至少 6GB 显存；安装后保留已有引擎选择，"
                       "需要切换时添加 --activate。",
        },
        {
            "id": "latentsync",
            "name": "LatentSync 1.6 高保真唇同步",
            "kind": "avatar",
            "tier": "pro",
            "min_vram_gb": 18,
            "disk_gb": 8,
            "installed": _model_installed(config, "latentsync"),
            "desc": "字节开源扩散唇同步，质量高于 MuseTalk 一档；4090 约 10 分钟/分钟",
            "install": "git clone https://github.com/bytedance/LatentSync ../LatentSync，"
                       "按其 README 下载 checkpoints（≥18GB 显存机器）",
        },
        {
            "id": "echomimic_v3",
            "name": "EchoMimic V3 半身驱动",
            "kind": "avatar",
            "tier": "pro",
            "min_vram_gb": 12,
            "disk_gb": 6,
            "installed": _model_installed(config, "echomimic_v3"),
            "desc": "蚂蚁开源 1.3B，音频驱动半身+手势，最接近'全身流畅'的本地方案",
            "install": "git clone https://github.com/antgroup/echomimic_v3 ../echomimic_v3，"
                       "按其 README 下载权重（≥12GB 显存机器）",
        },
        {
            "id": "omnihuman_api",
            "name": "即梦数字人 · OmniHuman（云端）",
            "kind": "avatar",
            "tier": "cloud",
            "min_vram_gb": 0,
            "disk_gb": 0,
            "installed": True,  # 纯 API 客户端，随应用内置
            "desc": "字节全身驱动数字人，头/手势/身体全动，任意机器可用；1元/秒按量计费",
            "install": "在模型配置 → 数字人模型中填写兼容 CVSubmitTask/CVGetResult 和素材上传的 OmniHuman 中转地址、密钥及模型",
        },
        {
            "id": "seedance_api",
            "name": "Seedance 2.0 空镜生成（云端）",
            "kind": "broll",
            "tier": "cloud",
            "min_vram_gb": 0,
            "disk_gb": 0,
            "installed": True,
            "desc": "文/图生视频，用于自动生成 B-roll 产品/场景空镜；约 1 元/秒",
            "install": "在模型配置 → 云端引擎密钥中填写 OpenAI 风格视频中转地址、密钥和模型",
        },
    ]
    if profile is None:
        profile = get_profile(config)
    gpus = profile.get("gpus", [])
    max_gb = gpus[0]["total_mb"] / 1024 if gpus else 0
    for m in catalog:
        m["allowed"] = (tier_allows(profile["tier"], m["tier"])
                        and max_gb >= m["min_vram_gb"])
        m["installation_check"] = "files"  # Dependency/model loading is verified by installer.
    return catalog
