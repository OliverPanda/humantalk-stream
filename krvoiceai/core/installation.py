"""Linux-first installation policy and uv runner, also usable before app dependencies exist."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .hardware import query_gpus
from .downloads import download_environment, github_url, torch_index
from .pro_models import MANIFEST, model_directory

PROFILE_MIN_MB = {"cloud": 0, "voice": 4096, "standard": 6144, "pro": 18432}
MUSETALK_REV = "2536eaf06b85f775d8e04b7fd81bf6699155a58d"
MUSETALK_FILES = tuple(item["path"] for item in MANIFEST["musetalk"])


def recommend_profile(max_vram_mb: int) -> str:
    for name in ("pro", "standard", "voice"):
        if max_vram_mb >= PROFILE_MIN_MB[name]:
            return name
    return "cloud"


def env_python(directory: Path, system: str | None = None) -> Path:
    return directory / ("Scripts/python.exe" if (system or platform.system()) == "Windows"
                        else "bin/python")


def pro_config(root: Path, runtime: Path | None = None, *,
               directories: dict[str, Path] | None = None,
               system: str | None = None) -> dict:
    """安装器和独立下载器共用的 Pro 配置，不依赖本机 GPU 探测。"""
    root = root.resolve()
    runtime = (runtime or root / "workspace_data/runtime").resolve()
    paths = {name: model_directory(runtime, name) for name in ("voxcpm", "musetalk", "whisper")}
    paths.update(directories or {})
    repo = paths["musetalk"].parent
    config = {
        "llm": {"provider": "deepseek", "base_url": "https://api.deepseek.com/v1",
                "model": "deepseek-chat"},
        "tts": {
            "provider": "voxcpm", "fallback_chain": ["voxcpm"],
            "voxcpm": {
                "venv_python": env_python(runtime / "voxcpm", system).as_posix(),
                "model_dir": paths["voxcpm"].as_posix(),
                "worker_script": (root / "scripts/voxcpm_worker.py").as_posix(),
            },
        },
        "avatar": {
            "provider": "musetalk_local", "timeout": 1800,
            "musetalk_local": {
                "env_python": env_python(runtime / "musetalk", system).as_posix(),
                "repo_dir": repo.as_posix(),
                "worker_script": (root / "scripts/musetalk_worker.py").as_posix(),
                "version": "v15", "batch_size": 4, "use_float16": True,
            },
        },
        "asr": {"provider": "whisper_local", "whisper": {
            "model_size": "small", "model_dir": paths["whisper"].as_posix(),
            "device": "cpu", "compute_type": "int8", "vad_filter": False,
        }},
        "broll_auto": {"enabled": False},
        "cover": {"mode": "frame_overlay"},
        "publisher": {"mode": "manual"},
    }
    if (system or platform.system()) == "Linux":
        config["subtitle"] = {"font_name": "Noto Sans CJK SC"}
        config["cover"]["font_path"] = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    return config


def make_plan(root: Path, requested: str, gpus: list[dict], *,
              system: str | None = None, machine: str | None = None) -> dict:
    root = root.resolve()
    system, machine = system or platform.system(), machine or platform.machine()
    # All current workers use cuda:0. Do not recommend against a different GPU.
    max_mb = gpus[0]["total_mb"] if gpus else 0
    profile = recommend_profile(max_mb) if requested == "auto" else requested
    if profile not in PROFILE_MIN_MB:
        raise ValueError(f"未知安装方案: {profile}")
    if max_mb < PROFILE_MIN_MB[profile]:
        raise ValueError(f"{profile} 方案需要至少 {PROFILE_MIN_MB[profile] // 1024}GB 显存，"
                         f"实际检测 {max_mb / 1024:g}GB；请使用 auto 或 cloud。")
    if system not in ("Linux", "Windows"):
        raise ValueError("自动安装目前支持 Linux 和 Windows。")
    if profile != "cloud" and machine.lower() not in ("x86_64", "amd64"):
        raise ValueError("GPU 安装仅支持 Linux/Windows x86_64；请选 cloud。")
    runtime = root / "workspace_data" / "runtime"
    models = [] if profile == "cloud" else ["voxcpm"]
    config = {
        "tts": {"provider": "edge_tts", "edge_voice": "zh-CN-XiaoxiaoNeural"},
        # An unconfigured remote endpoint fails clearly; never silently enable mock output.
        "avatar": {"provider": "musetalk_api", "api_base": "", "timeout": 1800},
        "asr": {"provider": "whisper_local", "whisper": {
            "device": "cpu", "compute_type": "int8"}},
    }
    if models:
        config["tts"].update({
            "provider": "voxcpm",
            "voxcpm": {
                "venv_python": env_python(runtime / "voxcpm", system).as_posix(),
                "model_dir": (runtime / "models/VoxCPM-0.5B").as_posix(),
                "worker_script": (root / "scripts/voxcpm_worker.py").as_posix(),
            },
        })
    if profile in ("standard", "pro"):
        models.append("musetalk")
        config["avatar"] = {
            "provider": "musetalk_local", "timeout": 1800,
            "musetalk_local": {
                "env_python": env_python(runtime / "musetalk", system).as_posix(),
                "repo_dir": (runtime / "MuseTalk").as_posix(),
                "worker_script": (root / "scripts/musetalk_worker.py").as_posix(),
                "batch_size": 2 if max_mb < 8192 else 4, "use_float16": True,
            },
        }
    if system == "Linux":
        config["subtitle"] = {"font_name": "Noto Sans CJK SC"}
        config["cover"] = {
            "font_path": "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"}
    if profile == "pro":
        models.append("whisper")
        config = pro_config(root, runtime, system=system)
    return {
        "root": root.as_posix(), "runtime": runtime.as_posix(), "system": system,
        "profile": profile, "max_vram_mb": max_mb, "models": models,
        "optional_models": [],
        "app_python": env_python(runtime / "app", system).as_posix(), "config": config,
    }


def run(argv: list[str | Path], *, cwd: Path | None = None) -> None:
    args = [str(x) for x in argv]
    print("+ " + subprocess.list2cmdline(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True, env=download_environment())


def uv_command() -> list[str]:
    configured = os.environ.get("FREEVIDEOAI_UV")
    if configured:
        if not Path(configured).is_file():
            raise RuntimeError(f"FREEVIDEOAI_UV 指定的程序不存在: {configured}")
        return [configured]
    if shutil.which("uv"):
        return ["uv"]
    home_binary = Path.home() / ".local/bin/uv"
    if home_binary.is_file() and os.name != "nt":
        return [str(home_binary)]
    if importlib.util.find_spec("uv"):
        return [sys.executable, "-m", "uv"]
    raise RuntimeError("未找到 uv。请先安装 uv，或使用 install.sh / install.ps1。")


def preflight(plan: dict) -> None:
    uv_command()
    needed = ["ffmpeg", "ffprobe"]
    if "musetalk" in plan["models"]:
        needed.append("git")
    missing = [name for name in needed if not shutil.which(name)]
    if missing:
        raise RuntimeError("缺少系统工具: " + ", ".join(missing)
                           + "。Linux 请先执行文档中的 apt-get 安装命令。")
    if plan["system"] == "Linux":
        font = Path(plan["config"]["cover"]["font_path"])
        if not font.is_file():
            raise RuntimeError("缺少中文字体，请安装 fonts-noto-cjk。")
    runtime = Path(plan["runtime"])
    runtime.mkdir(parents=True, exist_ok=True)
    required_gb = 4 + (7 if "voxcpm" in plan["models"] else 0) + (
        18 if "musetalk" in plan["models"] else 0)
    if shutil.disk_usage(runtime).free < required_gb * 1024**3:
        raise RuntimeError(f"可用磁盘空间不足，建议至少保留 {required_gb}GB 用于依赖和权重。")


def create_env(directory: Path, python: str) -> Path:
    executable = env_python(directory)
    if not executable.is_file():
        run(uv_command() + ["venv", "--python", python, "--seed", str(directory)])
    return executable


def pip_install(executable: Path, *packages: str) -> None:
    run(uv_command() + ["pip", "install", "--python", str(executable), *packages])


def verify_cuda(executable: Path) -> None:
    run([executable, "-c",
         "import torch; assert torch.cuda.is_available(), 'CUDA 不可用，请检查驱动'; "
         "torch.zeros(1, device='cuda').add_(1); print(torch.cuda.get_device_name(0))"])


def install_base(plan: dict) -> None:
    root, runtime = Path(plan["root"]), Path(plan["runtime"])
    executable = create_env(runtime / "app", "3.11")
    # 安装 SDK 在先，所有权重下载只通过 ModelScope。
    pip_install(executable, "modelscope==1.34.0")
    pip_install(executable, "-e", f"{root}[cloud,local]", "uv")
    run([executable, "-m", "pip", "check"])
    run([executable, "-c", "import faster_whisper, fastapi; "
         "from krvoiceai.web.server import create_app; create_app()"], cwd=root)


def install_voxcpm(plan: dict) -> None:
    root, runtime = Path(plan["root"]), Path(plan["runtime"])
    executable = create_env(runtime / "voxcpm", "3.11")
    # Separate environment: MuseTalk needs torch 2.0, VoxCPM requires >=2.5.
    pip_install(executable, "torch==2.5.1", "torchaudio==2.5.1",
                "--index-url", torch_index("cu121"))
    pip_install(executable, "voxcpm==1.0.5", "transformers==4.51.3",
                "numpy<2", "gradio<6", "huggingface-hub<1",
                "torch==2.5.1+cu121", "torchaudio==2.5.1+cu121")
    verify_cuda(executable)
    model_dir = plan["config"]["tts"]["voxcpm"]["model_dir"]
    run([plan["app_python"], root / "scripts/download_pro_models.py",
         "--models", "voxcpm", "--destination", model_dir, "--no-config"])
    run([executable, "-m", "pip", "check"])
    # Loading actual weights catches incompatible packages, CUDA and corrupt downloads.
    run([executable, "-c",
         "import os; os.environ['HF_HUB_OFFLINE']='1'; "
         "os.environ['TRANSFORMERS_OFFLINE']='1'; "
         "import torch, sys; assert torch.cuda.is_available(), 'CUDA 不可用'; "
         "from voxcpm import VoxCPM; "
         "VoxCPM.from_pretrained(sys.argv[1], load_denoiser=False, optimize=False)", model_dir])


def install_musetalk(plan: dict) -> None:
    root, runtime = Path(plan["root"]), Path(plan["runtime"])
    executable = create_env(runtime / "musetalk", "3.10")
    repo = runtime / "MuseTalk"
    if not (repo / ".git").is_dir():
        repo.mkdir(parents=True, exist_ok=True)
        run(["git", "init", repo])
        run(["git", "-C", repo, "remote", "add", "origin",
             "https://github.com/TMElyralab/MuseTalk.git"])
    run(["git", "-C", repo, "fetch", "--depth", "1",
         github_url("https://github.com/TMElyralab/MuseTalk.git"), MUSETALK_REV])
    run(["git", "-C", repo, "checkout", "--detach", "FETCH_HEAD"])
    pip_install(executable, "pip==24.3.1", "setuptools==68.2.2", "wheel==0.45.1")
    # Shared with Docker; avoid maintaining two subtly different dependency sets.
    run([executable, root / "scripts/install_musetalk.py", "--repo", repo])
    verify_cuda(executable)
    run([plan["app_python"], root / "scripts/download_pro_models.py",
         "--models", "musetalk", "--destination", repo / "models", "--no-config"])
    run([executable, "-m", "pip", "check"])
    run([executable, root / "scripts/verify_musetalk.py", "--repo", repo, "--cuda"])


def install_whisper(plan: dict) -> None:
    root = Path(plan["root"])
    directory = plan["config"]["asr"]["whisper"]["model_dir"]
    run([plan["app_python"], root / "scripts/download_pro_models.py",
         "--models", "whisper", "--destination", directory, "--no-config"])
    run([plan["app_python"], "-c",
         "from faster_whisper import WhisperModel; import sys; "
         "WhisperModel(sys.argv[1], device='cpu', compute_type='int8', "
         "local_files_only=True); print('Whisper small 本地加载验证通过')", directory])


def save_config(destination: Path, recommended: dict, *, activate: bool = False) -> None:
    """Atomic save, existing scalar choices win unless activation was explicitly requested."""
    import yaml
    from .config_store import config_lock, write_config

    def merge(base, extra):
        result = dict(base)
        for key, value in extra.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge(result[key], value)
            elif key not in result or activate:
                result[key] = value
        return result

    with config_lock(destination):
        existing = yaml.safe_load(destination.read_text(encoding="utf-8")) if destination.exists() else {}
        if existing is None:
            existing = {}
        if not isinstance(existing, dict):
            raise ValueError("用户配置必须为 YAML 对象。")
        # DeepSeek 可以使用兼容代理；固定模型选型不更换已配置的服务地址或密钥。
        if recommended.get("llm", {}).get("provider") == "deepseek":
            base_url = existing.get("llm", {}).get("base_url")
            if base_url:
                recommended = {**recommended, "llm": {
                    **recommended["llm"], "base_url": base_url,
                }}
        write_config(destination, merge(existing, recommended))


def validate_pro_environment(recommended: dict) -> None:
    """阻止旧环境变量覆盖刚安装的固定模型或本地路径，不输出密钥。"""
    from .config import Config, _apply_env_overrides, resolve_path

    expected = Config(recommended)
    effective = Config(_apply_env_overrides(recommended))
    conflicts = []
    for key in (
        "llm.provider", "llm.model", "tts.provider", "tts.fallback_chain",
        "avatar.provider", "asr.provider", "asr.whisper.model_size",
        "asr.whisper.model_dir", "asr.whisper.device", "asr.whisper.compute_type",
        "asr.whisper.vad_filter",
        "tts.voxcpm.model_dir", "tts.voxcpm.venv_python", "tts.voxcpm.worker_script",
        "avatar.musetalk_local.repo_dir", "avatar.musetalk_local.env_python",
        "avatar.musetalk_local.version", "avatar.musetalk_local.worker_script",
        "broll_auto.enabled",
    ):
        wanted, actual = expected.get(key), effective.get(key)
        if wanted is None:
            continue
        if key.endswith((".model_dir", ".repo_dir", ".env_python", ".venv_python",
                         ".worker_script")):
            wanted, actual = resolve_path(wanted), resolve_path(actual)
        if wanted != actual:
            conflicts.append("KRVOICEAI_" + key.replace(".", "_").upper())
    if conflicts:
        raise RuntimeError(
            "环境变量将覆盖 Pro 本地配置: " + ", ".join(conflicts)
            + "。请从启动环境或 .env 移除这些冲突项，再重新安装；已下载文件保留。"
        )


def execute_plan(plan: dict, *, activate: bool = False) -> None:
    try:
        from filelock import FileLock
    except ImportError as exc:
        raise RuntimeError("安装执行需要 filelock，请使用 install.sh / install.ps1 入口。") from exc
    preflight(plan)
    runtime = Path(plan["runtime"])
    runtime.mkdir(parents=True, exist_ok=True)
    with FileLock(str(runtime / "install.lock"), timeout=0):
        install_base(plan)
        for model in plan["models"]:
            {"voxcpm": install_voxcpm, "musetalk": install_musetalk,
             "whisper": install_whisper}[model](plan)
        # Run through the installed interpreter: bootstrap Python need not have PyYAML.
        run([plan["app_python"], "-m", "krvoiceai.core.installation", "--save-config",
             json.dumps(plan["config"], ensure_ascii=True),
             *(["--activate"] if activate or plan["profile"] == "pro" else [])],
            cwd=Path(plan["root"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FreeVideoAI Pro 本地安装（Linux 优先）")
    parser.add_argument("--profile", choices=["auto", "pro"], default="pro")
    parser.add_argument("--install", action="store_true", help="执行安装；默认仅预览")
    parser.add_argument("--activate", action="store_true", help="安装成功后切换现有引擎配置")
    parser.add_argument("--json", action="store_true", help="输出 JSON 安装计划")
    parser.add_argument("--save-config", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    try:
        if args.save_config:
            from .config import Config
            recommended = json.loads(args.save_config)
            Config.load()  # 加载 .env，再检查其是否会覆盖固定模型。
            if recommended.get("llm", {}).get("provider") == "deepseek":
                validate_pro_environment(recommended)
            save_config(root / "config/user_config.yaml", recommended,
                        activate=args.activate)
            return 0
        plan = make_plan(root, "pro", query_gpus())
        if args.json:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
        else:
            print(f"系统: {plan['system']}，显存: {plan['max_vram_mb'] / 1024:g}GB\n"
                  f"推荐方案: {plan['profile']}\n"
                  f"本地模型: {', '.join(plan['models'])}\n"
                  f"安装位置: {plan['runtime']}")
        if args.install:
            execute_plan(plan, activate=args.activate)
            print("安装验证通过，已应用 Pro 本地配置；DeepSeek 密钥和用户素材保留。")
            print(f"启动: \"{plan['app_python']}\" start.py --host 0.0.0.0")
            print("下一步：配置 DeepSeek API Key，注册声音和形象，生成短文案验证。")
        elif not args.json:
            print("预览完成，添加 --install 执行；验证通过后应用固定 Pro 选型。")
        return 0
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"安装失败: {exc}\n已下载文件保留，可修复原因后重跑同一命令。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
