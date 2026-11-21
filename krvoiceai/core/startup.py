"""启动时展示生效的 Pro 配置；不下载权重、不加载 GPU、不输出凭据。"""
from __future__ import annotations

import shutil

from .config import PROJECT_ROOT, Config, resolve_path
from .installation import pro_config
from .pro_models import require_model, require_musetalk_assets


def print_startup_summary(config: Config) -> None:
    defaults = Config(pro_config(PROJECT_ROOT))
    lines = [
        "\nFreeVideoAI Pro 启动说明",
        ("Pro 链路：DeepSeek Chat API → 本地 VoxCPM-0.5B → 本地 MuseTalk 1.5"
         " → 本地 Whisper small → FFmpeg"),
        f"配置文件：{PROJECT_ROOT / 'config/user_config.yaml'}（环境变量优先于 YAML）",
        ("模型来源：ModelScope；推理不自动联网补下载。"
         "启动仅检查文件及大小，未执行 SHA256 校验、CUDA 或实际推理验证。"),
    ]
    for section in ("llm", "tts", "avatar", "asr"):
        actual = config.get(f"{section}.provider", "未配置")
        expected = defaults.get(f"{section}.provider")
        lines.append(f"生效引擎 {section}：{actual}")
        if actual != expected:
            lines.append(f"  [待配置] Pro 应为 {expected}；检查用户配置及 "
                         f"KRVOICEAI_{section.upper()}_PROVIDER 环境变量。")
    lines.append("DeepSeek API Key：" + (
        "已设置（隐藏）" if config.get("llm.api_key") else
        "未设置；请在网页设置或 KRVOICEAI_LLM_API_KEY 中填写。"))
    lines.append(f"MuseTalk 版本：{config.get('avatar.musetalk_local.version', '未配置')}（Pro 使用 v15）")
    lines.append(f"字幕参数：model={config.get('asr.whisper.model_size', '未配置')}，"
                 f"device={config.get('asr.whisper.device', '未配置')}，"
                 f"compute_type={config.get('asr.whisper.compute_type', '未配置')}，"
                 f"vad_filter={config.get('asr.whisper.vad_filter', '未配置')}")
    for name, key in (
        ("VoxCPM-0.5B", "tts.voxcpm.model_dir"),
        ("MuseTalk", "avatar.musetalk_local.repo_dir"),
        ("Whisper small", "asr.whisper.model_dir"),
    ):
        directory = resolve_path(config.get(key) or defaults.get(key))
        lines.append(f"{name} 目录：{directory}")
        try:
            if name == "MuseTalk":
                require_musetalk_assets(directory)
            else:
                require_model("voxcpm" if name == "VoxCPM-0.5B" else "whisper", directory)
            lines.append("  [文件齐全]")
        except (RuntimeError, OSError) as exc:
            lines.append(f"  [待修复] {exc}")
    for label, key in (
        ("VoxCPM Python", "tts.voxcpm.venv_python"),
        ("VoxCPM worker", "tts.voxcpm.worker_script"),
        ("MuseTalk Python", "avatar.musetalk_local.env_python"),
        ("MuseTalk worker", "avatar.musetalk_local.worker_script"),
    ):
        path = resolve_path(config.get(key) or defaults.get(key))
        lines.append(f"{label}：{path} [{'文件存在' if path.is_file() else '缺失'}]")
    repo = resolve_path(config.get("avatar.musetalk_local.repo_dir")
                        or defaults.get("avatar.musetalk_local.repo_dir"))
    lines.append(f"MuseTalk 源码：{repo / 'scripts/inference.py'} "
                 f"[{'文件存在' if (repo / 'scripts/inference.py').is_file() else '缺失'}]")
    for tool in ("ffmpeg", "ffprobe"):
        lines.append(f"{tool}：{shutil.which(tool) or '未找到，请安装并加入 PATH'}")
    lines.extend([
        ("补齐模型并自动配置：用主程序 Python 执行 scripts/download_pro_models.py；"
         "自定义目录使用与下载时相同的 --runtime / --destination 参数。"),
        ("补齐依赖和独立环境：bash install.sh --profile pro --install --activate"
         "（Windows：powershell -ExecutionPolicy Bypass -File "
         ".\\install.ps1 --profile pro --install --activate）。"),
        ("avatar 加载验证：用上面显示的 MuseTalk Python 执行 "
         "scripts/verify_musetalk.py --repo <上面的MuseTalk目录> --cuda；"
         "任务推理详情见对应任务的 musetalk.log。"),
        ("Pro 探针：用主程序 Python 执行 scripts/probe_pro.py；"
         "加 --check-api 验证 DeepSeek，加 --face 真人素材 验证数字人出片。"
         "报告及日志保存到 workspace_data/probes/。"),
        "重启：在运行服务的终端按 Ctrl+C，再执行原来的 start.py 启动命令。\n",
    ])
    print("\n".join(lines), flush=True)
