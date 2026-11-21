"""Pro 唯一权重下载入口：VoxCPM、MuseTalk 及附属模型、Whisper small。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from krvoiceai.core.pro_models import MODEL_NAMES, download_model, model_directory


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--runtime", type=Path, default=ROOT / "workspace_data/runtime")
    parser.add_argument("--destination", type=Path, help="单个模型组的目标目录")
    parser.add_argument("--no-config", action="store_true",
                        help="仅下载；供安装器加载验证后统一保存配置")
    args = parser.parse_args(argv)
    if args.destination and len(args.models) != 1:
        parser.error("--destination 只能与一个 --models 配合使用")
    runtime = (ROOT / args.runtime).resolve()
    destination = (ROOT / args.destination).resolve() if args.destination else None
    if destination and args.models == ["musetalk"] and destination.name != "models":
        parser.error("MuseTalk --destination 必须为其源码目录下的 models 目录")
    directories = {}
    for model in args.models:
        directory = destination or model_directory(runtime, model)
        download_model(model, directory)
        directories[model] = directory
    if args.no_config:
        print("模型下载校验完成；本次仅下载，配置由调用方在环境验证后保存。", flush=True)
        return

    from krvoiceai.core.config import Config
    from krvoiceai.core.installation import pro_config, save_config, validate_pro_environment

    recommended = pro_config(ROOT, runtime, directories=directories)
    if set(args.models) != set(MODEL_NAMES):
        sections = {"llm", *({"voxcpm": "tts", "musetalk": "avatar", "whisper": "asr"}[name]
                            for name in args.models)}
        recommended = {key: value for key, value in recommended.items() if key in sections}
    Config.load()  # 同安装器：加载 .env 后检查是否覆盖刚写入的本地选型。
    validate_pro_environment(recommended)
    config_path = ROOT / "config/user_config.yaml"
    save_config(config_path, recommended, activate=True)
    print(f"[配置完成] {config_path}\n"
          "已保存所下载模型的本地路径和引擎选型，保留 DeepSeek 密钥及素材目录；"
          "旧配置备份为 user_config.yaml.bak（若原文件存在）。\n"
          "模型文件校验不代表 GPU 环境验证；尚未安装独立环境时，请执行 "
          "bash install.sh --profile pro --install --activate"
          "（Windows 使用 install.ps1）。\n"
          "服务已运行时请 Ctrl+C 停止后重新执行 start.py，使新配置生效。",
          flush=True)


if __name__ == "__main__":
    main()
