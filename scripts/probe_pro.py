"""Pro 探针：校验本地模型、CUDA、短音频推理和 FFmpeg，保存逐项日志及 JSON 报告。"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import wave
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from krvoiceai.core.config import Config
from krvoiceai.core.installation import PROFILE_MIN_MB, pro_config, query_gpus
from krvoiceai.core.pro_models import model_status, require_musetalk_assets

OFFLINE_CODE = """
import socket
def block_network(*args, **kwargs):
    raise RuntimeError('本地模型探针已禁止网络连接')
socket.socket.connect = block_network
socket.socket.connect_ex = block_network
socket.create_connection = block_network
"""
CUDA_CODE = """
import torch
assert torch.cuda.is_available(), 'CUDA 不可用，请检查驱动和独立 Python 环境'
torch.zeros(1, device='cuda').add_(1)
torch.cuda.synchronize()
print('CUDA:', torch.cuda.get_device_name(0), flush=True)
"""


def run_probe(config: Config, output: Path, *, timeout: int = 180,
              check_api: bool = False, face: Path | None = None) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("探针输出目录非空，请使用新的空目录，避免旧产物造成误判。")
    defaults = Config(pro_config(ROOT))
    checks = []
    results = {}
    secrets = [config.get("llm.api_key"), config.get("llm.base_url")]

    def redact(value) -> str:
        text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
        for secret in secrets:
            if secret:
                text = text.replace(str(secret), "[已隐藏]")
        return text

    def path(key: str) -> Path:
        return (ROOT / (config.get(key) or defaults.get(key))).resolve()

    def command(name: str, argv: list) -> str:
        log = output / f"{name}.log"
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONIOENCODING": "utf-8",
               "PYTHONUNBUFFERED": "1"}
        try:
            result = subprocess.run(
                [str(arg) for arg in argv], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            text = redact(exc.output or "")
            with log.open("a", encoding="utf-8") as stream:
                stream.write(text + f"\n超时：{timeout} 秒\n")
            raise RuntimeError(f"超时（{timeout} 秒）；日志：{log}\n{text[-1200:]}") from exc
        except OSError as exc:
            log.write_text(redact(exc), encoding="utf-8")
            raise RuntimeError(f"无法启动子进程；日志：{log}；{exc}") from exc
        text = redact(result.stdout or "")
        with log.open("a", encoding="utf-8") as stream:
            stream.write(text + "\n")
        if result.returncode:
            raise RuntimeError(f"退出码 {result.returncode}；日志：{log}\n{text[-1200:]}")
        return result.stdout or ""

    def check(name: str, action=None, *, depends: tuple = (), skip: str = "") -> None:
        blocked = [key for key in depends if results.get(key) != "pass"]
        if skip or blocked:
            status, detail = "skip", skip or f"前置检查未通过：{', '.join(blocked)}"
        else:
            print(f"[检查] {name}", flush=True)
            try:
                detail = action()
                status = "pass"
            except Exception as exc:  # noqa: BLE001 -- 探针需继续收集其他阶段错误
                status, detail = "fail", f"{type(exc).__name__}: {exc}"
        detail = redact(detail)
        results[name] = status
        checks.append({"id": name, "status": status, "detail": detail})
        print(f"[{status.upper()}] {name}：{detail}", flush=True)

    def configuration() -> str:
        fixed = (
            "llm.provider", "llm.model", "tts.provider", "tts.fallback_chain",
            "avatar.provider", "avatar.musetalk_local.version", "asr.provider",
            "asr.whisper.model_size", "asr.whisper.device", "asr.whisper.compute_type",
            "asr.whisper.vad_filter", "broll_auto.enabled",
        )
        wrong = [key for key in fixed if config.get(key) != defaults.get(key)]
        if wrong:
            raise RuntimeError("Pro 选型不符：" + ", ".join(wrong)
                               + "；检查 YAML 和 KRVOICEAI_* 环境变量，重跑下载器自动配置。")
        return "DeepSeek Chat API + VoxCPM-0.5B + MuseTalk v15 + Whisper small CPU int8"

    check("configuration", configuration)

    def gpu() -> str:
        gpus = query_gpus()
        if not gpus:
            raise RuntimeError("nvidia-smi 未检测到 NVIDIA 显卡")
        first = gpus[0]
        if first["total_mb"] < PROFILE_MIN_MB["pro"]:
            raise RuntimeError(f"第一张显卡 {first['name']} 显存 {first['total_mb'] / 1024:g}GiB，"
                               "Pro 需要至少 18GiB（不合并多卡显存）")
        return f"{first['name']}，{first['total_mb'] / 1024:g}GiB"

    check("gpu", gpu)

    def key_check() -> str:
        if not config.get("llm.api_key"):
            raise RuntimeError("未配置 DeepSeek API Key")
        return "已配置（隐藏）"

    check("deepseek_key", key_check)

    def api_check() -> str:
        import httpx

        response = httpx.post(
            config.get("llm.base_url").rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {config.get('llm.api_key')}"},
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "只回复 OK"}],
                  "max_tokens": 8, "stream": False},
            timeout=min(timeout, 30),
        )
        if response.status_code != 200:
            raise RuntimeError(f"DeepSeek API HTTP {response.status_code}")
        if not response.json()["choices"][0]["message"]["content"].strip():
            raise RuntimeError("DeepSeek API 返回空内容")
        return "真实 Chat API 请求通过"

    check("deepseek_api", api_check, depends=("configuration", "deepseek_key"),
          skip="" if check_api else "未请求 API；加 --check-api 验证真实 Chat 请求")
    directories = {
        "voxcpm": path("tts.voxcpm.model_dir"),
        "musetalk": path("avatar.musetalk_local.repo_dir") / "models",
        "whisper": path("asr.whisper.model_dir"),
    }
    for model, directory in directories.items():
        def weights(model=model, directory=directory):
            if model == "musetalk":
                require_musetalk_assets(directory.parent, checksum=True)
            else:
                status = model_status(model, directory, checksum=True)
                if not status["ready"]:
                    raise RuntimeError(f"{directory}：文件缺失或 SHA256 不符："
                                       + ", ".join(status["missing"])
                                       + "；用 download_pro_models.py 修复。")
            return f"大小及 SHA256 校验通过：{directory}"
        check(f"{model}_weights", weights)

    for model, keys in (
        ("voxcpm", ("tts.voxcpm.venv_python", "tts.voxcpm.worker_script")),
        ("musetalk", ("avatar.musetalk_local.env_python", "avatar.musetalk_local.worker_script")),
    ):
        def environment(keys=keys, model=model):
            files = [path(key) for key in keys]
            if model == "musetalk":
                files.append(directories["musetalk"].parent / "scripts/inference.py")
            missing = [str(file) for file in files if not file.is_file()]
            if missing:
                raise RuntimeError("文件缺失：" + ", ".join(missing)
                                   + "；请运行 Pro 安装器补齐依赖/源码。")
            return "Python、worker 和源码文件存在"
        check(f"{model}_environment", environment)

    audio = output / "voxcpm.wav"

    def voxcpm() -> str:
        task = output / "voxcpm-task.json"
        task.write_text(json.dumps({
            "text": "这是本地探针测试。", "out_wav": str(audio),
            "model_dir": str(directories["voxcpm"]),
            "inference_timesteps": config.get("tts.voxcpm.inference_timesteps", 10),
            "seed": 42,
        }, ensure_ascii=False), encoding="utf-8")
        code = OFFLINE_CODE + CUDA_CODE + """
import runpy, sys
worker, task = sys.argv[1:]
sys.argv = [worker, task]
runpy.run_path(worker, run_name='__main__')
"""
        command("voxcpm", [path("tts.voxcpm.venv_python"), "-c", code,
                           path("tts.voxcpm.worker_script"), task])
        if not audio.is_file() or not audio.stat().st_size:
            raise RuntimeError("进程退出但未生成有效音频；查看 voxcpm.log")
        return f"CUDA 运算及真实短音频合成通过：{audio}"

    check("voxcpm", voxcpm, depends=("gpu", "voxcpm_weights", "voxcpm_environment"))

    def musetalk() -> str:
        code = OFFLINE_CODE + """
import runpy, sys
script, repo = sys.argv[1:]
sys.argv = [script, '--repo', repo, '--cuda']
runpy.run_path(script, run_name='__main__')
"""
        command("musetalk", [path("avatar.musetalk_local.env_python"), "-c", code,
                             ROOT / "scripts/verify_musetalk.py",
                             directories["musetalk"].parent])
        return "CUDA 运算和 MuseTalk 模型栈加载通过；出片需 --face"

    check("musetalk", musetalk, depends=("gpu", "musetalk_weights", "musetalk_environment"))

    def whisper() -> str:
        code = OFFLINE_CODE + """
import json, sys
sys.path.insert(0, sys.argv[1])
from krvoiceai.core.asr_utils import get_whisper_model_from_cfg
model = get_whisper_model_from_cfg(json.loads(sys.argv[2]))
if sys.argv[3]:
    sample = sys.argv[3]
else:
    import numpy as np
    sample = np.zeros(16000, dtype=np.float32)
segments, info = model.transcribe(sample, language='zh', beam_size=1, vad_filter=False)
print('Whisper 推理通过；片段数：', len(list(segments)))
"""
        options = {key: config.get(f"asr.whisper.{key}", defaults.get(f"asr.whisper.{key}"))
                   for key in ("model_size", "device", "compute_type")}
        options["model_dir"] = str(directories["whisper"])
        sample = str(audio) if results["voxcpm"] == "pass" else ""
        command("whisper", [sys.executable, "-c", code, ROOT, json.dumps(options), sample])
        return "真实合成音频转写通过" if sample else "静音推理通过（尚未验证真实语音识别）"

    check("whisper", whisper, depends=("whisper_weights",))

    def ffmpeg() -> str:
        for tool in ("ffmpeg", "ffprobe"):
            if not shutil.which(tool):
                raise RuntimeError(f"{tool} 不在 PATH")
        silence = output / "silence.wav"
        with wave.open(str(silence), "wb") as stream:
            stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            stream.writeframes(b"\x00\x00" * 16000)
        video = output / "ffmpeg.mp4"
        command("ffmpeg", ["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i",
                          "color=c=black:s=320x240:r=25", "-i", silence, "-t", "1",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", video])
        if not video.is_file() or not video.stat().st_size:
            raise RuntimeError("FFmpeg 未生成视频")
        metadata = json.loads(command("ffmpeg", [
            "ffprobe", "-v", "error", "-show_entries", "stream=codec_type:format=duration",
            "-of", "json", video,
        ]))
        if not {"video", "audio"} <= {item["codec_type"] for item in metadata["streams"]}:
            raise RuntimeError("FFmpeg 视频缺少音轨或画面")
        if float(metadata["format"]["duration"]) <= 0:
            raise RuntimeError("FFmpeg 视频时长无效")
        return f"真实 H.264/AAC 合成及 ffprobe 检查通过：{video}"

    check("ffmpeg", ffmpeg)

    def avatar_video() -> str:
        reference = (ROOT / face).resolve()
        if not reference.is_file():
            raise RuntimeError(f"参考素材不存在：{reference}")
        video = output / "avatar.mp4"
        code = OFFLINE_CODE + """
import runpy, sys
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
"""
        command("avatar_video", [
            path("avatar.musetalk_local.env_python"), "-c", code,
            path("avatar.musetalk_local.worker_script"), "--repo", directories["musetalk"].parent,
            "--face", reference, "--audio", audio, "--outfile", video,
            "--version", "v15", "--batch_size", str(config.get("avatar.musetalk_local.batch_size", 4)),
            *(["--use_float16"] if config.get("avatar.musetalk_local.use_float16", True) else []),
        ])
        if not video.is_file() or not video.stat().st_size:
            raise RuntimeError("MuseTalk 未生成数字人视频；查看 avatar_video.log")
        return f"真实数字人视频已生成：{video}"

    check("avatar_video", avatar_video, depends=("voxcpm", "musetalk", "ffmpeg"),
          skip="" if face else "未验证数字人出片；加 --face 真人参考视频或照片进行验证")
    report = {
        "ok": not any(item["status"] == "fail" for item in checks),
        "api_verified": results["deepseek_api"] == "pass",
        "avatar_verified": results["avatar_video"] == "pass",
        "created_at": datetime.now().astimezone().isoformat(),
        "output": str(output), "checks": checks,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n探针{'通过' if report['ok'] else '失败'}；"
          f"API 实测={report['api_verified']}，数字人出片实测={report['avatar_verified']}\n"
          f"报告：{output / 'report.json'}", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="指定用户 YAML；默认读取项目配置及 .env")
    parser.add_argument("--output", type=Path, help="报告和日志目录；默认 workspace_data/probes/时间")
    parser.add_argument("--timeout", type=int, default=180, help="每个子进程超时秒数，默认 180")
    parser.add_argument("--check-api", action="store_true", help="真实调用 DeepSeek Chat API")
    parser.add_argument("--face", type=Path, help="真人参考视频/照片；额外验证 MuseTalk 实际出片")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout 必须大于 0")
    output = (ROOT / args.output) if args.output else (
        ROOT / "workspace_data/probes" / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]))
    try:
        config = Config.load(config_path=ROOT / args.config if args.config else None)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"配置读取失败（{type(exc).__name__}），请检查 YAML 文件格式和路径。", file=sys.stderr)
        return 1
    try:
        report = run_probe(config, output, timeout=args.timeout, check_api=args.check_api, face=args.face)
    except (OSError, ValueError) as exc:
        print(f"探针输出失败：{exc}", file=sys.stderr)
        return 1
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
