"""LatentSync 1.6 本地推理 Worker（beta，须在 ≥18GB 显存机器上运行）

契约与 musetalk_worker 对齐：--face(视频) + --audio + --outfile → 生成 mp4。
由 krvoiceai.modules.avatar_engine (provider=latentsync_local) 通过 subprocess 调用，
调用前主进程已过显存护栏（hardware.ensure_vram），小卡根本走不到这里。

⚠️ beta 状态：本 worker 依照 LatentSync 官方 README 的 CLI 契约编写
（https://github.com/bytedance/LatentSync，scripts.inference +
 configs/unet/stage2.yaml + checkpoints/latentsync_unet.pt），
尚未在真机联调。首次在 4090 机器上使用时如目录结构与官方新版有出入，
按 README 调整下方 UNET_CONFIG/CKPT 常量即可。

用法（在 LatentSync 自己的 venv 里）：
  python latentsync_worker.py --repo E:/LatentSync \
      --face ref_video.mp4 --audio in.wav --outfile out.mp4 --inference_steps 25
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# 官方仓库默认布局（新版若改名，改这里）
UNET_CONFIG = "configs/unet/stage2.yaml"
CKPT = "checkpoints/latentsync_unet.pt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="LatentSync 仓库根目录")
    ap.add_argument("--face", required=True, help="参考出镜视频（mp4）")
    ap.add_argument("--audio", required=True, help="输入音频（wav）")
    ap.add_argument("--outfile", required=True, help="输出 mp4 路径")
    ap.add_argument("--inference_steps", type=int, default=25)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    unet_config = repo / UNET_CONFIG
    ckpt = repo / CKPT
    if not unet_config.exists():
        print(f"[worker] UNet 配置缺失: {unet_config}（检查官方仓库结构）",
              file=sys.stderr, flush=True)
        return 2
    if not ckpt.exists():
        print(f"[worker] 权重缺失: {ckpt}（按官方 README 下载 checkpoints）",
              file=sys.stderr, flush=True)
        return 2

    outfile = Path(args.outfile).resolve()
    outfile.parent.mkdir(parents=True, exist_ok=True)

    # 官方推理入口：python -m scripts.inference（cwd=repo）
    cmd = [
        sys.executable, "-m", "scripts.inference",
        "--unet_config_path", str(unet_config),
        "--inference_ckpt_path", str(ckpt),
        "--inference_steps", str(args.inference_steps),
        "--video_path", str(Path(args.face).resolve()),
        "--audio_path", str(Path(args.audio).resolve()),
        "--video_out_path", str(outfile),
    ]
    print(f"[worker] LatentSync 推理: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(repo))
    if result.returncode != 0:
        return result.returncode

    if not outfile.exists() or outfile.stat().st_size == 0:
        print("[worker] 推理结束但输出不存在/为空", file=sys.stderr, flush=True)
        return 3
    print(f"[worker] 完成: {outfile} ({outfile.stat().st_size // 1024}KB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
