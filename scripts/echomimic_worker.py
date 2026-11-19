"""EchoMimic V3 本地推理 Worker（beta，须在 ≥12GB 显存机器上运行）

契约与 musetalk_worker 对齐：--face(照片) + --audio + --outfile → 生成 mp4。
由 krvoiceai.modules.avatar_engine (provider=echomimic_local) 通过 subprocess 调用，
调用前主进程已过显存护栏（hardware.ensure_vram），小卡根本走不到这里。

⚠️ beta 状态：EchoMimic V3（https://github.com/antgroup/echomimic_v3）的推理入口
是 infer.py（图 + 音频 → 半身驱动视频），权重从 HuggingFace BadToBest/EchoMimicV3
下载到 <repo>/models/。官方 CLI 参数随版本演进较快，本 worker 采用
"探测入口脚本→透传核心参数"的宽松策略，首次在真机使用时按官方 README 校准。

用法（在 echomimic_v3 自己的 venv 里）：
  python echomimic_worker.py --repo E:/echomimic_v3 \
      --face photo.jpg --audio in.wav --outfile out.mp4 --sample_steps 5
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# 官方入口脚本候选（按版本先后探测）
ENTRY_CANDIDATES = ["infer.py", "inference.py", "app_infer.py"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="echomimic_v3 仓库根目录")
    ap.add_argument("--face", required=True, help="参考照片（jpg/png）或视频")
    ap.add_argument("--audio", required=True, help="输入音频（wav）")
    ap.add_argument("--outfile", required=True, help="输出 mp4 路径")
    ap.add_argument("--sample_steps", type=int, default=5)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    entry = None
    for name in ENTRY_CANDIDATES:
        if (repo / name).exists():
            entry = repo / name
            break
    if entry is None:
        print(f"[worker] 未找到推理入口（试过 {ENTRY_CANDIDATES}），"
              f"检查 {repo} 是否为官方仓库", file=sys.stderr, flush=True)
        return 2
    if not (repo / "models").exists():
        print(f"[worker] 权重目录缺失: {repo / 'models'}"
              f"（从 HuggingFace BadToBest/EchoMimicV3 下载）",
              file=sys.stderr, flush=True)
        return 2

    outfile = Path(args.outfile).resolve()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    out_dir = outfile.parent / "echomimic_out"
    out_dir.mkdir(exist_ok=True)

    cmd = [
        sys.executable, str(entry),
        "--image_path", str(Path(args.face).resolve()),
        "--audio_path", str(Path(args.audio).resolve()),
        "--output_dir", str(out_dir),
        "--sample_steps", str(args.sample_steps),
    ]
    print(f"[worker] EchoMimic V3 推理: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(repo))
    if result.returncode != 0:
        return result.returncode

    # 官方按 output_dir 落盘，取最新 mp4 搬到契约路径
    mp4s = sorted(out_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not mp4s:
        print("[worker] 推理结束但 output_dir 无 mp4", file=sys.stderr, flush=True)
        return 3
    mp4s[-1].replace(outfile)
    print(f"[worker] 完成: {outfile} ({outfile.stat().st_size // 1024}KB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
