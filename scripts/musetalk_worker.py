"""MuseTalk 本地推理 Worker（由独立环境 E:\\musetalk_env 的 Python 执行）

契约与 Wav2Lip inference.py 对齐：--face + --audio + --outfile → 生成 mp4。
由 krvoiceai.modules.avatar_engine (provider=musetalk_local) 通过 subprocess 调用。

职责：
1. 显存闸门：等待 GPU 空闲显存 >= min_free_vram_mb（另一进程可能短暂占卡）
2. 生成 MuseTalk 任务 yaml → runpy 调官方 scripts.inference（cwd=repo 根，模型相对路径）
3. 官方 inference.py 对单任务异常只打印不抛错（returncode 恒 0），
   本 worker 以“输出文件存在且非空”为准判定成败，失败非零退出
4. OOM 兜底：首次失败自动按 batch_size 减半重试一次

用法（示例）：
  E:\\musetalk_env\\Scripts\\python.exe musetalk_worker.py \
      --repo E:\\MuseTalk --face face.jpg --audio in.wav --outfile out.mp4 \
      --batch_size 4 --use_float16
"""
from __future__ import annotations

import argparse
import os
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def query_free_vram_mb() -> int | None:
    """通过 nvidia-smi 查询 GPU0 空闲显存（MB）。查询失败返回 None。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode != 0:
            return None
        line = out.stdout.strip().splitlines()[0]
        return int(re.search(r"\d+", line).group())
    except Exception:
        return None


def wait_for_vram(min_free_mb: int, retries: int, wait_seconds: int) -> bool:
    """等待空闲显存达标。nvidia-smi 不可用时放行（由后续推理自行报错）。"""
    for attempt in range(retries + 1):
        free = query_free_vram_mb()
        if free is None:
            print(f"[worker] nvidia-smi 不可用，跳过显存闸门", flush=True)
            return True
        if free >= min_free_mb:
            print(f"[worker] 显存闸门通过: free={free}MB >= {min_free_mb}MB", flush=True)
            return True
        if attempt < retries:
            print(f"[worker] 空闲显存不足 ({free}MB < {min_free_mb}MB)，"
                  f"等待 {wait_seconds}s 后重试 ({attempt + 1}/{retries})", flush=True)
            time.sleep(wait_seconds)
    print(f"[worker] 显存闸门超时: 空闲显存持续不足 {min_free_mb}MB", flush=True)
    return False


def normalize_face_input(face: Path, work_dir: Path) -> Path:
    """规整参考素材尺寸为偶数宽高（libx264 要求，官方脚本不处理奇数尺寸）

    图片：PIL 裁到偶数；视频：ffmpeg crop 到偶数。失败时原样返回（由推理自行报错）。
    """
    try:
        if face.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            from PIL import Image
            img = Image.open(face)
            w, h = img.size
            if w % 2 == 0 and h % 2 == 0:
                return face
            fixed = work_dir / f"face_even{face.suffix.lower()}"
            img.crop((0, 0, w // 2 * 2, h // 2 * 2)).save(str(fixed), quality=95)
            print(f"[worker] 参考图奇数尺寸 {w}x{h} -> 裁为偶数 {w//2*2}x{h//2*2}",
                  flush=True)
            return fixed
        if face.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv"):
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height", "-of", "csv=p=0",
                 str(face)],
                capture_output=True, text=True, timeout=60,
            )
            w, h = (int(x) for x in probe.stdout.strip().split(",")[:2])
            if w % 2 == 0 and h % 2 == 0:
                return face
            fixed = work_dir / "face_even.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", str(face),
                 "-vf", "crop=floor(iw/2)*2:floor(ih/2)*2",
                 "-c:v", "libx264", "-crf", "18", "-an", str(fixed)],
                capture_output=True, timeout=600, check=True,
            )
            print(f"[worker] 参考视频奇数尺寸 {w}x{h} -> 裁为偶数", flush=True)
            return fixed
    except Exception as e:
        print(f"[worker] 参考素材尺寸规整失败（继续用原素材）: {e}", flush=True)
    return face


def run_musetalk_once(args, batch_size: int, work_dir: Path) -> Path | None:
    """跑一次官方 scripts.inference，成功返回输出 mp4 路径，失败返回 None。"""
    repo = Path(args.repo).resolve()
    face = normalize_face_input(Path(args.face).resolve(), work_dir)
    audio = Path(args.audio).resolve()
    result_dir = work_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    # 生成任务 yaml（官方 inference_config 格式）
    task_yaml = work_dir / "task.yaml"
    task_yaml.write_text(
        "task_0:\n"
        f" video_path: \"{face.as_posix()}\"\n"
        f" audio_path: \"{audio.as_posix()}\"\n",
        encoding="utf-8",
    )

    if args.version == "v15":
        unet_model_path = repo / "models" / "musetalkV15" / "unet.pth"
        unet_config = repo / "models" / "musetalkV15" / "musetalk.json"
    else:
        unet_model_path = repo / "models" / "musetalk" / "pytorch_model.bin"
        unet_config = repo / "models" / "musetalk" / "musetalk.json"

    out_name = "musetalk_out.mp4"
    argv = [
        "scripts.inference",
        "--inference_config", str(task_yaml),
        "--result_dir", str(result_dir),
        "--unet_model_path", str(unet_model_path),
        "--unet_config", str(unet_config),
        "--whisper_dir", str(repo / "models" / "whisper"),
        "--version", args.version,
        "--batch_size", str(batch_size),
        "--fps", str(args.fps),
        "--extra_margin", str(args.extra_margin),
        "--parsing_mode", args.parsing_mode,
        "--output_vid_name", out_name,
    ]
    if args.use_float16:
        argv.append("--use_float16")

    # 官方推理脚本依赖 cwd=repo 根目录（models/、musetalk/utils/ 相对路径）
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    old_argv = sys.argv
    sys.argv = argv
    t0 = time.time()
    try:
        runpy.run_module("scripts.inference", run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            print(f"[worker] scripts.inference 退出码 {e.code}", flush=True)
    except BaseException as e:
        # OOM/加载失败等：不让异常终止 worker，走降 batch 重试或 fallback
        print(f"[worker] scripts.inference 异常: {type(e).__name__}: {e}", flush=True)
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    finally:
        sys.argv = old_argv
    elapsed = time.time() - t0

    # 官方脚本吞异常：以输出文件为准
    produced = result_dir / args.version / out_name
    if produced.exists() and produced.stat().st_size > 10240:
        print(f"[worker] 推理完成 batch={batch_size} 耗时={elapsed:.1f}s "
              f"输出={produced} ({produced.stat().st_size // 1024}KB)", flush=True)
        return produced
    print(f"[worker] 推理失败（无输出文件）batch={batch_size} 耗时={elapsed:.1f}s", flush=True)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="MuseTalk local worker")
    parser.add_argument("--repo", required=True, help="MuseTalk 仓库根目录")
    parser.add_argument("--face", required=True, help="参考人脸图片或视频")
    parser.add_argument("--audio", required=True, help="驱动音频 wav")
    parser.add_argument("--outfile", required=True, help="输出 mp4 路径")
    parser.add_argument("--version", default="v15", choices=["v1", "v15"])
    parser.add_argument("--batch_size", type=int, default=4,
                        help="推理 batch（6GB 显存建议 <=4）")
    parser.add_argument("--use_float16", action="store_true", default=False)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--extra_margin", type=int, default=10)
    parser.add_argument("--parsing_mode", default="jaw")
    parser.add_argument("--min_free_vram_mb", type=int, default=4096)
    parser.add_argument("--vram_wait_retries", type=int, default=10)
    parser.add_argument("--vram_wait_seconds", type=int, default=60)
    args = parser.parse_args()

    outfile = Path(args.outfile).resolve()
    outfile.parent.mkdir(parents=True, exist_ok=True)

    # 1. 显存闸门
    if not wait_for_vram(args.min_free_vram_mb, args.vram_wait_retries,
                         args.vram_wait_seconds):
        return 3

    # 2. 推理（OOM/失败自动降 batch 重试一次）
    # 注意：不能用系统 %TEMP%（含中文用户名），cv2.imwrite 在 Windows
    # 非 ASCII 路径下会静默失败 → 放 MuseTalk 仓库下的 results/_jobs
    job_root = Path(args.repo).resolve() / "results" / "_jobs"
    job_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="musetalk_job_", dir=str(job_root)))
    try:
        produced = run_musetalk_once(args, args.batch_size, work_dir)
        if produced is None and args.batch_size > 1:
            retry_batch = max(args.batch_size // 2, 1)
            print(f"[worker] 降 batch 重试: {args.batch_size} -> {retry_batch}", flush=True)
            produced = run_musetalk_once(args, retry_batch, work_dir)
        if produced is None:
            print("[worker] MuseTalk 推理最终失败", flush=True)
            return 2
        shutil.move(str(produced), str(outfile))
        print(f"[worker] OK {outfile}", flush=True)
        return 0
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
