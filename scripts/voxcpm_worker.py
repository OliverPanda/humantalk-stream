# -*- coding: utf-8 -*-
"""VoxCPM-0.5B 子进程合成 worker（GPU 声音克隆）

由主项目（FreeVideoAI，torch-CPU venv）通过 subprocess 调用，
必须用 VoxCPM 专属 venv 的 python 执行（含 torch-CUDA + voxcpm 包）：

    <voxcpm 环境的 python> voxcpm_worker.py <task.json>

与 Wav2Lip 同范式：模型每次调用加载一次，合成完进程退出，显存自动释放。

task.json 字段：
    text            必填，要合成的文案
    out_wav         必填，输出 wav 绝对路径（模型原生采样率，由调用方再转 16k）
    prompt_wav      可选，克隆参考音频绝对路径（零样本克隆）
    prompt_text     可选，参考音频的转写文本（有 prompt_wav 时必须提供）
    model_dir       可选，默认 workspace_data/runtime/models/VoxCPM-0.5B
    cfg_value       可选，默认 2.0
    inference_timesteps 可选，默认 10
    seed            可选，默认 42
    max_seg_chars   可选，长文本分段字数上限，默认 120
    gap_seconds     可选，段间静音，默认 0.25

输出协议：stdout 最后一行 "VOXCPM_RESULT:{json}"
    成功: {"success": true, "out_wav": ..., "duration": s, "sample_rate": hz,
           "segments": [{"text":..., "duration": s}], "load_time": s,
           "synth_time": s, "rtf": synth_time/duration}
    失败: {"success": false, "error": "..."} 且退出码非 0
"""
import io
import json
import os
import re
import sys
import time
from pathlib import Path

DEFAULT_MODEL = str(Path(__file__).resolve().parents[1]
                    / "workspace_data/runtime/models/VoxCPM-0.5B")
RESULT_PREFIX = "VOXCPM_RESULT:"


def _reconfigure_streams():
    """Windows 控制台默认 cp936，强制 utf-8 避免中文任务字段乱码"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def split_text(text: str, max_chars: int = 120) -> list:
    """按句号级标点切分长文本（与主项目 split_text_to_segments 语义一致的精简版）"""
    sentences = re.split(r"(?<=[。！？!?\n])", text)
    segments, buf = [], ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(buf) + len(s) <= max_chars:
            buf += s
        else:
            if buf:
                segments.append(buf)
            # 单句仍超长时按逗号再切，最后硬切
            if len(s) > max_chars:
                parts = re.split(r"(?<=[，,、；;])", s)
                cur = ""
                for p in parts:
                    if len(cur) + len(p) <= max_chars:
                        cur += p
                    else:
                        if cur:
                            segments.append(cur)
                        while len(p) > max_chars:
                            segments.append(p[:max_chars])
                            p = p[max_chars:]
                        cur = p
                buf = cur
            else:
                buf = s
    if buf:
        segments.append(buf)
    return segments or [text]


def fail(msg: str) -> None:
    print(RESULT_PREFIX + json.dumps({"success": False, "error": msg}, ensure_ascii=False), flush=True)
    sys.exit(1)


def main() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    _reconfigure_streams()
    if len(sys.argv) < 2:
        fail("usage: python voxcpm_worker.py <task.json>")
    task_path = sys.argv[1]
    if not os.path.exists(task_path):
        fail(f"task.json 不存在: {task_path}")
    with io.open(task_path, "r", encoding="utf-8") as f:
        task = json.load(f)

    text = (task.get("text") or "").strip()
    out_wav = task.get("out_wav")
    if not text or not out_wav:
        fail("task.json 缺少 text 或 out_wav")

    prompt_wav = task.get("prompt_wav") or None
    prompt_text = (task.get("prompt_text") or "").strip() or None
    if prompt_wav and not os.path.exists(prompt_wav):
        fail(f"prompt_wav 不存在: {prompt_wav}")
    if prompt_wav and not prompt_text:
        # VoxCPM 零样本克隆要求 prompt_wav / prompt_text 成对提供
        print("[warn] prompt_wav 提供但 prompt_text 为空，退化为非克隆合成", flush=True)
        prompt_wav = None

    model_dir = task.get("model_dir") or DEFAULT_MODEL
    if not os.path.exists(model_dir):
        fail(f"模型目录不存在: {model_dir}")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from krvoiceai.core.pro_models import require_model
    try:
        require_model("voxcpm", Path(model_dir))
    except RuntimeError as exc:
        fail(str(exc))
    cfg_value = float(task.get("cfg_value", 2.0))
    steps = int(task.get("inference_timesteps", 10))
    seed = int(task.get("seed", 42))
    max_seg_chars = int(task.get("max_seg_chars", 120))
    gap_seconds = float(task.get("gap_seconds", 0.25))

    import numpy as np
    import soundfile as sf
    import torch
    from voxcpm import VoxCPM

    print(f"[..] loading VoxCPM: {model_dir}", flush=True)
    t0 = time.time()
    # Disable torch.compile for portability (Windows and cards without Triton support).
    model = VoxCPM.from_pretrained(model_dir, load_denoiser=False, optimize=False)
    sr = model.tts_model.sample_rate
    load_time = time.time() - t0
    print(f"[ok] loaded {load_time:.1f}s sr={sr}", flush=True)

    segments = split_text(text, max_seg_chars)
    gap = np.zeros(int(gap_seconds * sr), dtype=np.float32)
    chunks, seg_meta = [], []
    t1 = time.time()
    for i, seg in enumerate(segments):
        torch.manual_seed(seed + i)
        wav = model.generate(
            text=seg,
            prompt_wav_path=prompt_wav,
            prompt_text=prompt_text,
            cfg_value=cfg_value,
            inference_timesteps=steps,
            normalize=False,
            denoise=False,
        )
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        if chunks:
            chunks.append(gap)
        chunks.append(wav)
        seg_dur = len(wav) / sr
        seg_meta.append({"text": seg, "duration": round(seg_dur, 3)})
        print(f"[ok] seg {i + 1}/{len(segments)} {seg_dur:.2f}s", flush=True)
    synth_time = time.time() - t1

    audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    os.makedirs(os.path.dirname(os.path.abspath(out_wav)), exist_ok=True)
    sf.write(out_wav, audio, sr)
    duration = len(audio) / sr

    result = {
        "success": True,
        "out_wav": out_wav,
        "duration": round(duration, 3),
        "sample_rate": sr,
        "segments": seg_meta,
        "load_time": round(load_time, 2),
        "synth_time": round(synth_time, 2),
        "rtf": round(synth_time / duration, 3) if duration > 0 else None,
    }
    print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # 任何异常都以协议格式报告，便于父进程解析
        fail(f"{type(e).__name__}: {e}")
