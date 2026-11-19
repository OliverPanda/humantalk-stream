"""多景别切换（虚拟机位）

零成本提升口播节奏感：同一条数字人视频裁出"近景机位"（放大裁切），
与原始"中景机位"按固定间隔交替硬切——观感上像双机位拍摄。
纯 ffmpeg CPU 操作，任意机器可用；音频取自原视频，全程连续不受切换影响。

实现：两遍式
1. 先产出近景变体（crop 放大回原分辨率，垂直偏上对准脸部）
2. concat demuxer 按 inpoint/outpoint 交替取两个文件的片段，重编码输出

config: multicam.enabled / switch_interval / closeup_crop / fade_frames
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from ..core.ffmpeg_utils import FFmpegRunner


def apply_multicam(
    video_path: Path,
    work_dir: Path,
    config,
    ffmpeg: FFmpegRunner,
    logger,
) -> Path:
    """对口播视频做虚拟双机位切换；任何失败都回退原视频（不阻断流水线）。"""
    interval = float(config.get("multicam.switch_interval", 12))
    # 下限钳制：interval<=0 会让切换段循环永不前进（死循环）；想关多景别请用 enabled=false
    interval = max(interval, 2.0)
    crop = float(config.get("multicam.closeup_crop", 0.25))
    crop = min(max(crop, 0.05), 0.5)

    info = ffmpeg.probe_video_info(video_path)
    duration = info.duration if info else 0
    if duration < interval * 2:
        logger.info(
            f"多景别跳过：视频 {duration:.1f}s 不足两个切换段（interval={interval:g}s）"
        )
        return video_path

    try:
        # 1. 近景变体：裁掉四周 crop 比例，垂直偏上（对准脸部），缩回原分辨率
        close_path = work_dir / "multicam_close.mp4"
        keep = 1.0 - crop
        vf = (
            f"crop=iw*{keep:.3f}:ih*{keep:.3f}:(iw-ow)/2:(ih-oh)*0.25,"
            f"scale=trunc(iw/{keep:.3f}/2)*2:trunc(ih/{keep:.3f}/2)*2"
        )
        subprocess.run(
            [ffmpeg.ffmpeg, "-y", "-i", str(video_path),
             "-vf", vf, "-an",
             "-c:v", "libx264", "-preset", "fast", "-crf", "18",
             str(close_path)],
            capture_output=True, check=True,
        )

        # 2. 交替片段清单（wide → close → wide ...），尾段吃满剩余时长
        list_path = work_dir / "multicam_list.txt"
        lines = []
        t = 0.0
        idx = 0
        while t < duration - 0.05:
            end = min(t + interval, duration)
            src = video_path if idx % 2 == 0 else close_path
            # concat demuxer 路径转义：单引号包裹 + 内部单引号转义
            p = str(Path(src).resolve()).replace("'", r"'\''")
            lines.append(f"file '{p}'")
            lines.append(f"inpoint {t:.3f}")
            lines.append(f"outpoint {end:.3f}")
            t = end
            idx += 1
        list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # 3. 拼接 + 原音频（音频连续，不参与切换）
        out_path = work_dir / "multicam_output.mp4"
        subprocess.run(
            [ffmpeg.ffmpeg, "-y",
             "-f", "concat", "-safe", "0", "-i", str(list_path),
             "-i", str(video_path),
             "-map", "0:v", "-map", "1:a?",
             "-c:v", "libx264", "-preset", "fast", "-crf", "18",
             "-c:a", "aac", "-b:a", "192k",
             "-shortest",
             str(out_path)],
            capture_output=True, check=True,
        )
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise RuntimeError("多景别输出为空")
        logger.info(
            f"多景别切换完成：{idx} 段，interval={interval:g}s，近景放大 {1/keep:.2f}x"
        )
        return out_path
    except Exception as e:
        logger.warning(f"多景别切换失败，使用原视频: {e}")
        return video_path
