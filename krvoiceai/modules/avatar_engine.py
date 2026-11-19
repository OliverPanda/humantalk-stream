"""数字人口播生成模块

provider：
- musetalk_local: 本地 MuseTalk 1.5（GPU，近实时口型同步，默认；独立环境 musetalk_env）
                  失败自动降级 wav2lip（fallback 链 musetalk_local→wav2lip）
- wav2lip:     本地 Wav2Lip 唇形同步（CPU 可跑，输入真人照片/视频+音频→嘴唇会动）
- musetalk:    调用云端 MuseTalk API（口型同步）
- latentsync:  调用云端 LatentSync API（备选）
- mock:        音频 + 静态占位图合成视频（保证流程可跑通）

输出：口播视频 mp4 文件
"""
from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from ..core.base_module import BaseModule, JobContext, ModuleResult
from ..core.ffmpeg_utils import FFmpegRunner
from ..core.gpu_runner import GPURunner


class AvatarEngine(BaseModule):
    """数字人口播生成模块"""

    name = "avatar"
    requires_gpu = False  # wav2lip CPU 也可跑

    def __init__(self, config=None, gpu_runner: GPURunner | None = None,
                 ffmpeg: FFmpegRunner | None = None):
        super().__init__(config)
        self.provider = self.config.get("avatar.provider", "mock")
        self.api_base = self.config.get("avatar.api_base", "")
        self.avatars_dir = Path(self.config.get("avatar.avatars_dir", "./config/avatars"))
        self.default_avatar = self.config.get("avatar.default_avatar", "default")
        self.output_fps = self.config.get("avatar.output_fps", 25)
        res = self.config.get("avatar.output_resolution", [1080, 1920])
        self.output_resolution = tuple(res) if isinstance(res, list) else (1080, 1920)
        # Wav2Lip 配置
        self.wav2lip_config = self.config.get("avatar.wav2lip", {})
        self.wav2lip_checkpoint = self.wav2lip_config.get(
            "checkpoint_path", "./Wav2Lip/checkpoints/wav2lip_gan.pth"
        )
        self.wav2lip_env_python = self.wav2lip_config.get(
            "env_python", "../wav2lip_env/Scripts/python.exe"
        )
        self.wav2lip_inference_script = self.wav2lip_config.get(
            "inference_script", "../Wav2Lip/inference.py"
        )
        # MuseTalk 本地配置（GPU 独立环境）
        self.musetalk_config = self.config.get("avatar.musetalk_local", {}) or {}
        # 环境就绪缓存（仅缓存成功结果；深度检查含独立环境 torch import，约 10-30s）
        self._musetalk_env_ok = False
        # 微动作配置
        self.micro_motion_cfg = self.config.get("avatar.micro_motion", {}) or {}
        self.gpu = gpu_runner or GPURunner()
        self.ffmpeg = ffmpeg or FFmpegRunner()

    def setup(self) -> None:
        if self.provider == "musetalk_local":
            # 本地 MuseTalk（GPU 独立环境）：未就绪时警告，运行时自动降级 Wav2Lip
            ready, reason = self._check_musetalk_env()
            if not ready:
                self.logger.warning(
                    f"MuseTalk 本地环境未就绪: {reason}。"
                    f"运行时将自动降级 Wav2Lip（fallback 链 musetalk_local→wav2lip）。"
                )
            else:
                self.logger.info(
                    f"数字人模块初始化 provider=musetalk_local "
                    f"repo={self.musetalk_config.get('repo_dir', '../MuseTalk')} "
                    f"batch={self.musetalk_config.get('batch_size', 4)} "
                    f"fp16={self.musetalk_config.get('use_float16', True)}"
                )
        elif self.provider == "wav2lip":
            # Wav2Lip 真实运行需要独立 Python3.8 环境（torch 1.13 等）
            # 不再静默降级到 mock —— 用户明确要求真实唇形同步
            ready, reason = self._check_wav2lip_env()
            if not ready:
                # 环境未就绪：保持 provider=wav2lip，run() 时会给出明确报错
                self.logger.warning(
                    f"Wav2Lip 环境未就绪: {reason}。"
                    f"请运行 scripts/setup_wav2lip_env.bat 安装，"
                    f"或将 avatar.provider 改为 mock 跳过真实唇形同步。"
                )
            else:
                self.logger.info(
                    f"数字人模块初始化 provider=wav2lip "
                    f"checkpoint={Path(self.wav2lip_checkpoint).name} "
                    f"env={Path(self.wav2lip_env_python).name}"
                )
        elif self.provider == "omnihuman_api":
            import os
            if os.environ.get("OMNIHUMAN_API_BASE") and os.environ.get("OMNIHUMAN_API_KEY"):
                self.logger.info("数字人模块初始化 provider=omnihuman_api（中转站）")
            else:
                self.logger.warning(
                    "OmniHuman 中转站未配置地址或密钥，运行时将报错。"
                )
        elif self.provider in ("musetalk", "latentsync", "echomimic"):
            available = self.gpu.health_check_avatar()
            if not available:
                self.logger.warning(
                    f"{self.provider} 云端服务不可用，将报错（不再静默降级 mock）"
                )
            else:
                self.logger.info(f"数字人模块初始化 provider={self.provider}")
        else:
            self.logger.info(f"数字人模块初始化 provider={self.provider}")
        super().setup()

    def _check_wav2lip_env(self) -> tuple[bool, str]:
        """检查 Wav2Lip 独立运行环境是否就绪

        不再静默降级。返回 (ready, reason)。
        ready=True 表示可真实推理；ready=False 时 reason 给出具体缺失项和指引。
        """
        from ..core.config import resolve_path as _abs

        # 1. 独立 Python 环境存在
        env_python = _abs(self.wav2lip_env_python)
        if not env_python.exists():
            return False, (
                f"独立 Python 环境不存在: {env_python}。"
                f"请运行 scripts/setup_wav2lip_env.bat 创建 wav2lip_env。"
            )

        # 2. 推理脚本存在
        script = _abs(self.wav2lip_inference_script)
        if not script.exists():
            return False, f"Wav2Lip 推理脚本不存在: {script}"

        # 3. 模型权重存在
        ckpt = _abs(self.wav2lip_checkpoint)
        if not ckpt.exists():
            return False, (
                f"模型权重不存在: {ckpt}。"
                f"请从 hf-mirror 下载 wav2lip_gan.pth。"
            )

        # 4. 依赖（torch/librosa）能在独立环境 import
        try:
            result = subprocess.run(
                [str(env_python), "-c",
                 "import torch, librosa, cv2, numpy, scipy; "
                 "print('deps ok', torch.__version__)"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                return False, (
                    f"wav2lip_env 依赖缺失: {result.stderr[-200:]}。"
                    f"请在该环境执行 pip install torch librosa opencv-python scipy。"
                )
        except Exception as e:
            return False, f"检测 wav2lip_env 依赖异常: {e}"

        return True, "ok"

    def _check_musetalk_env(self) -> tuple[bool, str]:
        """检查 MuseTalk 本地独立运行环境是否就绪

        风格与 _check_wav2lip_env 一致：返回 (ready, reason)。
        ready=False 时 reason 给出具体缺失项和指引（不静默降级，由 run() 显式 fallback）。
        成功结果缓存（避免每次 job 重复付出独立环境 torch import 的 10-30s）。
        """
        if self._musetalk_env_ok:
            return True, "ok (cached)"
        from ..core.config import resolve_path as _abs

        # 1. 独立 Python 环境存在
        env_python = _abs(self.musetalk_config.get(
            "env_python", "../musetalk_env/Scripts/python.exe"))
        if not env_python.exists():
            return False, (
                f"独立 Python 环境不存在: {env_python}。"
                f"请参考 _upgrade/A2_数字人报告.md 创建 musetalk_env。"
            )

        # 2. Worker 脚本存在
        worker = _abs(self.musetalk_config.get(
            "worker_script", "./scripts/musetalk_worker.py"))
        if not worker.exists():
            return False, f"MuseTalk worker 脚本不存在: {worker}"

        # 3. 仓库与关键权重存在
        repo = _abs(self.musetalk_config.get("repo_dir", "../MuseTalk"))
        if not (repo / "scripts" / "inference.py").exists():
            return False, f"MuseTalk 仓库不存在或不完整: {repo}"
        required_weights = [
            repo / "models" / "musetalkV15" / "unet.pth",
            repo / "models" / "sd-vae" / "diffusion_pytorch_model.bin",
            repo / "models" / "whisper" / "pytorch_model.bin",
            repo / "models" / "dwpose" / "dw-ll_ucoco_384.pth",
            repo / "models" / "face-parse-bisent" / "79999_iter.pth",
        ]
        missing = [str(p) for p in required_weights if not p.exists()]
        if missing:
            return False, f"MuseTalk 模型权重缺失: {missing[0]} 等 {len(missing)} 个"

        # 4. 依赖与 CUDA 能在独立环境使用（GPU 不可用则宁可降级 Wav2Lip）
        try:
            result = subprocess.run(
                [str(env_python), "-c",
                 "import torch, cv2, transformers, mmpose, diffusers; "
                 "assert torch.cuda.is_available(), 'CUDA not available'; "
                 "print('deps ok', torch.__version__)"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                return False, (
                    f"musetalk_env 依赖/CUDA 检查失败: {result.stderr[-200:]}"
                )
        except Exception as e:
            return False, f"检测 musetalk_env 依赖异常: {e}"

        self._musetalk_env_ok = True
        return True, "ok"

    def _check_wav2lip_deps(self) -> bool:
        """[已废弃] 检查主环境依赖 —— 改用 _check_wav2lip_env 检测独立环境

        保留仅为向后兼容。Wav2Lip 现在用独立 Python 3.8 环境，主环境无需这些依赖。
        """
        return False

    def run(self, ctx: JobContext) -> ModuleResult:
        """根据 ctx.audio_path 生成口播视频"""
        if not ctx.audio_path or not ctx.audio_path.exists():
            return ModuleResult(success=False, error="无音频文件，无法生成数字人视频")

        avatar_id = ctx.avatar_id or self.default_avatar
        output_path = ctx.work_dir / "avatar_output.mp4"

        try:
            start = time.time()
            if self.provider == "musetalk_local":
                # fallback 链：musetalk_local → wav2lip（失败降级，日志写明原因）
                video_path = None
                ready, reason = self._check_musetalk_env()
                if ready:
                    try:
                        video_path = self._generate_musetalk_local(
                            ctx, avatar_id, output_path)
                        ctx.metadata["avatar_backend"] = "musetalk_local"
                    except Exception as e:
                        self.logger.warning(
                            f"MuseTalk 本地推理失败，降级 Wav2Lip: {e}")
                else:
                    self.logger.warning(
                        f"MuseTalk 环境未就绪，降级 Wav2Lip: {reason}")
                if video_path is None:
                    w_ready, w_reason = self._check_wav2lip_env()
                    if not w_ready:
                        return ModuleResult(
                            success=False,
                            error=(
                                f"musetalk_local 失败且 Wav2Lip fallback "
                                f"未就绪: {w_reason}"
                            ),
                        )
                    self.logger.info("使用 Wav2Lip fallback 生成数字人")
                    video_path = self._generate_wav2lip(
                        ctx, avatar_id, output_path)
                    ctx.metadata["avatar_backend"] = "wav2lip_fallback"
            elif self.provider == "wav2lip":
                # 真实 Wav2Lip：环境未就绪必须报错（不静默降级）
                ready, reason = self._check_wav2lip_env()
                if not ready:
                    return ModuleResult(
                        success=False,
                        error=f"Wav2Lip 环境未就绪: {reason}",
                    )
                video_path = self._generate_wav2lip(ctx, avatar_id, output_path)
            elif self.provider == "omnihuman_api":
                # 即梦数字人云端（OmniHuman 全身驱动）。云端失败不静默降级本地
                # （A4 教训：静默降级会掩盖主档故障），报错并给出人话原因。
                video_path = self._generate_omnihuman(ctx, avatar_id, output_path)
                ctx.metadata["avatar_backend"] = "omnihuman_api"
            elif self.provider in ("latentsync_local", "echomimic_local"):
                # 4090 级本地档：先过显存护栏（不够档立即拒绝，绝不试跑），
                # 被拦截时降级 musetalk_local 链并在日志/metadata 写明原因。
                video_path = self._generate_pro_local(ctx, avatar_id, output_path)
            elif self.provider == "mock":
                video_path = self._generate_mock(ctx, avatar_id, output_path)
            else:
                video_path = self._generate_cloud(ctx, avatar_id, output_path)

            # 微动作后处理（可选，缓解静态照片/低质量数字人的恐怖谷）
            if self.micro_motion_cfg.get("enabled", False):
                video_path = self._apply_micro_motion(video_path, ctx)

            # 多景别切换（可选，虚拟双机位交替，提升节奏感；失败自动回退原视频）
            if self.config.get("multicam.enabled", False):
                from .multicam import apply_multicam
                video_path = apply_multicam(
                    video_path, ctx.work_dir, self.config,
                    self.ffmpeg, self.logger,
                )

            ctx.raw_video_path = video_path
            ctx.metadata["avatar_provider"] = self.provider

            # 探测视频信息
            info = self.ffmpeg.probe_video_info(video_path)
            duration = info.duration if info else ctx.audio_duration

            elapsed = time.time() - start
            self.logger.info(f"数字人生成完成 provider={self.provider} 耗时={elapsed:.1f}s")

            return ModuleResult(
                success=True,
                data={
                    "video_path": str(video_path),
                    "duration": duration,
                    "avatar_id": avatar_id,
                    "provider": self.provider,
                    "elapsed": elapsed,
                },
            )
        except Exception as e:
            return ModuleResult(success=False, error=str(e))

    def _to_portrait_or_original(self, video_path: Path, ctx: JobContext) -> Path:
        """输出统一收尾：转竖屏（人脸居中+模糊背景）；失败回退原视频不阻断。

        所有 provider 分支共用，保证输出契约一致（分辨率/fps/背景处理）。
        """
        portrait_path = ctx.work_dir / "avatar_portrait.mp4"
        try:
            self.ffmpeg.to_portrait(
                video=video_path,
                output=portrait_path,
                target_resolution=self.output_resolution,
                fps=self.output_fps,
                background="blur",
            )
            return portrait_path
        except Exception as e:
            self.logger.warning(f"竖屏转换失败，使用原始输出: {e}")
            return video_path

    def _generate_musetalk_local(
        self, ctx: JobContext, avatar_id: str, output_path: Path
    ) -> Path:
        """使用本地 MuseTalk 1.5（GPU 独立环境）生成唇形同步视频

        与 Wav2Lip 相同的 subprocess 隔离范式：
        由 musetalk_env 的 Python 执行 scripts/musetalk_worker.py，
        worker 内部完成显存闸门、官方推理调用与 OOM 降 batch 重试。
        输出契约与 Wav2Lip 一致：mp4 → 竖屏转换。
        """
        self.logger.info(
            f"MuseTalk 唇形同步 avatar={avatar_id} audio={ctx.audio_path.name} "
            f"duration={ctx.audio_duration:.1f}s"
        )

        # 获取参考人脸（照片或视频，优先视频）
        face_path = self._get_avatar_reference(avatar_id)
        if not face_path:
            raise RuntimeError(
                f"数字人 {avatar_id} 无参考照片/视频，请先上传真人照片或视频注册形象"
            )
        self.logger.info(f"参考人脸: {face_path}")

        # 准备音频（MuseTalk 需要 wav 格式）
        audio_path = ctx.audio_path
        if audio_path.suffix.lower() != ".wav":
            wav_path = ctx.work_dir / "musetalk_input.wav"
            self.ffmpeg.convert_audio(audio_path, wav_path)
            audio_path = wav_path

        from ..core.config import resolve_path as _abs

        cfg = self.musetalk_config
        env_python = _abs(cfg.get("env_python", "../musetalk_env/Scripts/python.exe"))
        worker = _abs(cfg.get("worker_script", "./scripts/musetalk_worker.py"))
        repo = _abs(cfg.get("repo_dir", "../MuseTalk"))
        timeout = int(cfg.get("timeout", 1800))

        cmd = [
            str(env_python), str(worker),
            "--repo", str(repo),
            "--face", str(Path(face_path).resolve()),
            "--audio", str(Path(audio_path).resolve()),
            "--outfile", str(Path(output_path).resolve()),
            "--version", str(cfg.get("version", "v15")),
            "--batch_size", str(cfg.get("batch_size", 4)),
            "--fps", str(cfg.get("fps", self.output_fps)),
            "--extra_margin", str(cfg.get("extra_margin", 10)),
            "--parsing_mode", str(cfg.get("parsing_mode", "jaw")),
            "--min_free_vram_mb", str(cfg.get("min_free_vram_mb", 4096)),
            "--vram_wait_retries", str(cfg.get("vram_wait_retries", 10)),
            "--vram_wait_seconds", str(cfg.get("vram_wait_seconds", 60)),
        ]
        if cfg.get("use_float16", True):
            cmd.append("--use_float16")

        self.logger.info(
            f"运行 MuseTalk 推理 (GPU, env={env_python.parent.parent.name})，"
            f"音频 {ctx.audio_duration:.1f}s，近实时（含模型加载约 1-3 分钟）..."
        )
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(repo),
        )

        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "")[-500:]
            self.logger.error(f"MuseTalk 推理失败(code={result.returncode}): {tail}")
            raise RuntimeError(f"MuseTalk 推理失败: {tail[-300:]}")

        if not output_path.exists():
            raise RuntimeError("MuseTalk 推理完成但输出文件不存在")

        self.logger.info(
            f"MuseTalk 唇形同步完成: {output_path.name} "
            f"({output_path.stat().st_size // 1024}KB)"
        )

        # 转竖屏（人脸居中 + 模糊背景，与 Wav2Lip 输出契约一致）
        return self._to_portrait_or_original(output_path, ctx)

    def _generate_omnihuman(
        self, ctx: JobContext, avatar_id: str, output_path: Path
    ) -> Path:
        """即梦数字人云端（OmniHuman）：一张照片 + 音频 → 全身驱动数字人视频

        任意机器可用（无显存要求）。鉴权走 OmniHuman 中转站的 API Key，
        未配置时报人话错误。提交-轮询-下载逻辑在 omnihuman_client。
        """
        from .omnihuman_client import OmniHumanClient

        # OmniHuman 输入是单张照片：优先 reference.jpg / preview.jpg（视频参考抽帧）
        avatar_dir = self.avatars_dir / avatar_id
        image_path = None
        for name in ("reference.jpg", "preview.jpg", "reference.png", "avatar.jpg"):
            p = avatar_dir / name
            if p.exists():
                image_path = p
                break
        if image_path is None:
            # 只有视频参考：抽第一帧
            ref = self._get_avatar_reference(avatar_id)
            if ref and Path(ref).suffix.lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
                image_path = ctx.work_dir / "omnihuman_ref.jpg"
                subprocess.run(
                    [self.ffmpeg.ffmpeg, "-y", "-i", str(ref),
                     "-frames:v", "1", "-q:v", "2", str(image_path)],
                    capture_output=True, check=True,
                )
        if image_path is None or not Path(image_path).exists():
            raise RuntimeError(
                f"数字人 {avatar_id} 无参考照片，请先在'我的形象'上传照片"
            )

        # 超长音频直接拒绝：截断会让字幕/合成按全长音频走、成片被 -shortest
        # 砍成残片却报成功（且已按秒计费）。诚实报错让用户改文案或换引擎。
        max_dur = float(self.config.get("avatar.omnihuman.max_duration", 60))
        audio_path = ctx.audio_path
        if ctx.audio_duration and ctx.audio_duration > max_dur:
            raise RuntimeError(
                f"口播音频 {ctx.audio_duration:.0f}s 超过即梦数字人单段上限 "
                f"{max_dur:.0f}s（官方建议 ≤15s 效果最佳）。"
                f"请缩短文案，或将数字人引擎切回本地 MuseTalk 生成长口播。"
            )

        client = OmniHumanClient(self.config, self.logger)
        # 提交后立即落盘，跨进程重试也复用已计费任务。
        task_file = ctx.work_dir / "omnihuman_task_id.txt"
        submitting_file = ctx.work_dir / "omnihuman_submitting.txt"
        model_file = ctx.work_dir / "omnihuman_task_model.txt"
        resume_task = ctx.metadata.get("omnihuman_task_id")
        if task_file.exists():
            resume_task = task_file.read_text(encoding="utf-8").strip()
            if not resume_task:
                raise RuntimeError(f"OmniHuman 任务记录为空，停止重新提交: {task_file}")
            ctx.metadata["omnihuman_task_id"] = resume_task
        if resume_task and model_file.exists():
            client.version = model_file.read_text(encoding="utf-8").strip()
            if client.version not in ("v15", "v1"):
                raise RuntimeError(f"OmniHuman 任务模型记录无效，停止续查: {model_file}")
        if submitting_file.exists() and not resume_task:
            raise RuntimeError(
                f"OmniHuman 提交结果不明，已停止重新提交以避免重复计费；"
                f"请先核查中转站任务记录: {submitting_file}")
        if resume_task:
            self.logger.info(
                f"检测到上次未完成的即梦任务 {resume_task}，续轮询（不重新提交/计费）")
        else:
            self.logger.info(
                f"提交即梦数字人任务（OmniHuman 云端）：image={Path(image_path).name} "
                f"audio={Path(audio_path).name}，排队+生成通常需数分钟..."
            )
        def remember_task(task_id: str) -> None:
            ctx.metadata["omnihuman_task_id"] = task_id
            pending = task_file.with_suffix(".tmp")
            pending.write_text(task_id, encoding="utf-8")
            os.replace(pending, task_file)
            submitting_file.unlink(missing_ok=True)

        def remember_submission() -> None:
            model_pending = model_file.with_suffix(".tmp")
            model_pending.write_text(client.version, encoding="utf-8")
            os.replace(model_pending, model_file)
            pending = submitting_file.with_suffix(".tmp")
            pending.write_text("submitting", encoding="utf-8")
            os.replace(pending, submitting_file)

        if not output_path.exists() or not output_path.stat().st_size:
            client.generate(
                Path(image_path), Path(audio_path), output_path,
                resume_task_id=resume_task,
                on_video_task=remember_task,
                on_video_submit_start=remember_submission,
            )
        if not output_path.exists() or not output_path.stat().st_size:
            raise RuntimeError("OmniHuman 任务完成但输出文件不存在")

        # 转竖屏（与本地引擎输出契约一致）
        result = self._to_portrait_or_original(output_path, ctx)
        task_file.unlink(missing_ok=True)
        submitting_file.unlink(missing_ok=True)
        model_file.unlink(missing_ok=True)
        ctx.metadata.pop("omnihuman_task_id", None)
        return result

    def _generate_pro_local(
        self, ctx: JobContext, avatar_id: str, output_path: Path
    ) -> Path:
        """4090 级本地档（latentsync_local / echomimic_local）

        铁律：先过显存护栏——不够档立即拒绝并降级 MuseTalk 链，绝不在小卡上
        尝试加载大模型（防 OOM 拖垮整条流水线，也是"功能可以有但别真跑"的落点）。
        环境未安装同样降级（安装指引见模型管理页）。
        ⚠️ worker 为 beta：尚未在 ≥16GB 真机联调，首跑失败会自动降级并记录原因。
        """
        from ..core import hardware as hw

        pretty = ("LatentSync 1.6" if self.provider == "latentsync_local"
                  else "EchoMimic V3")
        cfg = self.config.get(f"avatar.{self.provider}", {}) or {}
        min_vram = float(cfg.get("min_vram_gb",
                                 18 if self.provider == "latentsync_local" else 12))

        def _degrade(reason: str) -> Path:
            self.logger.warning(f"{pretty} 不可用: {reason} → 降级 MuseTalk")
            ctx.metadata["avatar_degraded_from"] = self.provider
            ctx.metadata["avatar_degrade_reason"] = reason
            ready, m_reason = self._check_musetalk_env()
            if not ready:
                raise RuntimeError(
                    f"{pretty} 不可用（{reason}）且 MuseTalk 未就绪（{m_reason}）"
                )
            path = self._generate_musetalk_local(ctx, avatar_id, output_path)
            ctx.metadata["avatar_backend"] = "musetalk_local_fallback"
            return path

        # 1. 显存护栏（硬拦截，绝不试跑）
        try:
            hw.ensure_vram(min_vram, pretty)
        except hw.VramGateError as e:
            return _degrade(str(e))

        # 2. 环境就绪检查（repo/venv/worker 存在即可，深检查交给 worker 自身）
        from ..core.config import resolve_path

        # 空配置必须先拦：resolve_path("") 会解析成项目根目录（存在），
        # 否则 exists() 护栏误过、subprocess 试图"执行一个目录"抛费解的 OSError
        for key in ("env_python", "worker_script", "repo_dir"):
            if not str(cfg.get(key, "")).strip():
                return _degrade(f"配置缺失 avatar.{self.provider}.{key}")

        env_python = resolve_path(cfg.get("env_python"))
        worker = resolve_path(cfg.get("worker_script"))
        repo = resolve_path(cfg.get("repo_dir"))
        if not env_python.exists() or not repo.exists():
            return _degrade(
                f"未安装（repo={repo.name} 或其 venv 缺失，安装指引见模型管理页）"
            )
        if not worker.exists():
            return _degrade(f"worker 脚本缺失: {worker}")

        face_path = self._get_avatar_reference(avatar_id)
        if not face_path:
            raise RuntimeError(
                f"数字人 {avatar_id} 无参考照片/视频，请先注册形象"
            )
        if (self.provider == "latentsync_local"
                and Path(face_path).suffix.lower() not in
                (".mp4", ".mov", ".avi", ".mkv", ".webm")):
            return _degrade("LatentSync 需要出镜视频参考（当前形象只有照片）")

        audio_path = ctx.audio_path
        if audio_path.suffix.lower() != ".wav":
            wav_path = ctx.work_dir / f"{self.provider}_input.wav"
            self.ffmpeg.convert_audio(audio_path, wav_path)
            audio_path = wav_path

        cmd = [
            str(env_python), str(worker),
            "--repo", str(repo),
            "--face", str(Path(face_path).resolve()),
            "--audio", str(Path(audio_path).resolve()),
            "--outfile", str(Path(output_path).resolve()),
        ]
        if self.provider == "latentsync_local":
            cmd += ["--inference_steps", str(cfg.get("inference_steps", 25))]
        else:
            cmd += ["--sample_steps", str(cfg.get("sample_steps", 5))]

        timeout = int(cfg.get("timeout", 3600))
        self.logger.info(f"运行 {pretty} 推理（beta，GPU 独立环境）...")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout, cwd=str(repo),
            )
        except subprocess.TimeoutExpired:
            return _degrade(f"推理超时（>{timeout}s）")
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "")[-500:]
            return _degrade(f"推理失败(code={result.returncode}): {tail[-300:]}")
        if not output_path.exists():
            return _degrade("推理完成但输出文件不存在")

        ctx.metadata["avatar_backend"] = self.provider
        return self._to_portrait_or_original(output_path, ctx)

    def _generate_wav2lip(
        self, ctx: JobContext, avatar_id: str, output_path: Path
    ) -> Path:
        """使用 Wav2Lip 生成唇形同步视频

        输入：真人照片或视频 + 音频
        输出：嘴唇会动的视频
        """
        self.logger.info(
            f"Wav2Lip 唇形同步 avatar={avatar_id} audio={ctx.audio_path.name} "
            f"duration={ctx.audio_duration:.1f}s"
        )

        # 获取参考人脸（照片或视频）
        face_path = self._get_avatar_reference(avatar_id)
        if not face_path:
            raise RuntimeError(
                f"数字人 {avatar_id} 无参考照片/视频，请先上传真人照片或视频注册形象"
            )

        self.logger.info(f"参考人脸: {face_path}")

        # 准备音频（Wav2Lip 需要 wav 格式）
        audio_path = ctx.audio_path
        if audio_path.suffix.lower() != ".wav":
            wav_path = ctx.work_dir / "wav2lip_input.wav"
            self.ffmpeg.convert_audio(audio_path, wav_path)
            audio_path = wav_path

        # 调用 Wav2Lip 推理（使用独立的 Python 3.8 环境，非主项目 3.12）
        # 解析路径为绝对路径（配置里的相对路径基于项目根目录）
        from ..core.config import resolve_path as _abs

        env_python = _abs(self.wav2lip_env_python)
        inference_script = _abs(self.wav2lip_inference_script)

        # 使用绝对路径（Wav2Lip 从自身目录运行，相对路径会失效）
        checkpoint_abs = _abs(self.wav2lip_checkpoint)
        face_abs = Path(face_path).resolve()
        audio_abs = Path(audio_path).resolve()
        output_abs = Path(output_path).resolve()

        # wav2lip_env 的 site-packages 在 PYTHONPATH，需保证用独立解释器
        cmd = [
            str(env_python), str(inference_script),
            "--checkpoint_path", str(checkpoint_abs),
            "--face", str(face_abs),
            "--audio", str(audio_abs),
            "--outfile", str(output_abs),
            "--pads", *[str(p) for p in self.wav2lip_config.get("pads", [0, 20, 0, 0])],
            "--face_det_batch_size", str(self.wav2lip_config.get("face_det_batch_size", 8)),
            "--wav2lip_batch_size", str(self.wav2lip_config.get("wav2lip_batch_size", 16)),
            "--resize_factor", str(self.wav2lip_config.get("resize_factor", 1)),
        ]
        if self.wav2lip_config.get("nosmooth", False):
            cmd.append("--nosmooth")

        self.logger.info(
            f"运行 Wav2Lip 推理 (CPU模式, env={env_python.parent.parent.name})，"
            f"音频 {ctx.audio_duration:.1f}s，预计耗时数分钟至数十分钟..."
        )
        # Wav2Lip 推理脚本内部加载依赖文件用相对路径，必须在 Wav2Lip 根目录运行
        wav2lip_root = inference_script.parent
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 120分钟超时（resize_factor=1 CPU 模式长视频可能超过 60 分钟）
            cwd=str(wav2lip_root),
        )

        if result.returncode != 0:
            self.logger.error(f"Wav2Lip 推理失败: {result.stderr[-500:]}")
            raise RuntimeError(f"Wav2Lip 推理失败: {result.stderr[-300:]}")

        if not output_path.exists():
            raise RuntimeError("Wav2Lip 推理完成但输出文件不存在")

        self.logger.info(
            f"Wav2Lip 唇形同步完成: {output_path.name} "
            f"({output_path.stat().st_size // 1024}KB)"
        )

        # Wav2Lip 输出尺寸 = face 输入尺寸（通常横屏，如 1920x1080）
        # 转成竖屏 1080x1920（人脸居中 + 模糊背景，主流口播做法）
        return self._to_portrait_or_original(output_path, ctx)

    def _get_avatar_reference(self, avatar_id: str) -> Path | None:
        """获取数字人参考素材

        Wav2Lip 视频驱动模式（保留原视频头部动作+表情，只换嘴型）：
          优先级：reference_video.mp4 > reference.mp4 > avatar.mp4
                  > reference.jpg > reference.png > avatar.jpg

        视频输入让 Wav2Lip 逐帧保留原视频的姿态/表情/手势，只重绘嘴部；
        照片输入会退化成"静态照片+嘴动"（头不动），仅作降级。
        """
        avatar_dir = self.avatars_dir / avatar_id
        if not avatar_dir.exists():
            return None

        # 优先查找参考视频（视频驱动 = 保留原动作）
        for name in ("reference_video.mp4", "reference.mp4", "avatar.mp4"):
            p = avatar_dir / name
            if p.exists():
                return p

        # 降级：参考照片（照片驱动 = 头不动，仅嘴动）
        for name in ("reference.jpg", "reference.png", "avatar.jpg", "avatar.png"):
            p = avatar_dir / name
            if p.exists():
                return p

        return None

    def _generate_cloud(
        self, ctx: JobContext, avatar_id: str, output_path: Path
    ) -> Path:
        """调用云端数字人 API（LatentSync / MuseTalk / EchoMimic）"""
        self.logger.info(
            f"云端数字人生成 provider={self.provider} "
            f"avatar={avatar_id} audio={ctx.audio_path}"
        )

        # 读取音频并 base64 编码
        audio_b64 = base64.b64encode(ctx.audio_path.read_bytes()).decode()

        payload = {
            "audio_base64": audio_b64,
            "avatar_id": avatar_id,
            "output_fps": self.output_fps,
            "output_resolution": list(self.output_resolution),
        }

        # LatentSync 专用参数（覆盖云端默认）
        if self.provider == "latentsync":
            ls_cfg = self.config.get("avatar.latentsync", {}) or {}
            payload["inference_steps"] = ls_cfg.get("inference_steps", 25)
            payload["resolution"] = ls_cfg.get("resolution", 512)
            payload["config_name"] = ls_cfg.get("config", "high_quality")

        resp = self.gpu.call_avatar(payload)

        # 记录实际使用的后端（云端返回）
        backend = resp.get("backend")
        if backend:
            ctx.metadata["avatar_backend"] = backend
            self.logger.info(f"云端实际后端: {backend}")

        video_b64 = resp.get("video_base64") or resp.get("data", {}).get("video_base64")
        if not video_b64:
            # 如果返回的是 URL，下载
            video_url = resp.get("video_url")
            if video_url:
                import httpx
                r = httpx.get(video_url, timeout=120)
                r.raise_for_status()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(r.content)
                return output_path
            raise RuntimeError(f"数字人 API 返回无视频数据: {resp}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(base64.b64decode(video_b64))
        self.logger.info(f"云端数字人生成完成 video={output_path}")
        return output_path

    def _generate_mock(
        self, ctx: JobContext, avatar_id: str, output_path: Path
    ) -> Path:
        """Mock 模式：生成占位头像图 + 音频合成视频"""
        self.logger.info(
            f"Mock 数字人生成 avatar={avatar_id} "
            f"audio={ctx.audio_path} duration={ctx.audio_duration:.2f}s"
        )

        # 生成或获取占位头像图
        avatar_image = self._get_avatar_image(avatar_id)

        # 用 ffmpeg 合成图片 + 音频 = 视频
        self.ffmpeg.image_audio_to_video(
            image=avatar_image,
            audio=ctx.audio_path,
            output=output_path,
            fps=self.output_fps,
            resolution=self.output_resolution,
            video_bitrate="4M",
        )
        self.logger.info(f"Mock 数字人视频生成完成: {output_path}")
        return output_path

    def _apply_micro_motion(self, video_path: Path, ctx: JobContext) -> Path:
        """应用微动作层（FFmpeg 后处理）

        给数字人口播视频叠加：呼吸缩放 + 微抖动 + 眨眼亮度节奏。
        纯 FFmpeg 实现，CPU 即可，缓解"只有嘴动"的恐怖谷。
        失败时降级返回原视频（不阻断流程）。
        """
        output_path = ctx.work_dir / "avatar_with_motion.mp4"
        w, h = self.output_resolution
        try:
            result_path = self.ffmpeg.add_micro_motion(
                video=video_path,
                output=output_path,
                width=w,
                height=h,
                fps=self.output_fps,
                breathing_scale=float(
                    self.micro_motion_cfg.get("breathing_scale", 0.02)
                ),
                breathing_period=float(
                    self.micro_motion_cfg.get("breathing_period", 4.0)
                ),
                shake_amplitude=float(
                    self.micro_motion_cfg.get("shake_amplitude", 0.3)
                ),
                shake_period=float(
                    self.micro_motion_cfg.get("shake_period", 2.0)
                ),
                blink_enabled=bool(
                    self.micro_motion_cfg.get("blink_enabled", True)
                ),
                blink_interval=float(
                    self.micro_motion_cfg.get("blink_interval", 4.0)
                ),
            )
            self.logger.info(f"微动作层应用完成: {result_path.name}")
            return result_path
        except Exception as e:
            self.logger.warning(
                f"微动作层应用失败，降级使用原视频: {e}"
            )
            return video_path

    def _get_avatar_image(self, avatar_id: str) -> Path:
        """获取数字人头像图片

        优先使用已注册的参考图，否则生成占位图。
        """
        # 查找已注册形象
        avatar_dir = self.avatars_dir / avatar_id
        if avatar_dir.exists():
            for name in ("reference.jpg", "reference.png", "avatar.jpg"):
                p = avatar_dir / name
                if p.exists():
                    return p

        # 生成占位图
        placeholder = self.avatars_dir / avatar_id / "placeholder.jpg"
        placeholder.parent.mkdir(parents=True, exist_ok=True)
        self._generate_placeholder_image(placeholder, avatar_id)
        return placeholder

    def _generate_placeholder_image(
        self, output: Path, avatar_id: str
    ) -> None:
        """生成占位头像图（纯色背景 + 文字标识）"""
        w, h = self.output_resolution
        # 浅灰背景
        img = Image.new("RGB", (w, h), color=(60, 70, 90))
        draw = ImageDraw.Draw(img)

        # 尝试加载字体，失败用默认
        font_path = self.config.get("cover.font_path", "")
        try:
            font_large = ImageFont.truetype(font_path, 80) if font_path else ImageFont.load_default()
            font_small = ImageFont.truetype(font_path, 40) if font_path else ImageFont.load_default()
        except Exception:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # 中心圆形头像占位
        cx, cy = w // 2, h // 2 - 100
        r = 200
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            fill=(120, 140, 180), outline=(200, 210, 230), width=4,
        )

        # 文字
        title = "数字人口播"
        subtitle = f"Avatar: {avatar_id}"
        for text, font, y_offset in [
            (title, font_large, 80),
            (subtitle, font_small, 180),
        ]:
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            except Exception:
                tw, th = 400, 80
            draw.text(
                ((w - tw) // 2, cy + y_offset),
                text, fill=(255, 255, 255), font=font,
            )

        # 底部标识
        footer = "FreeVideoAI · Mock Mode"
        try:
            bbox = draw.textbbox((0, 0), footer, font=font_small)
            tw = bbox[2] - bbox[0]
        except Exception:
            tw = 300
        draw.text(((w - tw) // 2, h - 120), footer, fill=(180, 190, 210), font=font_small)

        img.save(str(output), "JPEG", quality=90)

    def register_avatar(
        self, avatar_id: str, reference_video: Path
    ) -> bool:
        """注册数字人形象

        Args:
            avatar_id: 形象 ID
            reference_video: 参考视频或照片（3-10s 正面说话，嘴巴不动）
                - wav2lip 模式：直接保存为参考素材，用于唇形同步
                - mock 模式：从视频抽一帧作为占位图
                - 云端模式：上传到云端服务
        """
        avatar_dir = self.avatars_dir / avatar_id
        avatar_dir.mkdir(parents=True, exist_ok=True)
        reference_video = Path(reference_video)

        if self.provider in ("musetalk_local", "wav2lip", "omnihuman_api",
                             "latentsync_local", "echomimic_local"):
            # 本地保存参考素材（照片或视频）。omnihuman_api 虽是云端生成，
            # 但参考照片同样存本地 avatar_dir（生成时经中转站上传）
            try:
                ext = reference_video.suffix.lower()
                # 清理旧参考素材
                for old in avatar_dir.glob("reference*"):
                    old.unlink(missing_ok=True)
                # 根据类型保存
                if ext in (".jpg", ".jpeg", ".png", ".webp"):
                    ref_path = avatar_dir / "reference.jpg"
                    if ext != ".jpg":
                        # 转换为 jpg
                        from PIL import Image as _Image
                        img = _Image.open(reference_video).convert("RGB")
                        img.save(str(ref_path), "JPEG", quality=95)
                    else:
                        shutil.copy2(reference_video, ref_path)
                    kind = "photo"
                elif ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
                    ref_path = avatar_dir / "reference_video.mp4"
                    shutil.copy2(reference_video, ref_path)
                    kind = "video"
                else:
                    raise RuntimeError(f"不支持的参考素材格式: {ext}")

                # 生成预览图（视频抽一帧，照片直接缩略）
                preview = avatar_dir / "reference.jpg" if kind == "photo" else avatar_dir / "preview.jpg"
                if kind == "video":
                    subprocess.run(
                        [
                            self.ffmpeg.ffmpeg, "-y",
                            "-i", str(ref_path),
                            "-frames:v", "1",
                            "-q:v", "2",
                            str(preview),
                        ],
                        capture_output=True, check=True,
                    )

                # 保存元数据
                import json
                (avatar_dir / "meta.json").write_text(
                    json.dumps({
                        "avatar_id": avatar_id,
                        "source": str(reference_video),
                        "mode": self.provider,
                        "reference_type": kind,
                        "reference_path": str(ref_path),
                        "has_lip_sync": True,
                    }, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                self.logger.info(
                    f"本地形象注册成功({self.provider}): {avatar_id} -> {ref_path.name} ({kind})"
                )
                return True
            except Exception as e:
                self.logger.error(f"本地形象注册失败({self.provider}): {e}")
                return False
        elif self.provider == "mock":
            # Mock 模式：从视频抽一帧作为参考图
            try:
                ref_img = avatar_dir / "reference.jpg"
                subprocess.run(
                    [
                        self.ffmpeg.ffmpeg, "-y",
                        "-i", str(reference_video),
                        "-frames:v", "1",
                        "-q:v", "2",
                        str(ref_img),
                    ],
                    capture_output=True, check=True,
                )
                self.logger.info(f"Mock 形象注册成功: {avatar_id} -> {ref_img}")
                # 保存元数据
                import json
                (avatar_dir / "meta.json").write_text(
                    json.dumps({
                        "avatar_id": avatar_id,
                        "source": str(reference_video),
                        "mode": "mock",
                    }, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return True
            except Exception as e:
                self.logger.error(f"形象注册失败: {e}")
                return False
        else:
            # 云端模式：上传参考视频
            try:
                video_b64 = base64.b64encode(Path(reference_video).read_bytes()).decode()
                resp = self.gpu.call_avatar_register({
                    "avatar_id": avatar_id,
                    "reference_video_base64": video_b64,
                })
                return resp.get("success", False)
            except Exception as e:
                self.logger.error(f"云端形象注册失败: {e}")
                return False
