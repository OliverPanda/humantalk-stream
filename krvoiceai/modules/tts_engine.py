"""TTS 声音克隆模块

六种 provider：
- voxcpm:     本地 VoxCPM-0.5B（GPU 声音克隆，独立 venv 子进程运行，默认档）
- moss_nano:  本地 MOSS-TTS-Nano ONNX（CPU 声音克隆，0.1B 模型，5s 样本零克隆）
- mimo:       调用小米 MiMo TTS API（OpenAI 兼容 chat/completions 端点）
- gpt_sovits: 调用云端 GPT-SoVITS API（声音克隆）
- edge_tts:   使用 edge-tts 标准音色（无克隆，CPU 可跑）
- mock:       生成静音 wav（保证流程可跑通）

降级链：合成失败时按 tts.fallback_chain（默认 voxcpm→moss_nano→edge_tts）逐级降级。
结果缓存：hash(文本+音色+关键参数) → workspace_data/cache/tts/<hash>.wav，命中直接复用。

输出：wav/mp3 音频文件 + 时长 + 分句时间戳
"""
from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Any

import httpx

from ..core.audio_utils import (
    estimate_speech_duration,
    generate_silent_wav,
    get_wav_duration,
    split_text_to_segments,
)
from ..core.base_module import BaseModule, JobContext, ModuleResult
from ..core.gpu_runner import GPURunner


class TTSEngine(BaseModule):
    """TTS 声音克隆/合成模块"""

    name = "tts"
    requires_gpu = True  # 真实模式需要 GPU（moss_nano/edge_tts/mock 可纯 CPU 运行）

    # 无需云端 GPU 服务即可本地运行的 provider
    # （voxcpm 用本机 GPU + 独立 venv 子进程，不依赖云端服务）
    CPU_ONLY_PROVIDERS = {"voxcpm", "moss_nano", "edge_tts", "mock"}

    def __init__(self, config=None, gpu_runner: GPURunner | None = None):
        super().__init__(config)
        self.provider = self.config.get("tts.provider", "mock")
        self.api_base = self.config.get("tts.api_base", "")
        self.api_key = self.config.get("tts.api_key", "")
        self.edge_voice = self.config.get("tts.edge_voice", "zh-CN-XiaoxiaoNeural")
        self.voices_dir = Path(self.config.get("tts.voices_dir", "./config/voices"))
        self.default_voice = self.config.get("tts.default_voice", "default")
        self.timeout = self.config.get("tts.timeout", 120)
        self.gpu = gpu_runner or GPURunner()
        # MOSS-TTS-Nano 运行时（懒加载，首次 moss_nano 合成时初始化）
        self._moss_runtime = None
        # 最近一次实际完成合成的 provider（降级链可能与 self.provider 不同）
        self._last_provider = self.provider

    def setup(self) -> None:
        # 判断真实可用性
        if self.provider == "gpt_sovits":
            available = self.gpu.health_check_tts()
            if not available:
                self.logger.warning(
                    "GPT-SoVITS 服务不可用，降级到 mock 模式"
                )
                self.provider = "mock"
        self.logger.info(f"TTS 模块初始化 provider={self.provider}")
        super().setup()

    def run(self, ctx: JobContext) -> ModuleResult:
        """根据 ctx.script_text 合成音频"""
        text = ctx.script_text or ctx.input_script
        if not text:
            return ModuleResult(success=False, error="无文案可合成")

        voice_id = ctx.voice_id or self.default_voice
        output_path = ctx.work_dir / "tts_output.wav"

        # 从 audio 配置段读取语速/音量/音高/情感（UI 持久化到此）
        audio_cfg = self.config.get("audio", {}) or {}
        speed = audio_cfg.get("speed")
        volume = audio_cfg.get("volume")
        pitch = audio_cfg.get("pitch")
        emotion = audio_cfg.get("emotion")
        # 类型转换与边界保护
        try:
            speed = float(speed) if speed is not None else None
        except (TypeError, ValueError):
            speed = None
        try:
            volume = int(volume) if volume is not None else None
        except (TypeError, ValueError):
            volume = None
        try:
            pitch = int(pitch) if pitch is not None else None
        except (TypeError, ValueError):
            pitch = None

        try:
            start = time.time()
            audio_path, duration, timestamps = self._synth_with_fallback(
                text, voice_id, output_path,
                speed=speed, volume=volume, pitch=pitch, emotion=emotion,
            )

            ctx.audio_path = audio_path
            ctx.audio_duration = duration
            ctx.metadata["tts_timestamps"] = timestamps
            ctx.metadata["tts_provider"] = self._last_provider
            ctx.metadata["tts_audio_opts"] = {
                "speed": speed, "volume": volume, "pitch": pitch, "emotion": emotion,
            }

            return ModuleResult(
                success=True,
                data={
                    "audio_path": str(audio_path),
                    "duration": duration,
                    "voice_id": voice_id,
                    "provider": self._last_provider,
                    "segments": len(timestamps),
                    "speed": speed,
                    "emotion": emotion,
                },
            )
        except Exception as e:
            return ModuleResult(success=False, error=str(e))

    def synthesize(
        self, text: str, voice_id: str, output_path: Path,
        speed: float | None = None, volume: int | None = None,
        pitch: int | None = None, emotion: str | None = None,
    ) -> tuple[Path, float, list[dict]]:
        """公共合成方法（provider 无关，供 UI 试听/预览使用，无需构造 JobContext）

        与 run() 的分发逻辑一致，但直接返回 (音频路径, 时长, 时间戳)，
        不依赖 ctx，也不走 run_single_module 的笨重前置步骤。

        Args:
            text: 要合成的文案
            voice_id: 音色 ID（default 或已注册音色）
            output_path: 输出 wav 路径
            speed: 语速倍率（0.5-2.0，1.0 为正常），None 时用引擎默认
            volume: 音量百分比（0-200，100 为正常），None 时用引擎默认
            pitch: 音高半音偏移（-12 到 +12，0 为正常），None 时用引擎默认
            emotion: 情感标签（neutral/calm/excited/gentle/serious/cheerful），
                     目前仅记录到 metadata，由支持情感的 provider 使用

        Returns:
            (audio_path: Path, duration: float, timestamps: list[dict])
        """
        if not text or not text.strip():
            raise ValueError("无文案可合成")
        audio_opts = {"speed": speed, "volume": volume, "pitch": pitch, "emotion": emotion}
        return self._synth_with_fallback(text, voice_id, output_path, **audio_opts)

    # ------------------------------------------------------------------
    # provider 分发 / 降级链 / 结果缓存
    # ------------------------------------------------------------------

    def _dispatch_provider(
        self, provider: str, text: str, voice_id: str, output_path: Path,
        **audio_opts,
    ) -> tuple[Path, float, list[dict]]:
        """按 provider 名称分发到具体合成实现"""
        if provider == "voxcpm":
            return self._synth_voxcpm(text, voice_id, output_path, **audio_opts)
        if provider == "moss_nano":
            return self._synth_moss_nano(text, voice_id, output_path, **audio_opts)
        if provider == "mimo":
            return self._synth_mimo(text, voice_id, output_path, **audio_opts)
        if provider == "gpt_sovits":
            return self._synth_gpt_sovits(text, voice_id, output_path, **audio_opts)
        if provider == "edge_tts":
            return self._synth_edge(text, voice_id, output_path, **audio_opts)
        if provider == "mock":
            return self._synth_mock(text, voice_id, output_path, **audio_opts)
        raise ValueError(f"未知 TTS provider: {provider}")

    def _synth_with_fallback(
        self, text: str, voice_id: str, output_path: Path, **audio_opts,
    ) -> tuple[Path, float, list[dict]]:
        """按降级链依次尝试 provider，全部失败才抛出

        仅当当前 provider 在 tts.fallback_chain（默认 voxcpm → moss_nano →
        edge_tts）中时才逐级降级，且只向后降（voxcpm 失败走 moss_nano，
        moss_nano 失败走 edge_tts）。链外 provider（gpt_sovits/mimo/mock）
        保持原有 fail-fast 行为。
        每个 provider 尝试前先查结果缓存（hash(文本+音色+关键参数)）。
        """
        fallback = list(self.config.get("tts.fallback_chain") or [])
        if self.provider in fallback:
            chain = fallback[fallback.index(self.provider):]
        else:
            chain = [self.provider]

        last_err: Exception | None = None
        for i, provider in enumerate(chain):
            try:
                cached = self._cache_load(provider, text, voice_id, audio_opts, output_path)
                if cached is not None:
                    self._last_provider = provider
                    return cached
                result = self._dispatch_provider(
                    provider, text, voice_id, output_path, **audio_opts
                )
                self._cache_store(provider, text, voice_id, audio_opts, result)
                self._last_provider = provider
                if i > 0:
                    self.logger.warning(
                        f"TTS 已降级至 provider={provider}"
                        f"（原 provider={chain[0]} 失败）"
                    )
                return result
            except Exception as e:
                last_err = e
                nxt = chain[i + 1] if i + 1 < len(chain) else None
                if nxt:
                    self.logger.warning(
                        f"TTS provider={provider} 合成失败: {e}；降级到 {nxt}"
                    )
                else:
                    self.logger.error(
                        f"TTS 全部 provider 失败，最后错误 provider={provider}: {e}"
                    )
        raise last_err if last_err else RuntimeError("TTS 合成失败（无可用 provider）")

    def _cache_dir(self) -> Path | None:
        """结果缓存目录；缓存关闭时返回 None"""
        cache_cfg = self.config.get("tts.cache", {}) or {}
        if not cache_cfg.get("enabled", True):
            return None
        from ..core.config import PROJECT_ROOT
        raw = cache_cfg.get("dir", "./workspace_data/cache/tts")
        p = Path(raw)
        return p if p.is_absolute() else (Path(PROJECT_ROOT) / raw).resolve()

    def _cache_key(
        self, provider: str, text: str, voice_id: str, audio_opts: dict,
    ) -> str:
        """hash(文本 + 音色 + provider + 关键合成参数) → 缓存键"""
        import hashlib
        payload: dict = {
            "provider": provider,
            "text": text,
            "voice_id": voice_id,
            "opts": {k: audio_opts.get(k) for k in ("speed", "volume", "pitch", "emotion")},
        }
        if provider == "voxcpm":
            vc = self.config.get("tts.voxcpm", {}) or {}
            payload["voxcpm"] = {
                "cfg_value": vc.get("cfg_value", 2.0),
                "inference_timesteps": vc.get("inference_timesteps", 10),
                "seed": vc.get("seed", 42),
                "model_dir": str(vc.get("model_dir", "")),
            }
        elif provider == "moss_nano":
            mc = self.config.get("tts.moss_nano", {}) or {}
            payload["moss"] = {"builtin_voice": mc.get("builtin_voice", "Junhao")}
        elif provider == "edge_tts":
            payload["edge_voice"] = voice_id if voice_id.endswith("Neural") else self.edge_voice
            payload["edge_pcm_revision"] = 2  # 旧缓存可能是缺依赖时生成的占位静音
        # 自定义音色把样本文件签名(mtime+size)纳入键：
        # 重新注册同名音色后旧缓存自动失效，否则新样本永远不生效
        sample = self._find_voice_sample(voice_id)
        if sample is not None:
            try:
                st = sample.stat()
                payload["sample_sig"] = f"{st.st_mtime_ns}:{st.st_size}"
            except OSError:
                pass
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def _cache_load(
        self, provider: str, text: str, voice_id: str, audio_opts: dict,
        output_path: Path,
    ) -> tuple[Path, float, list[dict]] | None:
        """查缓存；命中则复制到 output_path 并返回结果，未命中返回 None"""
        cache_dir = self._cache_dir()
        if cache_dir is None or provider == "mock":
            return None
        key = self._cache_key(provider, text, voice_id, audio_opts)
        wav = cache_dir / f"{key}.wav"
        if not wav.exists() or wav.stat().st_size < 1024:
            return None
        import shutil
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wav, output_path)
        duration = None
        timestamps = None
        meta_path = cache_dir / f"{key}.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                duration = meta.get("duration")
                timestamps = meta.get("timestamps")
            except Exception:
                pass
        if duration is None:
            duration = get_wav_duration(output_path)
        if not timestamps:
            timestamps = self._estimate_char_timestamps(text, duration)
        self.logger.info(
            f"TTS 缓存命中 (cache hit) provider={provider} key={key} "
            f"duration={duration:.2f}s -> {output_path.name}"
        )
        return output_path, duration, timestamps

    def _cache_store(
        self, provider: str, text: str, voice_id: str, audio_opts: dict,
        result: tuple[Path, float, list[dict]],
    ) -> None:
        """把合成结果写入缓存（仅缓存 wav；写失败只警告不影响主流程）"""
        cache_dir = self._cache_dir()
        if cache_dir is None or provider == "mock":
            return
        audio_path, duration, timestamps = result
        if Path(audio_path).suffix.lower() != ".wav":
            return  # mimo 可能输出 mp3，跳过
        try:
            import shutil
            cache_dir.mkdir(parents=True, exist_ok=True)
            key = self._cache_key(provider, text, voice_id, audio_opts)
            shutil.copy2(audio_path, cache_dir / f"{key}.wav")
            (cache_dir / f"{key}.json").write_text(
                json.dumps({
                    "provider": provider,
                    "voice_id": voice_id,
                    "text_len": len(text),
                    "duration": duration,
                    "timestamps": timestamps,
                    "created_at": time.time(),
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            self.logger.debug(f"TTS 结果已写入缓存 key={key}")
        except Exception as e:
            self.logger.warning(f"TTS 缓存写入失败（忽略）: {e}")

    def _estimate_char_timestamps(self, text: str, duration: float) -> list[dict]:
        """按分句字数比例估算时间戳（与 moss/mock 相同策略，后续 ASR 会校正）"""
        segments = split_text_to_segments(text)
        timestamps: list[dict] = []
        offset = 0.0
        total_chars = sum(len(s) for s in segments) or 1
        for seg in segments:
            seg_dur = duration * len(seg) / total_chars
            timestamps.append({
                "text": seg,
                "start": round(offset, 3),
                "end": round(offset + seg_dur, 3),
            })
            offset += seg_dur
        return timestamps

    # ------------------------------------------------------------------
    # VoxCPM-0.5B（GPU 声音克隆，独立 venv 子进程）
    # ------------------------------------------------------------------

    def _synth_voxcpm(
        self, text: str, voice_id: str, output_path: Path,
        speed: float | None = None, volume: int | None = None,
        pitch: int | None = None, emotion: str | None = None,
    ) -> tuple[Path, float, list[dict]]:
        """使用本地 VoxCPM-0.5B 合成（GPU 声音克隆）

        与 Wav2Lip 同范式：通过 subprocess 调用 VoxCPM 专属 venv 的 python
        运行 scripts/voxcpm_worker.py，模型每次调用加载一次，进程退出自动释放显存。
        speed/volume/pitch/emotion 为 VoxCPM 原生不支持的参数，仅记录不生效。
        """
        import subprocess
        from ..core.config import resolve_path as _abs

        cfg = self.config.get("tts.voxcpm", {}) or {}

        venv_python = _abs(cfg.get("venv_python", "../VoxCPM/.venv/Scripts/python.exe"))
        model_dir = _abs(cfg.get("model_dir", "../VoxCPM/pretrained_models/VoxCPM-0.5B"))
        worker = _abs(cfg.get("worker_script", "./scripts/voxcpm_worker.py"))
        if not venv_python.exists():
            raise RuntimeError(f"VoxCPM venv python 不存在: {venv_python}")
        if not worker.exists():
            raise RuntimeError(f"voxcpm_worker.py 不存在: {worker}")

        # 显存预检：不足时立刻失败 → 触发降级链（避免 OOM 卡死流水线）
        min_free = int(cfg.get("min_free_vram_mb", 2500))
        if min_free > 0:
            free_mb = self._query_free_vram_mb()
            if free_mb is not None and free_mb < min_free:
                raise RuntimeError(
                    f"GPU 空闲显存不足: {free_mb}MB < {min_free}MB（可能被其它进程占用）"
                )

        # 克隆参考：音色样本 + ref_text（meta.json，缺失时惰性转写补齐）
        prompt_wav = self._find_voice_sample(voice_id)
        prompt_text = ""
        if prompt_wav:
            prompt_text = self._ensure_ref_text(voice_id, prompt_wav)
            if not prompt_text:
                self.logger.warning(
                    f"音色 {voice_id} 无 ref_text 且转写失败，VoxCPM 退化为非克隆合成"
                )
                prompt_wav = None
        else:
            self.logger.info(
                f"VoxCPM 未找到音色样本 voice={voice_id}，使用模型默认音色"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path = output_path.parent / f"{output_path.stem}_voxcpm_raw.wav"
        task = {
            "text": text,
            "out_wav": str(raw_path.resolve()),
            "prompt_wav": str(prompt_wav) if prompt_wav else None,
            "prompt_text": prompt_text or None,
            "model_dir": str(model_dir),
            "cfg_value": float(cfg.get("cfg_value", 2.0)),
            "inference_timesteps": int(cfg.get("inference_timesteps", 10)),
            "seed": int(cfg.get("seed", 42)),
            "max_seg_chars": int(cfg.get("max_seg_chars", 120)),
        }
        task_path = output_path.parent / "voxcpm_task.json"
        task_path.write_text(
            json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        self.logger.info(
            f"VoxCPM 子进程合成 voice={voice_id} text_len={len(text)} "
            f"clone={'是' if prompt_wav else '否'} "
            f"cfg={task['cfg_value']} steps={task['inference_timesteps']}"
        )
        proc = subprocess.run(
            [str(venv_python), str(worker), str(task_path.resolve())],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=int(cfg.get("timeout", 600)),
            cwd=str(worker.parent),
        )

        result = None
        for line in reversed((proc.stdout or "").splitlines()):
            if line.startswith("VOXCPM_RESULT:"):
                try:
                    result = json.loads(line[len("VOXCPM_RESULT:"):])
                except json.JSONDecodeError:
                    pass
                break
        if proc.returncode != 0 or not result or not result.get("success"):
            err = (result or {}).get("error") or (proc.stderr or "")[-300:] \
                or f"exit={proc.returncode}"
            raise RuntimeError(f"VoxCPM worker 失败: {err}")
        if not raw_path.exists():
            raise RuntimeError("VoxCPM worker 报告成功但输出文件不存在")

        self.logger.info(
            f"VoxCPM 合成完成 duration={result.get('duration')}s "
            f"load={result.get('load_time')}s synth={result.get('synth_time')}s "
            f"RTF={result.get('rtf')}"
        )

        # 与 MOSS 相同的输出后处理：转 16kHz 单声道供 Wav2Lip 使用
        final_path = self._convert_to_16k_mono(raw_path, output_path)
        duration = get_wav_duration(final_path)
        timestamps = self._estimate_char_timestamps(text, duration)
        return final_path, duration, timestamps

    def _query_free_vram_mb(self) -> int | None:
        """查询 GPU 空闲显存（MB）；nvidia-smi 不可用时返回 None（跳过检查）"""
        import subprocess
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode != 0:
                return None
            return int(out.stdout.strip().splitlines()[0])
        except Exception:
            return None

    def _convert_to_16k_mono(self, src: Path, dst: Path) -> Path:
        """ffmpeg 转 16kHz 单声道 pcm_s16le（与 MOSS 输出后处理一致）

        失败时降级：直接把原始输出挪到 dst。
        """
        try:
            import subprocess
            tmp = dst.parent / f"{dst.stem}_16k_tmp.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(src),
                 "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(tmp)],
                capture_output=True, timeout=60,
            )
            if tmp.exists() and tmp.stat().st_size > 0:
                src.unlink(missing_ok=True)
                tmp.replace(dst)
                return dst
        except Exception as e:
            self.logger.warning(f"音频转 16k 失败，使用原始输出: {e}")
        if src != dst and src.exists():
            src.replace(dst)
        return dst

    def _find_voice_sample(self, voice_id: str) -> Path | None:
        """查找音色目录下的参考样本音频（与 MOSS 克隆样本查找规则一致）"""
        if not voice_id or voice_id == "default":
            return None
        from ..core.config import PROJECT_ROOT
        voices_dir = self.voices_dir if self.voices_dir.is_absolute() \
            else (Path(PROJECT_ROOT) / self.voices_dir)
        voice_dir = voices_dir / voice_id
        if not voice_dir.exists():
            return None
        for ext in (".wav", ".mp3", ".flac", ".m4a"):
            candidates = list(voice_dir.glob(f"sample*{ext}")) + list(
                voice_dir.glob(f"*{ext}")
            )
            if candidates:
                return candidates[0].resolve()
        return None

    def _ensure_ref_text(self, voice_id: str, sample_path: Path) -> str:
        """读取音色 meta.json 的 ref_text；老音色缺失时惰性转写补齐

        ref_text 用于 VoxCPM 零样本克隆的 prompt_text（与 prompt_wav 成对提供）。
        """
        meta_path = sample_path.parent / "meta.json"
        meta: dict = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as e:
                self.logger.warning(f"音色 meta.json 解析失败({e})，将重新转写")
                meta = {}
        ref_text = (meta.get("ref_text") or "").strip()
        if ref_text:
            return ref_text

        # 惰性补转写（老音色注册时代码尚无 meta.json）
        ref_text = self._transcribe_sample(sample_path)
        if ref_text:
            meta.update({
                "voice_id": voice_id,
                "sample": sample_path.name,
                "ref_text": ref_text,
                "ref_text_source": "faster-whisper(lazy)",
                "updated_at": time.time(),
            })
            try:
                meta_path.write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                self.logger.info(
                    f"音色 {voice_id} 惰性补转写完成，ref_text 已写入 {meta_path}"
                )
            except Exception as e:
                self.logger.warning(f"meta.json 写入失败: {e}")
        return ref_text

    def _transcribe_sample(self, sample_path: Path) -> str:
        """用主 venv 的 faster-whisper 转写音色样本（进程内单例，见 core/asr_utils）"""
        try:
            from ..core.asr_utils import transcribe_ref_text
            whisper_cfg = self.config.get("asr.whisper", {}) or {}
            language = self.config.get("asr.language", "zh")
            text = transcribe_ref_text(str(sample_path), whisper_cfg, language)
            self.logger.info(f"样本转写完成 len={len(text)}: {text[:50]}...")
            return text
        except Exception as e:
            self.logger.warning(f"样本转写失败（不影响注册/合成降级）: {e}")
            return ""

    def _get_moss_runtime(self):
        """懒加载 MOSS-TTS-Nano ONNX 运行时（仅依赖 onnxruntime + soundfile + sentencepiece）"""
        if self._moss_runtime is not None:
            return self._moss_runtime

        import sys
        from ..core.config import PROJECT_ROOT

        cfg = self.config.get("tts.moss_nano", {}) or {}

        # 路径解析策略：优先配置的绝对路径，其次相对 PROJECT_ROOT 解析，最后回退多个常见位置
        candidates = []
        raw_repo = cfg.get("repo_dir", "../../MOSS-TTS-Nano")
        if Path(raw_repo).is_absolute():
            candidates.append(Path(raw_repo))
        else:
            # 相对 PROJECT_ROOT（FreeVideoAI 目录）
            candidates.append((PROJECT_ROOT / raw_repo).resolve())
            candidates.append((PROJECT_ROOT / "../MOSS-TTS-Nano").resolve())
            candidates.append(Path(raw_repo).resolve())

        repo_dir = next((c for c in candidates if c.exists()), None)
        if repo_dir is None:
            raise RuntimeError(
                f"MOSS-TTS-Nano 仓库不存在，已尝试: {[str(c) for c in candidates]}。"
                f"请在设置中配置正确路径（tts.moss_nano.repo_dir）或克隆仓库"
            )

        # 把仓库根加入 sys.path 以便 import onnx_tts_runtime
        repo_str = str(repo_dir)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        from onnx_tts_runtime import OnnxTtsRuntime  # type: ignore

        # model_dir 解析：优先绝对路径 > 相对 repo_dir > 相对 PROJECT_ROOT > repo_dir/models
        raw_model = cfg.get("model_dir")
        model_candidates = []
        if raw_model:
            if Path(raw_model).is_absolute():
                model_candidates.append(Path(raw_model))
            else:
                model_candidates.append((repo_dir / raw_model).resolve())
                model_candidates.append((PROJECT_ROOT / raw_model).resolve())
                # raw_model 可能就是相对 repo 的（如 ../../MOSS-TTS-Nano/models），尝试 ../前缀剥离
                if raw_model.startswith("../"):
                    model_candidates.append((repo_dir.parent / raw_model[3:]).resolve())
        model_candidates.append((repo_dir / "models").resolve())
        model_dir_path = next((c for c in model_candidates if c.exists()), model_candidates[-1])
        model_dir = str(model_dir_path.resolve())
        self._moss_runtime = OnnxTtsRuntime(
            model_dir=model_dir,
            thread_count=int(cfg.get("cpu_threads", 4)),
            execution_provider=cfg.get("execution_provider", "cpu"),
        )
        self.logger.info(
            f"MOSS-TTS-Nano 运行时已加载 repo={repo_dir} model_dir={model_dir}"
        )
        return self._moss_runtime

    def _synth_moss_nano(
        self, text: str, voice_id: str, output_path: Path,
        speed: float | None = None, volume: int | None = None,
        pitch: int | None = None, emotion: str | None = None,
    ) -> tuple[Path, float, list[dict]]:
        """使用本地 MOSS-TTS-Nano ONNX 合成（支持声音克隆）

        若 voice_id 对应音色目录下有 sample 音频，则用该音频做零样本声音克隆；
        否则使用内置音色（如 Junhao）。
        """
        runtime = self._get_moss_runtime()
        cfg = self.config.get("tts.moss_nano", {}) or {}

        # 查找该音色的参考音频（用于声音克隆）
        prompt_audio_path = None
        builtin_voice = cfg.get("builtin_voice", "Junhao")
        if voice_id and voice_id != "default":
            voice_dir = self.voices_dir / voice_id
            if voice_dir.exists():
                # 找到任意 sample 音频
                for ext in (".wav", ".mp3", ".flac", ".m4a"):
                    candidates = list(voice_dir.glob(f"sample*{ext}")) + list(
                        voice_dir.glob(f"*{ext}")
                    )
                    if candidates:
                        prompt_audio_path = str(candidates[0].resolve())
                        self.logger.info(
                            f"MOSS 声音克隆 voice={voice_id} sample={prompt_audio_path}"
                        )
                        break

        if prompt_audio_path is None:
            self.logger.info(
                f"MOSS 使用内置音色 voice={builtin_voice}（未找到克隆样本）"
            )

        self.logger.info(
            f"MOSS-TTS-Nano 合成 voice={voice_id} text_len={len(text)}"
        )

        result = runtime.synthesize(
            text=text,
            voice=builtin_voice,
            prompt_audio_path=prompt_audio_path,
            output_audio_path=str(output_path.resolve()),
            streaming=bool(cfg.get("realtime_streaming", True)),
            max_new_frames=int(cfg.get("max_new_frames", 375)),
            voice_clone_max_text_tokens=int(cfg.get("voice_clone_max_text_tokens", 75)),
            enable_wetext=bool(cfg.get("enable_wetext", False)),
            enable_normalize_tts_text=bool(cfg.get("enable_normalize_tts_text", True)),
        )

        audio_path = Path(result["audio_path"])
        # MOSS 输出 48kHz 立体声 wav；转 16kHz 单声道供 Wav2Lip 使用
        final_path = audio_path
        try:
            import subprocess
            mono_path = output_path.parent / f"{output_path.stem}_16k.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(audio_path),
                 "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                 str(mono_path)],
                capture_output=True, timeout=30,
            )
            if mono_path.exists():
                # 替换为 16k 单声道版本
                audio_path.unlink(missing_ok=True)
                mono_path.rename(output_path)
                final_path = output_path
        except Exception as e:
            self.logger.warning(f"MOSS 音频转 16k 失败，使用原始输出: {e}")
            if audio_path != output_path and audio_path.exists():
                audio_path.rename(output_path)
                final_path = output_path

        duration = get_wav_duration(final_path)

        # MOSS 不返回逐句时间戳，按分句估算（用于字幕对齐，后续 ASR 会校正）
        segments = split_text_to_segments(text)
        timestamps: list[dict] = []
        offset = 0.0
        total_chars = sum(len(s) for s in segments) or 1
        for seg in segments:
            seg_dur = duration * len(seg) / total_chars
            timestamps.append({
                "text": seg,
                "start": round(offset, 3),
                "end": round(offset + seg_dur, 3),
            })
            offset += seg_dur

        self.logger.info(
            f"MOSS-TTS-Nano 合成完成 duration={duration:.2f}s segments={len(segments)}"
        )
        return final_path, duration, timestamps

    def _synth_mimo(
        self, text: str, voice_id: str, output_path: Path,
        speed: float | None = None, volume: int | None = None,
        pitch: int | None = None, emotion: str | None = None,
    ) -> tuple[Path, float, list[dict]]:
        """调用小米 MiMo TTS API（OpenAI 兼容 chat/completions 端点）

        MiMo TTS 特点：
        - 端点：{api_base}/chat/completions
        - 文本放在 assistant 角色消息中
        - 音色和格式放在 audio 对象中
        - 返回 base64 编码音频在 choices[0].message.audio.data
        """
        self.logger.info(f"MiMo TTS 合成 voice={voice_id} text_len={len(text)}")

        # MiMo 单次合成有长度限制，分句合成
        segments = split_text_to_segments(text, max_chars=300)
        timestamps: list[dict] = []
        combined_audio = bytearray()
        offset = 0.0

        # 音色映射：voice_id -> mimo voice
        mimo_voice = voice_id if voice_id != "default" else "mimo_default"

        for seg in segments:
            payload = {
                "model": self.config.get("tts.mimo_model", "mimo-v2.5-tts"),
                "messages": [
                    {"role": "assistant", "content": seg}
                ],
                "audio": {
                    "format": "mp3",
                    "voice": mimo_voice,
                },
                "stream": False,
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            url = f"{self.api_base.rstrip('/')}/chat/completions"

            r = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()

            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError(f"MiMo TTS 返回无 choices: {data}")

            audio_info = choices[0].get("message", {}).get("audio", {})
            audio_b64 = audio_info.get("data")
            if not audio_b64:
                raise RuntimeError(f"MiMo TTS 返回无音频数据: {choices[0]}")

            audio_bytes = base64.b64decode(audio_b64)
            combined_audio.extend(audio_bytes)

            # 估算该段时长（MiMo 不返回时间戳）
            seg_duration = estimate_speech_duration(seg)
            timestamps.append({
                "text": seg,
                "start": round(offset, 3),
                "end": round(offset + seg_duration, 3),
            })
            offset += seg_duration

        # 保存为 mp3（MiMo 返回 mp3 格式）
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mp3_path = output_path.with_suffix(".mp3")
        mp3_path.write_bytes(bytes(combined_audio))

        # 尝试用 ffmpeg 转 wav，失败则用 mp3
        try:
            import subprocess
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(mp3_path), "-acodec", "pcm_s16le",
                 "-ar", "16000", "-ac", "1", str(output_path)],
                capture_output=True, timeout=30,
            )
            if output_path.exists():
                mp3_path.unlink(missing_ok=True)
                final_path = output_path
            else:
                final_path = mp3_path
        except Exception:
            final_path = mp3_path

        duration = get_wav_duration(final_path) if final_path.suffix == ".wav" else offset

        self.logger.info(
            f"MiMo TTS 合成完成 duration={duration:.2f}s segments={len(segments)}"
        )
        return final_path, duration, timestamps

    def _synth_gpt_sovits(
        self, text: str, voice_id: str, output_path: Path,
        speed: float | None = None, volume: int | None = None,
        pitch: int | None = None, emotion: str | None = None,
    ) -> tuple[Path, float, list[dict]]:
        """调用 GPT-SoVITS 云端 API"""
        self.logger.info(f"GPT-SoVITS 合成 voice={voice_id} text_len={len(text)} speed={speed}")

        # 分句合成，便于时间戳对齐
        segments = split_text_to_segments(text)
        timestamps: list[dict] = []
        combined_audio = bytearray()
        sample_rate = 32000
        offset = 0.0
        # 语速：默认 1.0，支持外部传入精细控制
        tts_speed = speed if speed is not None else 1.0

        for seg in segments:
            payload = {
                "text": seg,
                "voice_id": voice_id,
                "speed": tts_speed,
            }
            resp = self.gpu.call_tts(payload)
            # 假设返回 base64 编码的 wav
            audio_b64 = resp.get("audio_base64") or resp.get("data", {}).get("audio_base64")
            if not audio_b64:
                raise RuntimeError(f"GPT-SoVITS 返回无音频数据: {resp}")
            audio_bytes = base64.b64decode(audio_b64)
            combined_audio.extend(audio_bytes)
            seg_duration = resp.get("duration", estimate_speech_duration(seg))
            timestamps.append({
                "text": seg,
                "start": round(offset, 3),
                "end": round(offset + seg_duration, 3),
            })
            offset += seg_duration
            if "sample_rate" in resp:
                sample_rate = resp["sample_rate"]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(bytes(combined_audio))
        duration = get_wav_duration(output_path) if output_path.exists() else offset

        self.logger.info(
            f"GPT-SoVITS 合成完成 duration={duration:.2f}s segments={len(segments)}"
        )
        return output_path, duration, timestamps

    def _synth_edge(
        self, text: str, voice_id: str, output_path: Path,
        speed: float | None = None, volume: int | None = None,
        pitch: int | None = None, emotion: str | None = None,
    ) -> tuple[Path, float, list[dict]]:
        """使用 edge-tts 合成（标准音色，无克隆）

        支持语速/音量/音高精细控制（edge-tts 库原生能力）：
        - speed: 0.5-2.0 倍率 → edge-tts rate "±N%"
        - volume: 0-200 百分比 → edge-tts volume "±N%"
        - pitch: -12 到 +12 半音 → edge-tts pitch "±NHz"（每半音约 4Hz）
        """
        try:
            import edge_tts
        except ImportError as e:
            raise RuntimeError(
                "Edge TTS 未安装，请在项目环境执行 python -m pip install 'edge-tts>=7.3.1'"
            ) from e

        # 构造 edge-tts 的 rate/volume/pitch 参数字符串
        kwargs: dict = {}
        if speed is not None and abs(speed - 1.0) > 0.01:
            rate_pct = int(round((speed - 1.0) * 100))
            kwargs["rate"] = f"{rate_pct:+d}%"
        if volume is not None and volume != 100:
            vol_pct = volume - 100
            kwargs["volume"] = f"{vol_pct:+d}%"
        if pitch is not None and pitch != 0:
            # 半音 → Hz 近似转换（每半音约 4Hz）
            pitch_hz = pitch * 4
            kwargs["pitch"] = f"{pitch_hz:+d}Hz"

        voice = voice_id if voice_id.endswith("Neural") else self.edge_voice
        self.logger.info(
            f"edge-tts 合成 voice={voice} "
            f"speed={speed} volume={volume} pitch={pitch} "
            f"kwargs={kwargs}"
        )

        async def _synth():
            communicate = edge_tts.Communicate(text, voice, **kwargs)
            await communicate.save(str(mp3_path))

        import tempfile
        from ..core.ffmpeg_utils import FFmpegRunner
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="freevideoai_tts_") as temp_dir:
            mp3_path = Path(temp_dir) / "speech.mp3"
            try:
                asyncio.run(_synth())
            except edge_tts.exceptions.NoAudioReceived as e:
                raise RuntimeError(
                    f"Edge TTS 未返回音频，音色 {voice} 可能不可用；"
                    "请在语音配置选择 zh-CN-XiaoxiaoNeural 或检查网络"
                ) from e
            FFmpegRunner().convert_audio(mp3_path, output_path, sample_rate=24000)

        duration = get_wav_duration(output_path)
        if duration <= 0:
            raise RuntimeError("Edge TTS 输出音频为空")
        timestamps = self._estimate_char_timestamps(text, duration)
        return output_path, duration, timestamps

    def _synth_mock(
        self, text: str, voice_id: str, output_path: Path,
        speed: float | None = None, volume: int | None = None,
        pitch: int | None = None, emotion: str | None = None,
    ) -> tuple[Path, float, list[dict]]:
        """Mock 模式：生成静音 wav，时长按文本估算"""
        duration = estimate_speech_duration(text)
        self.logger.info(
            f"Mock TTS 生成静音音频 voice={voice_id} "
            f"duration={duration:.2f}s text_len={len(text)}"
        )
        info = generate_silent_wav(output_path, duration)

        # 生成分句时间戳
        segments = split_text_to_segments(text)
        timestamps = []
        offset = 0.0
        total_chars = sum(len(s) for s in segments) or 1
        for seg in segments:
            seg_dur = duration * len(seg) / total_chars
            timestamps.append({
                "text": seg,
                "start": round(offset, 3),
                "end": round(offset + seg_dur, 3),
            })
            offset += seg_dur

        return info.path, info.duration, timestamps

    def register_voice(self, voice_id: str, sample_audio: Path) -> bool:
        """注册音色"""
        sample_audio = Path(sample_audio)
        voices_dir = Path(self.config.get("tts.voices_dir", "./config/voices"))
        voice_dir = voices_dir / voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)

        if self.provider != "gpt_sovits":
            # voxcpm / moss_nano / mimo / edge 模式：本地保存样本音频（用做零样本克隆参考）
            import shutil
            dest = voice_dir / f"sample{sample_audio.suffix or '.wav'}"
            shutil.copy2(sample_audio, dest)
            self.logger.info(f"本地音色注册成功: {voice_id} -> {dest}")
            # 用主 venv 的 faster-whisper 转写样本 → ref_text 存 meta.json
            # （VoxCPM 零样本克隆需要 prompt_text；转写失败不影响注册，合成时惰性补）
            ref_text = self._transcribe_sample(dest)
            meta = {
                "voice_id": voice_id,
                "sample": dest.name,
                "ref_text": ref_text,
                "ref_text_source": "faster-whisper" if ref_text else "",
                "created_at": time.time(),
            }
            try:
                (voice_dir / "meta.json").write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                self.logger.info(
                    f"音色 {voice_id} meta.json 已写入 "
                    f"ref_text_len={len(ref_text)}"
                )
            except Exception as e:
                self.logger.warning(f"meta.json 写入失败（不影响注册）: {e}")
            # moss_nano: 若样本不是 wav/48k，尝试转为标准 wav（MOSS 内部会重采样，但保留原始更稳）
            if self.provider == "moss_nano":
                self.logger.info(
                    f"音色 {voice_id} 已注册，MOSS 将用 {dest.name} 作为零样本克隆参考"
                )
            return True

        try:
            with open(sample_audio, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode()
            resp = self.gpu.call_tts_register({
                "voice_id": voice_id,
                "sample_audio_base64": audio_b64,
            })
            # 云端注册成功后也本地保存一份
            if resp.get("success"):
                import shutil
                dest = voice_dir / f"sample{sample_audio.suffix or '.wav'}"
                shutil.copy2(sample_audio, dest)
            return resp.get("success", False)
        except Exception as e:
            self.logger.error(f"音色注册失败: {e}")
            return False
