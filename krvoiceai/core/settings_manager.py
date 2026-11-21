"""用户设置管理器

支持运行时通过 UI 修改模型配置（LLM/TTS/ASR/数字人/视频/发布），
持久化到 user_config.yaml，并热更新到全局 Config 单例与已加载组件。

设计要点：
- user_config.yaml 与 default.yaml 深度合并（用户配置优先）
- API Key 等敏感字段在 GET 时做掩码处理（仅返回是否已配置）
- 提供 LLM/TTS 连接测试能力，便于用户在保存前验证
- 保存后通知 app 实例重建 LLM 客户端等组件
"""
from __future__ import annotations

import os
import threading
from copy import deepcopy
from typing import Any, Callable, Optional

import httpx
import yaml

from .config import PROJECT_ROOT, _deep_merge, get_config
from .config_store import config_lock, write_config
from .logger import get_logger

logger = get_logger().bind(component="settings_manager")

USER_CONFIG_PATH = PROJECT_ROOT / "config" / "user_config.yaml"

# 敏感字段路径（点号分隔），GET 时掩码处理
SENSITIVE_PATHS = {
    "llm.api_key",
    "tts.api_key",
    "avatar.api_key",
    "publisher.cookies_dir",  # 路径不掩码，但 cookie 文件不返回
}

# 各 provider 的预设配置（前端下拉选择时使用）
PROVIDER_PRESETS = {
    "llm": {"deepseek": {
        "label": "DeepSeek Chat API", "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat"], "api_key_url": "https://platform.deepseek.com/api_keys",
    }},
    "tts": {"voxcpm": {
        "label": "VoxCPM-0.5B 本地克隆", "needs_api_base": False, "needs_api_key": False,
    }},
    "avatar": {"musetalk_local": {
        "label": "MuseTalk 1.5 本地", "needs_api_base": False, "needs_api_key": False,
    }},
    "asr": {"whisper_local": {
        "label": "Whisper small 本地 CPU int8", "models": ["small"],
    }},
}

# 字幕样式预设（对标剪映/简影，6 种风格）
SUBTITLE_STYLE_PRESETS = {
    "minimal_white": {
        "label": "简约白",
        "preview": "白字黑边，经典百搭",
        "primary_color": "&H00FFFFFF",
        "outline_color": "&H00000000",
        "outline_width": 2,
        "shadow_color": "&H80000000",
        "bold": True,
    },
    "tech_blue": {
        "label": "科技蓝",
        "preview": "蓝字白边，科技感强",
        "primary_color": "&H00FFD040",
        "outline_color": "&H00FFFFFF",
        "outline_width": 1,
        "shadow_color": "&H00000000",
        "bold": True,
    },
    "classic_gold": {
        "label": "古风金",
        "preview": "金字黑边，典雅古朴",
        "primary_color": "&H0000D4FF",
        "outline_color": "&H00000000",
        "outline_width": 2,
        "shadow_color": "&H80000000",
        "bold": True,
    },
    "pop_pink": {
        "label": "卡点粉",
        "preview": "粉字白边，活泼吸睛",
        "primary_color": "&H00FF80FF",
        "outline_color": "&H00FFFFFF",
        "outline_width": 2,
        "shadow_color": "&H00000000",
        "bold": True,
    },
    "news_red": {
        "label": "新闻红",
        "preview": "红字白边，严肃权威",
        "primary_color": "&H000000FF",
        "outline_color": "&H00FFFFFF",
        "outline_width": 2,
        "shadow_color": "&H80000000",
        "bold": True,
    },
    "dark_black": {
        "label": "极简黑",
        "preview": "黑字白边，硬核极客",
        "primary_color": "&H00000000",
        "outline_color": "&H00FFFFFF",
        "outline_width": 2,
        "shadow_color": "&H80000000",
        "bold": True,
    },
    "douyin_hot": {
        "label": "抖音爆款",
        "preview": "黄字黑边，爆款标配",
        "primary_color": "&H0000FFFF",
        "outline_color": "&H00000000",
        "outline_width": 3,
        "shadow_color": "&H80000000",
        "bold": True,
    },
    "neon_glow": {
        "label": "霓虹发光",
        "preview": "霓虹紫发光，赛博风",
        "primary_color": "&H00FF00FF",
        "outline_color": "&H00FF00FF",
        "outline_width": 4,
        "shadow_color": "&H00FF00FF",
        "bold": True,
    },
    "variety_pop": {
        "label": "综艺花字",
        "preview": "白字红边，综艺感",
        "primary_color": "&H00FFFFFF",
        "outline_color": "&H000000FF",
        "outline_width": 4,
        "shadow_color": "&H80000000",
        "bold": True,
    },
}

# 字幕入场动画预设（对标剪映/Sayatoo）
SUBTITLE_ANIMATION_PRESETS = {
    "none": {"label": "无动画", "description": "静态显示"},
    "fade": {"label": "淡入淡出", "description": "平滑过渡，适合大多数场景"},
    "slide": {"label": "滑入", "description": "从下方滑入，动感明快"},
    "zoom": {"label": "缩放", "description": "由小变大，强调重点"},
    "bounce": {"label": "弹跳", "description": "活泼俏皮，适合娱乐"},
    "typewriter": {"label": "打字机", "description": "逐字出现，复古风格"},
    "karaoke": {"label": "卡拉OK", "description": "逐字点亮，节奏感强"},
}

# 情感预设（对标腾讯智影/HeyGen）
EMOTION_PRESETS = {
    "neutral": {"label": "中性", "description": "自然平稳，适合大多数场景"},
    "calm": {"label": "平静", "description": "舒缓从容，适合讲述"},
    "excited": {"label": "激昂", "description": "热情饱满，适合带货"},
    "gentle": {"label": "温柔", "description": "柔和细腻，适合情感"},
    "serious": {"label": "严肃", "description": "庄重权威，适合新闻"},
    "cheerful": {"label": "欢快", "description": "明朗活泼，适合科普"},
}

# 数字人姿态预设（对标腾讯智影）
POSE_PRESETS = {
    "standing": {"label": "站姿全身", "description": "全身站立，适合展示"},
    "sitting": {"label": "坐姿半身", "description": "坐着半身，适合访谈"},
    "half_body": {"label": "半身", "description": "腰部以上，口播标配"},
    "closeup": {"label": "特写", "description": "面部特写，表情清晰"},
}

# 滤镜预设（对标剪映/vsub，10 种滤镜）
FILTER_PRESETS = {
    "none": {"label": "无滤镜", "description": "原始画面"},
    "warm": {"label": "暖色", "description": "偏黄偏红，温馨感"},
    "cool": {"label": "冷色", "description": "偏蓝偏青，科技感"},
    "bw": {"label": "黑白", "description": "去色黑白，复古感"},
    "vintage": {"label": "复古", "description": "褪色颗粒，怀旧感"},
    "vivid": {"label": "鲜艳", "description": "高饱和度，吸睛感"},
    "cinematic": {"label": "电影感", "description": "青橙色调+暗角，大片质感"},
    "vlog": {"label": "Vlog清新", "description": "提亮降对比，日系清新"},
    "film": {"label": "胶片", "description": "降饱和+颗粒感，胶片质感"},
    "noir": {"label": "黑色电影", "description": "高对比黑白+暗角，硬核风"},
    "summer": {"label": "夏日", "description": "鲜艳偏青绿，夏日清新"},
}

# 转场预设（对标剪映，10+ 种转场）
TRANSITION_PRESETS = {
    "none": {"label": "无转场", "description": "硬切"},
    "fade": {"label": "淡入淡出", "description": "平滑过渡，最常用"},
    "fadeblack": {"label": "淡入黑场", "description": "章节分隔，情绪停顿"},
    "fadewhite": {"label": "闪白", "description": "强调冲击，回忆闪回"},
    "dissolve": {"label": "叠化", "description": "柔和过渡，时间流逝"},
    "slideleft": {"label": "左滑", "description": "B-roll切换，节奏感"},
    "slideright": {"label": "右滑", "description": "B-roll切换，节奏感"},
    "slideup": {"label": "上滑", "description": "动感切换"},
    "slidedown": {"label": "下滑", "description": "动感切换"},
    "circleopen": {"label": "圆形展开", "description": "创意转场，聚焦主体"},
    "zoomin": {"label": "缩放进入", "description": "强调瞬间，产品揭示"},
    "hblur": {"label": "水平模糊", "description": "柔和切换，梦境感"},
    "pixelize": {"label": "像素化", "description": "故障风，数字风转场"},
}

# 所有创作预设汇总（供前端一次性加载）
CREATIVE_PRESETS = {
    "subtitle_styles": SUBTITLE_STYLE_PRESETS,
    "subtitle_animations": SUBTITLE_ANIMATION_PRESETS,
    "emotions": EMOTION_PRESETS,
    "poses": POSE_PRESETS,
    "filters": FILTER_PRESETS,
    "transitions": TRANSITION_PRESETS,
}


class SettingsManager:
    """用户设置管理器（线程安全单例）"""

    _instance: Optional["SettingsManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._listeners: list[Callable[[dict], None]] = []
        self._load_user_config()

    def _load_user_config(self) -> None:
        """加载 user_config.yaml（若不存在则空 dict）"""
        self._user_data: dict[str, Any] = {}
        if USER_CONFIG_PATH.exists():
            try:
                with open(USER_CONFIG_PATH, "r", encoding="utf-8") as f:
                    self._user_data = yaml.safe_load(f) or {}
            except Exception as e:
                logger.error(f"加载 user_config.yaml 失败: {e}")
                self._user_data = {}

    # ============ 读取 ============

    def get_all(self, mask_sensitive: bool = True) -> dict[str, Any]:
        """获取合并后的完整配置（默认对敏感字段掩码）"""
        # 重新加载配置（合并 default + user + env）
        cfg = get_config(reload=True)
        data = cfg.as_dict()
        if mask_sensitive:
            data = self._mask_sensitive(data)
        # 附加元信息：哪些是用户自定义的
        data["_meta"] = {
            "user_config_path": str(USER_CONFIG_PATH),
            "user_config_exists": USER_CONFIG_PATH.exists(),
            "user_overridden_paths": self._collect_user_paths(self._user_data),
        }
        return data

    def get_section(self, section: str, mask_sensitive: bool = True) -> dict[str, Any]:
        """获取某个配置段（如 llm / tts）"""
        data = self.get_all(mask_sensitive=mask_sensitive)
        return data.get(section, {}) or {}

    def get_provider_presets(self) -> dict[str, Any]:
        """获取 provider 预设（供前端下拉选择）"""
        return deepcopy(PROVIDER_PRESETS)

    def get_creative_presets(self) -> dict[str, Any]:
        """获取创作预设（字幕样式/动画/情感/姿态/滤镜/转场）"""
        return deepcopy(CREATIVE_PRESETS)

    def get_templates(self) -> dict[str, Any]:
        """获取创作模板列表"""
        cfg = get_config()
        templates = cfg.get("templates", {}) or {}
        return deepcopy(templates)

    def apply_template(self, template_id: str) -> dict[str, Any]:
        """应用创作模板：一键设置字幕/BGM/情感/滤镜/转场

        Args:
            template_id: 模板 ID（如 news_broadcast）

        Returns:
            {"success": bool, "message": str, "applied": {...}}
        """
        cfg = get_config()
        templates = cfg.get("templates", {}) or {}
        if template_id not in templates:
            return {"success": False, "message": f"模板不存在: {template_id}"}

        tpl = templates[template_id]
        applied: dict[str, Any] = {}

        # 1. 字幕样式预设 + 动画
        style_preset = tpl.get("subtitle_preset", "minimal_white")
        animation = tpl.get("subtitle_animation", "fade")
        style = SUBTITLE_STYLE_PRESETS.get(style_preset, {})
        subtitle_payload = {
            "preset": style_preset,
            "animation": animation,
            "primary_color": style.get("primary_color", "&H00FFFFFF"),
            "outline_color": style.get("outline_color", "&H00000000"),
            "outline_width": style.get("outline_width", 2),
            "shadow_color": style.get("shadow_color", "&H80000000"),
            "bold": style.get("bold", True),
        }
        self.update_section("subtitle", subtitle_payload)
        applied["subtitle"] = subtitle_payload

        # 2. BGM + 情感
        audio_payload = {
            "emotion": tpl.get("emotion", "neutral"),
            "bgm": {
                "enabled": True,
                "track": tpl.get("bgm_track", "soft_piano"),
                "volume": 15,
                "fade_in": 1.0,
                "fade_out": 1.0,
            },
        }
        self.update_section("audio", audio_payload)
        applied["audio"] = audio_payload

        # 3. 滤镜 + 转场
        effects_payload = {
            "transition": tpl.get("transition", "fade"),
            "filter": tpl.get("filter", "none"),
        }
        self.update_section("effects", effects_payload)
        applied["effects"] = effects_payload

        logger.info(f"已应用创作模板: {template_id} ({tpl.get('label', '')})")
        return {
            "success": True,
            "message": f"已应用模板「{tpl.get('label', template_id)}」",
            "template_id": template_id,
            "applied": applied,
        }

    def _mask_sensitive(self, data: dict[str, Any]) -> dict[str, Any]:
        """对 API Key 等敏感字段掩码"""
        result = deepcopy(data)

        def mask(val: Any) -> Any:
            if not val or not isinstance(val, str):
                return ""
            if len(val) <= 8:
                return "****"
            return val[:4] + "*" * (len(val) - 8) + val[-4:]

        # llm.api_key
        if isinstance(result.get("llm"), dict):
            result["llm"]["api_key"] = mask(result["llm"].get("api_key", ""))
            result["llm"]["api_key_configured"] = bool(data.get("llm", {}).get("api_key"))
        if isinstance(result.get("tts"), dict):
            result["tts"]["api_key"] = mask(result["tts"].get("api_key", ""))
            result["tts"]["api_key_configured"] = bool(data.get("tts", {}).get("api_key"))
        if isinstance(result.get("avatar"), dict):
            result["avatar"]["api_key"] = mask(result["avatar"].get("api_key", ""))
            result["avatar"]["api_key_configured"] = bool(data.get("avatar", {}).get("api_key"))
        return result

    def _collect_user_paths(self, data: dict, prefix: str = "") -> list[str]:
        """收集用户自定义的配置路径（用于前端高亮显示）"""
        paths: list[str] = []
        for k, v in data.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                paths.extend(self._collect_user_paths(v, path))
            else:
                paths.append(path)
        return paths

    # ============ 写入 ============

    def update_section(self, section: str, payload: dict[str, Any]) -> dict[str, Any]:
        """更新某个配置段（如 llm），合并到 user_config.yaml 并热更新

        Args:
            section: 配置段名（llm/tts/avatar/asr/composer/cover/publisher/pipeline）
            payload: 该段的新字段（与 default.yaml 中该段结构一致）

        Returns:
            {"success": bool, "message": str, "section": str}
        """
        if section not in ("llm", "tts", "avatar", "asr", "composer",
                           "cover", "publisher", "pipeline", "project", "logging",
                           "subtitle", "scene", "audio", "effects", "originality",
                           "broll_auto", "multicam"):
            return {"success": False, "message": f"不允许修改的配置段: {section}", "section": section}

        with self._lock, config_lock(USER_CONFIG_PATH):
            self._load_user_config()  # 重新读取，避免外部修改
            payload = self._unmask_sensitive(section, payload)
            self._user_data[section] = _deep_merge(
                self._user_data.get(section, {}), payload
            )
            self._save_user_config()

        # 热更新全局配置
        get_config(reload=True)

        # 通知监听器（app 实例会重建 LLM/TTS 等组件）
        merged_section = self.get_section(section, mask_sensitive=False)
        self._notify_listeners({section: merged_section})

        logger.info(f"配置段已更新并持久化: {section}")
        return {
            "success": True,
            "message": f"配置段 {section} 已保存",
            "section": section,
            "data": self.get_section(section, mask_sensitive=True),
        }

    def reset_section(self, section: str) -> dict[str, Any]:
        """重置某段为默认配置（删除 user_config.yaml 中该段）"""
        with self._lock, config_lock(USER_CONFIG_PATH):
            self._load_user_config()
            if section in self._user_data:
                del self._user_data[section]
                self._save_user_config()
        get_config(reload=True)
        self._notify_listeners({"_reset": section})
        return {"success": True, "message": f"配置段 {section} 已重置为默认", "section": section}

    def reset_all(self) -> dict[str, Any]:
        """重置全部用户配置"""
        with self._lock, config_lock(USER_CONFIG_PATH):
            if USER_CONFIG_PATH.exists():
                USER_CONFIG_PATH.unlink()
            self._user_data = {}
        get_config(reload=True)
        self._notify_listeners({"_reset_all": True})
        return {"success": True, "message": "全部用户配置已重置"}

    def _unmask_sensitive(self, section: str, payload: dict[str, Any]) -> dict[str, Any]:
        """处理掩码字段：若 API Key 形如 'sk-x****abcd' 则保留原值"""
        if section in ("llm", "tts", "avatar") and "api_key" in payload:
            val = payload.get("api_key", "")
            # 形如 xxxx****xxxx 视为掩码，未修改
            if isinstance(val, str) and "****" in val:
                # 读取原值
                cfg = get_config(reload=True)
                original = cfg.get(f"{section}.api_key", "")
                payload["api_key"] = original
            # 空字符串视为清空，允许
        # 移除前端附加的元字段
        payload.pop("api_key_configured", None)
        return payload

    def _save_user_config(self) -> None:
        """保存到 user_config.yaml"""
        write_config(USER_CONFIG_PATH, self._user_data)
        logger.info(f"用户配置已保存: {USER_CONFIG_PATH}")

    # ============ 监听器（热更新） ============

    def add_listener(self, callback: Callable[[dict], None]) -> None:
        """注册配置变更监听器"""
        self._listeners.append(callback)

    def _notify_listeners(self, change: dict) -> None:
        for cb in self._listeners:
            try:
                cb(change)
            except Exception as e:
                logger.error(f"配置变更监听器执行失败: {e}")

    # ============ 连接测试 ============

    def test_llm(self, payload: dict[str, Any]) -> dict[str, Any]:
        """测试 LLM 连接

        payload: {provider, api_key, base_url, model}
        """
        provider = payload.get("provider", "mock")
        api_key = payload.get("api_key", "")
        base_url = (payload.get("base_url") or "").rstrip("/")
        model = payload.get("model", "")

        if provider == "mock":
            return {"success": True, "message": "Mock 模式无需测试，始终可用"}
        if not api_key:
            return {"success": False, "message": "API Key 未填写"}
        if not base_url:
            return {"success": False, "message": "base_url 未填写"}
        if not model:
            return {"success": False, "message": "model 未填写"}

        # 处理掩码
        if "****" in api_key:
            cfg = get_config()
            api_key = cfg.get("llm.api_key", "")

        url = f"{base_url}/chat/completions"
        test_payload = {
            "model": model,
            "messages": [{"role": "user", "content": "你好，请回复'连接成功'四个字"}],
            "max_tokens": 20,
            "temperature": 0.1,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            start = os.times().elapsed
            r = httpx.post(url, json=test_payload, headers=headers, timeout=30)
            elapsed = os.times().elapsed - start
            if r.status_code == 200:
                data = r.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                usage = data.get("usage", {})
                return {
                    "success": True,
                    "message": f"连接成功，模型回复: {content[:50]}",
                    "elapsed_ms": int(elapsed * 1000),
                    "usage": usage,
                    "model": model,
                }
            else:
                err_msg = ""
                try:
                    err_data = r.json()
                    err_msg = err_data.get("error", {}).get("message", "") or str(err_data)
                except Exception:
                    err_msg = r.text[:200]
                return {
                    "success": False,
                    "message": f"HTTP {r.status_code}: {err_msg}",
                    "status_code": r.status_code,
                }
        except httpx.ConnectError:
            return {"success": False, "message": f"无法连接到 {base_url}，请检查地址和网络"}
        except httpx.TimeoutException:
            return {"success": False, "message": f"连接超时（30s），请检查 {base_url}"}
        except Exception as e:
            return {"success": False, "message": f"测试失败: {e}"}

    def test_tts(self, payload: dict[str, Any]) -> dict[str, Any]:
        """测试 TTS 服务连接

        payload: {provider, api_base, api_key}
        """
        provider = payload.get("provider", "mock")
        api_base = (payload.get("api_base") or "").rstrip("/")
        api_key = payload.get("api_key", "")

        if provider == "mock":
            return {"success": True, "message": "Mock 模式无需测试，始终可用"}
        if provider == "edge_tts":
            # edge-tts 不需要服务，尝试 import
            try:
                import edge_tts  # noqa: F401
                return {"success": True, "message": "edge-tts 可用"}
            except ImportError:
                return {"success": False, "message": "edge-tts 未安装，请执行 pip install edge-tts"}
        if provider == "voxcpm":
            from .config import resolve_path
            from .pro_models import require_model
            config = get_config()
            try:
                executable = resolve_path(config.get("tts.voxcpm.venv_python", ""))
                if not executable.is_file():
                    raise RuntimeError(f"VoxCPM 独立 Python 不存在: {executable}")
                require_model("voxcpm", resolve_path(config.get("tts.voxcpm.model_dir", "")))
                return {"success": True, "message": "VoxCPM 本地权重与 Python 已就绪；推理以短文案生成验证。"}
            except RuntimeError as exc:
                return {"success": False, "message": str(exc)}
        if not api_base:
            return {"success": False, "message": "服务地址未填写"}

        # 处理掩码
        if "****" in api_key:
            cfg = get_config()
            api_key = cfg.get("tts.api_key", "")

        # MiMo TTS：调用 chat/completions 端点合成短文本验证
        if provider == "mimo":
            import base64
            import time as _time
            mimo_model = get_config().get("tts.mimo_model", "mimo-v2.5-tts")
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            test_payload = {
                "model": mimo_model,
                "messages": [{"role": "assistant", "content": "测试"}],
                "audio": {"format": "mp3", "voice": "mimo_default"},
                "stream": False,
            }
            try:
                t0 = _time.time()
                r = httpx.post(
                    f"{api_base}/chat/completions",
                    json=test_payload, headers=headers, timeout=30,
                )
                elapsed = (_time.time() - t0) * 1000
                if r.status_code == 200:
                    data = r.json()
                    audio_data = data.get("choices", [{}])[0].get("message", {}).get("audio", {}).get("data", "")
                    if audio_data:
                        audio_bytes = base64.b64decode(audio_data)
                        return {
                            "success": True,
                            "message": f"MiMo TTS 连接成功，合成 {len(audio_bytes)} 字节音频",
                            "elapsed_ms": round(elapsed),
                            "model": mimo_model,
                        }
                    return {"success": False, "message": "MiMo TTS 响应中无音频数据", "elapsed_ms": round(elapsed)}
                return {"success": False, "message": f"MiMo TTS 返回 HTTP {r.status_code}: {r.text[:200]}", "elapsed_ms": round(elapsed)}
            except Exception as e:
                return {"success": False, "message": f"MiMo TTS 连接失败: {e}"}

        # GPT-SoVITS 健康检查：尝试 GET 根路径或 /health
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        for path in ("/", "/health", "/ping"):
            try:
                r = httpx.get(f"{api_base}{path}", headers=headers, timeout=10)
                if r.status_code < 500:
                    return {
                        "success": True,
                        "message": f"GPT-SoVITS 服务可达 ({r.status_code})",
                        "endpoint": f"{api_base}{path}",
                    }
            except Exception:
                continue
        return {"success": False, "message": f"无法连接到 GPT-SoVITS 服务: {api_base}"}

    def test_avatar(self, payload: dict[str, Any]) -> dict[str, Any]:
        """测试数字人服务连接"""
        provider = payload.get("provider", "mock")
        api_base = (payload.get("api_base") or "").rstrip("/")

        if provider == "mock":
            return {"success": True, "message": "Mock 模式无需测试，始终可用"}
        if provider == "musetalk_local":
            from ..modules.avatar_engine import AvatarEngine
            ready, reason = AvatarEngine(config=get_config())._check_musetalk_env()
            return {"success": ready, "message": "MuseTalk 本地环境检查通过" if ready else reason}
        if provider == "omnihuman_api":
            from ..modules.omnihuman_client import OmniHumanClient
            ready, reason = OmniHumanClient().available()
            return {"success": ready, "message": "OmniHuman 中转站配置已就绪（真实连通性以首次生成为准）" if ready else reason}
        if provider == "musetalk_api":
            from types import SimpleNamespace
            from ..modules.musetalk_api_client import MuseTalkAPIClient
            key = payload.get("api_key") or ""
            if "****" in key:
                key = get_config().get("avatar.api_key", "")
            values = {"avatar.api_base": api_base, "avatar.api_key": key}
            try:
                data = MuseTalkAPIClient(SimpleNamespace(get=values.get)).health()
                return {"success": True, "message": f"MuseTalk-API 就绪，GPU: {data.get('gpu_name', '未知')}"}
            except Exception as exc:
                return {"success": False, "message": f"MuseTalk-API 连接失败: {exc}"}
        if not api_base:
            return {"success": False, "message": "服务地址未填写"}

        for path in ("/", "/health", "/ping"):
            try:
                r = httpx.get(f"{api_base}{path}", timeout=10)
                if r.status_code < 500:
                    return {
                        "success": True,
                        "message": f"数字人服务可达 ({r.status_code})",
                        "endpoint": f"{api_base}{path}",
                    }
            except Exception:
                continue
        return {"success": False, "message": f"无法连接到数字人服务: {api_base}"}


# 全局单例
_settings_manager: Optional[SettingsManager] = None


def get_settings_manager() -> SettingsManager:
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager
