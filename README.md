# FreeVideoAI 口播工坊 · 开源数字人口播视频工作室

> **Free**（免费）× **Video**（视频）× **AI** ——免费、本地可跑、按显卡自动分档的口播视频流水线。

**一句文案 → 一条带你自己的脸和声音的口播成片。** 本地可运行、按显卡自动分档、云端引擎可选的数字人视频自动化流水线：文案润色 → 声音克隆 → 数字人驱动 → 字幕 → 空镜混剪 → 合成 → 封面标题 → 发布清单，全流程一键。

![tier](https://img.shields.io/badge/GPU-6GB%20%E8%B5%B7%E6%AD%A5-8A6D3F) ![python](https://img.shields.io/badge/python-3.10%2B-3E7C5B) ![license](https://img.shields.io/badge/license-MIT-4A6B8A)

## ✨ 三条引擎路线，按你的机器自动分档

启动时自动检测显卡定档，不够档的重模型**在代码层被硬性锁定**（不会 OOM 拖垮任务）：

| 档位 | 硬件 | 数字人引擎 | 1 分钟口播耗时 | 成本 |
|---|---|---|---|---|
| **云端档** | 无独显也行 | 即梦数字人 OmniHuman（全身驱动，最流畅） | 排队+生成数分钟 | 1 元/秒 |
| **标准档** | 4-16GB（如 RTX 2060） | VoxCPM 音色克隆 + MuseTalk 嘴型 | ~8 分钟 | 免费 |
| **旗舰档** | ≥16GB（如 RTX 4090） | + LatentSync 1.6 / EchoMimic V3 本地（beta） | 1-10 分钟 | 免费 |

## 🎬 核心特性

- **零样本声音克隆**：VoxCPM-0.5B，一段 5-20 秒干净人声即可克隆音色（连语气和说话习惯一起学），失败自动降级 MOSS-TTS → edge-tts
- **数字人驱动**：MuseTalk 1.5 本地嘴型同步（推荐用 10-30 秒说话视频做形象参考，保留身体动作）；或一键切换即梦 OmniHuman 云端全身驱动
- **自动空镜混剪**（可选）：LLM 规划插入点 → Seedance 2.0 / MiniMax H3 生成场景空镜 → 自动混剪
- **多景别虚拟机位**（可选）：同一条口播自动近景/中景交替，零成本提升节奏感
- **制片间**：生成过程九阶段可视化（打磨文案 → 录制配音 → 数字人演绎 → …），全程人话
- **完整后期**：词级卡拉OK字幕、BGM、滤镜转场、封面标题自动生成、多平台发布清单
- **工程化**：SQLite 状态机断点续跑、TTS 结果缓存、每阶段子进程独占 GPU 用完即还、降级链全程日志留痕

## 🚀 快速开始

```bash
git clone https://github.com/333kjp/freevideoai && cd freevideoai
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -e .

cp .env.example .env        # 按需填密钥（LLM 必填，云端引擎可选）
python start.py
```

打开 http://127.0.0.1:8000 ：

1. **我的声音**：上传一段自己的人声 → 注册音色（即点即听原声）
2. **我的形象**：上传正面照片或 10-30 秒说话视频（视频效果远好于照片）
3. **开始创作**：输入文案 → 一键生成 → 制片间看进度 → 我的作品里回看下载

> 不装任何本地模型也能跑通：LLM + edge-tts + 云端数字人（模型配置 → 数字人模型中填写中转站配置）。本地模型（VoxCPM/MuseTalk）的安装指引见「模型配置 → 本机算力与模型库」每张模型卡。

## ⚙️ 配置

**全部配置可在 Web 界面完成**（模型配置页，保存即热生效）；等价的文件入口：

- `.env` — 密钥与中转站地址，模板见 `.env.example`
- `config/default.yaml` — 引擎选择、硬件分档覆盖、字幕/BGM/滤镜、空镜与多景别开关，每项有注释

注意：`.env` 中 `KRVOICEAI_*` 变量优先级高于界面设置（适合部署固化）；界面改了不生效时先检查 `.env`。

### 视频中转站

在「模型配置 → 云端引擎密钥」填写 OpenAI 风格视频接口的 Base URL（如 `https://mindspace.ccwu.cc/v1`）和密钥，点击“获取远端模型”后从下拉框选择。模型由中转站 `/models` 实时返回，不再使用内置候选名单；如远端未标注模型类型，列表会完整展示远端返回项。选中模型会立即保存，自动空镜和「AI 生成形象」共用这套视频接口。视频客户端使用 OpenAI Python SDK：创建任务走 SDK 的 JSON 请求入口以兼容中转站的 `images` 字段，查询和下载走 SDK 的视频接口。空镜功能仍需在配置中启用 `broll_auto.enabled`。

OmniHuman 是独立的「图片 + 配音 → 数字人」流程。中转站需兼容 `CVSubmitTask`、`CVGetResult` 及文件上传接口；在「模型配置 → 数字人模型」选择 OmniHuman 后填写 Base URL、API Key，并选择 v15 或 v1 模型。所给 mindspace 文档未列出 OmniHuman，不能仅通过填写它的视频 `/v1` 地址使用数字人流程。云端引擎不再读取火山 AK/SK、TOS 或 Ark 配置。

付费视频任务会在输出文件旁保存 `.task.json`，重试时先续查原任务。创建结果不明且中转站无法按请求号找回时，程序会停止再次提交；先向中转站核实是否已建单，再处理任务记录。OmniHuman 的视频任务 ID 同样保存在任务工作目录，超时后可续查；若提交响应丢失，也会停止自动重提。设置页的“清除中转配置”按钮可以移除已保存的地址、密钥和模型。

## 📂 架构

```
krvoiceai/
├── core/        # config / logger / ffmpeg / settings / hardware(显卡分档+显存护栏)
├── modules/     # tts_engine / avatar_engine / omnihuman_client / seedance_client
│                # broll_engine / multicam / subtitle / composer / cover / publisher
├── pipeline/    # SQLite 状态机编排：断点续跑 + 指数退避重试
└── web/         # FastAPI 60+ 端点 + 口播工坊 Web UI（vanilla JS，零构建）
```

设计原则：**每个模型是一个 provider 插槽**（TTS×6 / Avatar×9 / ASR×4，均含 mock），换模型=插新 provider；重模型走独立 venv 子进程，进程退出显存归还。

## ⚠️ 合规与授权（重要，商用前必读）

- **音色/肖像授权**：只克隆你本人或已取得书面授权的音色与形象。
- **AI 内容标识**：按《人工智能生成合成内容标识办法》（2025-09 施行），发布 AI 生成视频须显著标注；即梦 API 产物自带隐式水印元数据。
- **平台规则**：抖音/B站等平台要求 AI 内容声明，发布前自行检查。
- **第三方组件许可**：MuseTalk（MIT，可商用）、VoxCPM（Apache-2.0）、Wav2Lip（**仅限非商业用途**，商用请禁用该降级项）、faster-whisper（MIT）。云端 API 按所选中转站条款计费使用。

## 📜 License

本项目代码 MIT（见 LICENSE）。模型权重与第三方仓库遵循各自原始许可。
