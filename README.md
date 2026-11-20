# FreeVideoAI 口播工坊 · 开源数字人口播视频工作室

> **Free**（免费）× **Video**（视频）× **AI** ——免费、本地可跑、按显卡自动分档的口播视频流水线。

**一句文案 → 一条带你自己的脸和声音的口播成片。** 本地可运行、按显卡自动分档、云端引擎可选的数字人视频自动化流水线：文案润色 → 声音克隆 → 数字人驱动 → 字幕 → 空镜混剪 → 合成 → 封面标题 → 发布清单，全流程一键。

![tier](https://img.shields.io/badge/GPU-18GB%2B-8A6D3F) ![python](https://img.shields.io/badge/python-3.10%2B-3E7C5B) ![license](https://img.shields.io/badge/license-MIT-4A6B8A)

## ✨ Pro 本地部署

本文部署流程面向第一张 NVIDIA 显卡总显存至少 18GB 的 Linux x86_64 服务器。
Pro 安装器自动部署 **VoxCPM-0.5B 声音克隆 + MuseTalk 1.5 本地数字人**，
并配置 CPU Whisper 字幕识别、中文字体和模型路径。
安装完成后，在 Web 配置 LLM、注册声音与形象，即可开始生成。

LatentSync / EchoMimic 需按模型卡指引另行安装，不包含在 Pro 自动安装中。
实际推理还需预留空闲显存；LLM 和可选云端功能需要联网。

## 🎬 核心特性

- **零样本声音克隆**：VoxCPM-0.5B，一段 5-20 秒干净人声即可克隆音色（连语气和说话习惯一起学），失败自动降级 MOSS-TTS → edge-tts
- **数字人驱动**：MuseTalk 1.5 本地嘴型同步（推荐用 10-30 秒说话视频做形象参考，保留身体动作）；或一键切换即梦 OmniHuman 云端全身驱动
- **自动空镜混剪**（可选）：LLM 规划插入点 → Seedance 2.0 / MiniMax H3 生成场景空镜 → 自动混剪
- **多景别虚拟机位**（可选）：同一条口播自动近景/中景交替，零成本提升节奏感
- **制片间**：生成过程九阶段可视化（打磨文案 → 录制配音 → 数字人演绎 → …），全程人话
- **完整后期**：词级卡拉OK字幕、BGM、滤镜转场、封面标题自动生成、多平台发布清单
- **工程化**：SQLite 状态机断点续跑、TTS 结果缓存、每阶段子进程独占 GPU 用完即还、降级链全程日志留痕

## 🚀 Pro 快速部署（Linux 优先）

以 Ubuntu 22.04 / 24.04 或 Debian 12 为例，先确认 NVIDIA 驱动正常，
第一张卡总显存至少 18GB。首次安装需要下载依赖和权重。

```bash
nvidia-smi
sudo apt-get update
sudo apt-get install -y curl ca-certificates git build-essential ffmpeg \
  libsndfile1 libgl1 libglib2.0-0 fonts-noto-cjk

git clone https://github.com/OliverPanda/humantalk-stream.git freevideoai
cd freevideoai
# 自动安装依赖和权重，验证通过后应用本地引擎及路径
bash install.sh --profile pro --install --activate
# 安装成功后再启动
workspace_data/runtime/app/bin/python start.py --host 0.0.0.0 --port 8000
```

已有源码时进入包含本次安装脚本的项目目录即可，无需再次克隆。
`--activate` 会覆盖推荐方案包含的现有字段，保存前备份为
`config/user_config.yaml.bak`；已有环境执行前先核对配置。
服务器需允许客户端访问 8000 端口，然后打开 `http://服务器IP:8000`：

1. **模型配置 → LLM**：填写服务地址、API Key 和模型，测试连接并保存。
2. **模型配置 → TTS／数字人**：确认分别为 **VoxCPM**、**MuseTalk 本地**；路径已自动填写，无需远程数字人 API。
3. **我的声音／我的形象**：上传 5～20 秒干净人声注册音色，上传正面照片或 10～30 秒说话视频注册形象。
4. **开始创作**：选择声音和形象，先用 5～10 秒短文案生成验证，再生成完整内容。

基础口播无需配置视频中转站；使用 AI 空镜或 AI 生成形象时再配置。
首次字幕识别可能下载 Whisper 权重。完整步骤、Windows Pro 安装、
配置恢复和故障排查见 [Pro 部署指南](docs/DEPLOYMENT.md)。

## ⚙️ 配置

**全部配置可在 Web 界面完成**（模型配置页，保存即热生效）；等价的文件入口：

- `.env` — 密钥与中转站地址，模板见 `.env.example`
- `config/default.yaml` — 内置默认值及字段说明
- `config/user_config.yaml` — 网页与安装器保存的用户覆盖配置，部署时优先修改此文件

注意：`.env` 中 `KRVOICEAI_*` 变量优先级高于界面设置（适合部署固化）；界面改了不生效时先检查 `.env`。
文件配置按“默认配置 → 用户配置 → 环境变量”覆盖，原生部署默认加载根目录 `.env`。

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
