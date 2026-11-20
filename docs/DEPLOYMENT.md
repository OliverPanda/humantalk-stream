# FreeVideoAI Pro 部署指南

本文只介绍 Pro 原生部署，优先使用 Linux x86_64 服务器。
安装完成后，在 Web 配置 LLM、注册声音与形象即可开始生成。
快速命令见 [README](../README.md)。

## 1. 服务器要求与安装范围

- 第一张 NVIDIA 显卡总显存至少 18GB，`nvidia-smi` 正常；多卡显存不会合并。
- Linux x86_64，以下命令适用于 Ubuntu 22.04 / 24.04、Debian 12。
- 驱动需兼容 VoxCPM 的 PyTorch 2.5.1 cu121 和 MuseTalk 的 PyTorch 2.0.1 cu118。
- 安装器要求至少 29GiB 可用磁盘；另为下载缓存、任务中间文件与输出视频预留空间。
- 能访问 PyPI、PyTorch、OpenMMLab、GitHub、Hugging Face、Google Drive 和人脸检测模型下载站点。

Pro 自动安装主程序、VoxCPM-0.5B、MuseTalk 1.5 及对应依赖和权重，
配置 CPU Whisper 字幕识别、Linux 中文字体和模型路径。
LatentSync / EchoMimic 仅提供选装指引，需要按模型卡另行安装。
显存门槛按总显存检查，运行前还需确认没有其他进程大量占用显存。

## 2. Linux 安装

### 准备系统依赖与源码

```bash
nvidia-smi
sudo apt-get update
sudo apt-get install -y curl ca-certificates git build-essential ffmpeg \
  libsndfile1 libgl1 libglib2.0-0 fonts-noto-cjk
ffmpeg -version
ffprobe -version

git clone https://github.com/OliverPanda/humantalk-stream.git freevideoai
cd freevideoai
```

已有源码时进入项目根目录即可，使用包含当前安装脚本的版本。
其他发行版安装对应系统包，并确认中文字体路径可用。
安装入口会按需下载 uv 与 Python，并为主程序和两个模型创建独立环境，
无需手动准备系统 Python 虚拟环境。

### 安装 Pro 并应用配置

```bash
# 可选：只预览 Pro 计划；可能下载 uv 与 Python，不安装应用和模型
bash install.sh --profile pro --json
# 执行安装，并应用本地引擎及路径
bash install.sh --profile pro --install --activate
```

`--activate` 会覆盖推荐方案包含的现有字段，例如引擎、模型路径和字幕字体，
旧配置保存到 `config/user_config.yaml.bak`。已有环境执行前先核对配置。
不传 `--activate` 时保留已有字段，只补缺失字段，因此可能继续使用原来的引擎。

安装器检查依赖、CUDA 并实际加载模型；全部验证通过后才写推荐配置。
失败时不会应用本次推荐配置，已下载文件保留，修复后重跑同一命令。
首次耗时主要取决于依赖和权重下载，不要求手动逐项安装 Python 包。

| 路径 | 内容 |
|---|---|
| `workspace_data/runtime/app` | 主程序，Python 3.11 |
| `workspace_data/runtime/voxcpm` | VoxCPM 独立环境，Python 3.11 |
| `workspace_data/runtime/musetalk` | MuseTalk 独立环境，Python 3.10 |
| `workspace_data/runtime/models/VoxCPM-0.5B` | VoxCPM 权重 |
| `workspace_data/runtime/MuseTalk` | MuseTalk 源码及 `models/` 权重 |

## 3. 启动 Web

安装成功后，在项目根目录执行：

```bash
workspace_data/runtime/app/bin/python start.py --host 0.0.0.0 --port 8000
```

确保服务器防火墙／云安全组允许客户端访问 8000 端口，
浏览器打开 `http://服务器IP:8000`。
仅在服务器本机访问时可省略 `--host`，默认监听 `127.0.0.1`。
该命令在前台运行，停止后 Web 服务也会退出。

## 4. Web 配置与首次生成

| 页面 | 操作 | 完成标准 |
|---|---|---|
| 模型配置 → LLM | 填写服务地址、API Key、模型，测试连接并保存 | LLM 连接测试通过 |
| 模型配置 → TTS | 确认选择 VoxCPM | 使用安装器生成的本地环境与权重路径 |
| 模型配置 → 数字人 | 确认选择 MuseTalk 本地 | 使用安装器生成的本地环境、仓库和 worker 路径 |
| 我的声音 | 上传 5～20 秒干净人声并注册音色 | 创作时可选中该音色 |
| 我的形象 | 上传正面照片或 10～30 秒说话视频并注册形象 | 创作时可选中该形象 |
| 开始创作 | 选择声音与形象，输入 5～10 秒短文案并生成 | 配音可听、嘴型视频正常、中文字幕与封面显示正常 |

**本地 VoxCPM 和 MuseTalk 无需填写远程 TTS／数字人 API 地址或密钥。**
安装器已填写模型路径，一般无需手动修改。
参考说话视频可以保留身体动作，建议优先于单张照片。

基础口播只需配置实际使用的 LLM；AI 空镜、AI 生成形象属于可选功能，
启用时再填写视频中转站配置。LLM 和这些云端功能需要联网。
CPU Whisper small 权重会在首次字幕识别时下载，首次生成可能比后续慢。
先串行完成一条短视频验证，再生成完整内容。

## 5. 配置保存、备份与重启

生效优先级：环境变量 > `config/user_config.yaml` > `config/default.yaml`。
原生部署默认加载根目录 `.env`，可通过 `FREEVIDEOAI_ENV_FILE` 指定环境文件。
优先在 Web 或用户配置文件修改设置，避免直接改内置默认值。
网页保存仍不生效时，检查 `.env` 与进程环境中的同名变量，调整后重启服务。

安装器写配置、网页保存、重置单段和重置全部共用
`config/user_config.yaml.lock` 跨进程锁；读取、合并和保存／删除串行执行。
保存采用同目录临时文件原子替换，写入失败时保留原配置。
锁文件正常结束后可以保留，不要在操作期间删除；外部编辑器不参与此锁协议。

保存前备份到 `config/user_config.yaml.bak`，只保留最近一次保存前的版本。
“重置全部”删除用户配置，不会额外生成备份。
恢复前先停止 Web 服务与安装任务，再执行：

```bash
cp config/user_config.yaml.bak config/user_config.yaml
```

重新运行上面的启动命令即可，模型环境和权重无需重装。
完整数据备份还应包含 `.env`（或自定义环境文件）、`config/`、
`workspace_data/` 和 `output/`；单个 YAML 备份不包含素材、任务与作品。

原生 Web 的“模型配置 → 本机算力与模型库”也可安装推荐方案。
满足 Pro 条件时会推荐 Pro；需切换已有配置时勾选“应用推荐引擎”，完成后重启服务。
网页安装日志为 `workspace_data/runtime/install.log`，命令行安装直接输出日志。
安装任务通过 `workspace_data/runtime/install.lock` 互斥。

## 6. Windows Pro（次选）

同样要求第一张 NVIDIA 显卡总显存至少 18GB、驱动正常。
先安装可从 PATH 调用的 Python/pip、Git、FFmpeg 与 ffprobe，再进入项目根目录：

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1 --profile pro --install --activate
.\workspace_data\runtime\app\Scripts\python.exe start.py --host 0.0.0.0 --port 8000
```

Web 配置和首次验证步骤相同；各环境解释器使用 `Scripts/python.exe`。
MuseTalk 仍依赖上游 Windows wheel，缺失 wheel 或模型加载失败时安装会终止。
GPU 服务器部署优先选择 Linux。
恢复配置时使用 `Copy-Item config/user_config.yaml.bak config/user_config.yaml`。

## 7. 故障排查

| 现象 | 处理 |
|---|---|
| Pro 显存检查失败 | 确认第一张卡至少 18GB；多卡显存不相加，检查 `nvidia-smi` |
| 缺少 FFmpeg／中文字体 | 安装系统依赖，确认 `ffmpeg`、`ffprobe` 和中文字体文件存在 |
| CUDA 验证失败 | 检查驱动与两个模型的 PyTorch CUDA 环境兼容性 |
| 下载失败／Windows wheel 缺失 | 根据失败命令检查网络或平台兼容性，修复后重跑同一安装命令 |
| 安装后仍使用旧引擎 | 核对配置后使用 `--activate`，或在 Web 选择 VoxCPM 和 MuseTalk 本地并保存 |
| Web 访问不到 | 检查启动进程、`--host 0.0.0.0`、端口、防火墙与安全组 |
| 配置保存后未生效 | 检查 `.env` 和进程环境覆盖，调整后重启 |
| 首次字幕识别等待较久 | 确认 Whisper 权重下载网络正常 |
| 生成时显存不足 | 停止其他占用显存的任务，先串行生成短视频 |
| 提示已有安装任务 | 等待当前任务结束，查看日志；不要删除锁文件绕过互斥 |
