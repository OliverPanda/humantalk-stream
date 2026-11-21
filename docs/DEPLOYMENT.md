# FreeVideoAI Pro 部署指南

本文只介绍 Pro 原生部署，优先使用 Linux x86_64 服务器。
安装完成后，在 Web 配置 LLM、注册声音与形象即可开始生成。
快速命令见 [README](../README.md)。

## 1. 服务器要求与安装范围

- 第一张 NVIDIA 显卡总显存至少 18GB，`nvidia-smi` 正常；多卡显存不会合并。
- Linux x86_64，以下命令适用于 Ubuntu 22.04 / 24.04、Debian 12。
- 驱动需兼容 VoxCPM 的 PyTorch 2.5.1 cu121 和 MuseTalk 的 PyTorch 2.0.1 cu118。
- 项目所在磁盘至少有 29GiB 可用空间；另为下载缓存、任务中间文件与输出视频预留空间。
- 能访问下文列出的镜像站，以及 OpenMMLab 官方 wheel 地址；首次取得源码还需访问 GitHub 或自行同步源码。

24GB 显存满足 Pro 的显存门槛，但不能代替磁盘空间。进入准备安装的项目目录后，
用 `df -h .` 检查该目录所在挂载点；根盘空间不足时，先将源码放到容量足够的数据盘，
再执行安装。虚拟环境和生成的模型路径与安装目录有关，安装后不要直接搬动运行目录。

Pro 自动安装主程序、VoxCPM-0.5B、MuseTalk 1.5 及对应依赖和权重，
配置 CPU Whisper 字幕识别、Linux 中文字体和模型路径。
LatentSync / EchoMimic 仅提供选装指引，需要按模型卡另行安装。
显存门槛按总显存检查，运行前还需确认没有其他进程大量占用显存。

## 2. Linux 安装

### 准备系统依赖与源码

```bash
nvidia-smi
sudo apt-get update
sudo apt-get install -y git curl ca-certificates
git clone https://github.com/OliverPanda/humantalk-stream.git freevideoai
cd freevideoai
# root 用户省略 sudo
sudo bash scripts/install_system_deps.sh
ffmpeg -version
ffprobe -version
```

已有源码时进入项目根目录即可，使用包含当前安装脚本的版本。
其他发行版安装对应系统包，并确认中文字体路径可用。
安装入口会按需下载 uv 与 Python，并为主程序和两个模型创建独立环境，
无需手动准备系统 Python 虚拟环境。

系统依赖脚本通过临时 APT 源安装 Ubuntu／Debian 依赖，不改写系统源文件。
包括 FFmpeg、中文字体和 Python/pip；已有环境也可重跑。

### 默认镜像与自定义源

默认读取 `config/install-mirrors.env`，无需在每次安装前手动 export。
Linux、Windows、网页安装子进程、Docker 构建和首次 Whisper 下载共用这套配置。
已有非空环境变量优先于文件中的默认值。
镜像配置是简单的 `KEY=value` 数据文件，每行一个字段，不写 `export`、引号或 Shell 表达式。
可以修改该文件中的公开源地址，或通过 `FREEVIDEOAI_MIRROR_CONFIG` 指向自己的同格式文件；
它独立于 `.env`、`config/user_config.yaml`，不用于保存模型 API 密钥。

| 下载内容 | 默认源／方式 | 自定义变量 |
|---|---|---|
| pip / uv Python 包 | 清华 PyPI 镜像 | `PIP_INDEX_URL` / `UV_INDEX_URL` |
| uv 管理的 Python | 南京大学 python-build-standalone 镜像 | `UV_PYTHON_INSTALL_MIRROR` |
| PyTorch CUDA wheel | 上海交大 pytorch-wheels，自动追加 `cu121` / `cu118` | `FREEVIDEOAI_TORCH_INDEX` |
| VoxCPM、MuseTalk、Whisper、人脸附属权重 | HF 镜像，附属权重使用固定仓库版本 | `HF_ENDPOINT` |
| MuseTalk Git 源码 | ghfast GitHub 代理，保留原仓库 origin 与固定提交 | `FREEVIDEOAI_GITHUB_MIRROR` |
| Ubuntu / Debian 系统依赖 | 清华 APT 镜像，仅在系统依赖脚本／Docker 中使用 | `FREEVIDEOAI_APT_MIRROR` |
| Chromium（Docker 构建） | npmmirror Playwright 二进制镜像 | `PLAYWRIGHT_DOWNLOAD_HOST` |
| MMCV cu118 / torch2.0 wheel | 官方 OpenMMLab 地址，未找到可验证的公共替代源 | `FREEVIDEOAI_MMCV_INDEX` |

`FREEVIDEOAI_TORCH_INDEX` 填 wheel 根地址，不追加 `cu121`／`cu118`；
`UV_PYTHON_INSTALL_MIRROR` 填 Python release 镜像根目录；
`FREEVIDEOAI_GITHUB_MIRROR` 是替换 `https://github.com` 的 URL 前缀；
`FREEVIDEOAI_MMCV_INDEX` 则填写完整的 wheel 索引页面地址。
保持 HTTPS 证书验证，不需要添加 `--trusted-host` 或关闭 SSL 验证。

VoxCPM 和 MuseTalk 的 CUDA 版本、依赖版本不变，不会从普通 PyPI 镜像误装 CPU 版。
人脸解析权重和 S3FD 也走 HF 下载，不再要求直连 Google Drive 或原检测模型站点。
Linux 已有 Python/pip 时，uv 本身从 PyPI 镜像安装到用户目录；
缺少 Python/pip 时仍使用官方 uv 安装脚本，因此建议先执行系统依赖脚本。

镜像不可用时可在命令前指定自己的源，例如：

```bash
HF_ENDPOINT=https://你的模型镜像地址 \
  bash install.sh --profile pro --install --activate
```

若要同时替换 Python 包源，应设置 pip 和 uv 两个变量：

```bash
PIP_INDEX_URL=https://你的包镜像地址/simple \
UV_INDEX_URL=https://你的包镜像地址/simple \
  bash install.sh --profile pro --install --activate
```

系统依赖脚本通过 sudo 执行时，sudo 可能过滤当前环境中的变量。
需要指定 APT 镜像时，在 sudo 后显式传入：

```bash
sudo env FREEVIDEOAI_APT_MIRROR=https://你的APT镜像地址 \
  bash scripts/install_system_deps.sh
```

或跳过项目的全部镜像默认值，使用官方源：

```bash
FREEVIDEOAI_DOWNLOAD_SOURCE=official bash install.sh --profile pro --install --activate
```

该开关不会清除你自己已导出的环境变量；若之前手动设过镜像变量，需在新终端执行，
或先取消对应变量。它也不会改回系统既有的 APT／pip 全局配置。
需要仅为本次命令取消常见的手动源设置时，可执行：

```bash
env -u HF_ENDPOINT -u PIP_INDEX_URL -u UV_INDEX_URL \
  -u UV_PYTHON_INSTALL_MIRROR -u FREEVIDEOAI_TORCH_INDEX \
  -u FREEVIDEOAI_GITHUB_MIRROR -u FREEVIDEOAI_MMCV_INDEX \
  FREEVIDEOAI_DOWNLOAD_SOURCE=official \
  bash install.sh --profile pro --install --activate
```

| 网络参数 | 默认值 |
|---|---|
| `HF_HUB_ETAG_TIMEOUT` | 元数据请求 60 秒 |
| `HF_HUB_DOWNLOAD_TIMEOUT` | 文件下载 120 秒 |
| `HF_HUB_DISABLE_XET` | `1`，默认禁用 Xet 下载通道 |
| `PIP_TIMEOUT` / `PIP_RETRIES` | 120 秒 / 5 次 |
| `UV_HTTP_TIMEOUT` / `UV_HTTP_RETRIES` | 120 秒 / 5 次 |

这些是请求等待和重试参数，不限制整次安装的总时长。
离线变量 `HF_HUB_OFFLINE=1` 会阻止下载，在线安装前应取消。

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

### 已有服务器继续安装

先将包含本次镜像配置的源码同步到服务器，再进入原安装目录。
下载失败时不要删除 `workspace_data/runtime/`，重跑安装器会使用已有环境及下载缓存；
它仍会重新检查依赖、执行下载检查和模型验证，不表示完全跳过前面步骤。

```bash
df -h .
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
set -o pipefail
bash install.sh --profile pro --install --activate 2>&1 | tee install-pro.log
```

日志位于当前目录的 `install-pro.log`，每次执行上述命令会覆盖该日志。
`set -o pipefail` 保证安装失败时不会被 `tee` 的成功状态掩盖。
如果失败点是 VoxCPM 下载，检查最早的连接／超时／证书异常；
最后的 `LocalEntryNotFoundError` 不能单独确定网络失败原因。

## 3. 启动 Web

安装成功后，在项目根目录执行：

```bash
workspace_data/runtime/app/bin/python start.py --host 0.0.0.0 --port 8000
```

确保服务器防火墙／云安全组允许客户端访问 8000 端口，
浏览器打开 `http://服务器IP:8000`。
仅在服务器本机访问时可省略 `--host`，默认监听 `127.0.0.1`。
该命令在前台运行，停止后 Web 服务也会退出。
应使用上面的安装器环境解释器，不要仅凭终端显示 `(app)` 就使用裸 `python`。
`ModuleNotFoundError: No module named 'yaml'` 通常表示安装未完成或解释器选错；
先处理安装失败，再启动。参数是 `--host`，不是 `00host`。

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

Windows 同样默认加载镜像配置。临时指定模型镜像时：

```powershell
$env:HF_ENDPOINT = 'https://你的模型镜像地址'
powershell -ExecutionPolicy Bypass -File .\install.ps1 --profile pro --install --activate
```

切回项目的官方默认值时，先取消自己设置的镜像变量，再设置
`$env:FREEVIDEOAI_DOWNLOAD_SOURCE = 'official'`；该开关不会主动清除已有环境变量。

Web 配置和首次验证步骤相同；各环境解释器使用 `Scripts/python.exe`。
MuseTalk 仍依赖上游 Windows wheel，缺失 wheel 或模型加载失败时安装会终止。
GPU 服务器部署优先选择 Linux。
恢复配置时使用 `Copy-Item config/user_config.yaml.bak config/user_config.yaml`。

## 7. 故障排查

| 现象 | 处理 |
|---|---|
| Pro 显存检查失败 | 确认第一张卡至少 18GB；多卡显存不相加，检查 `nvidia-smi` |
| 提示至少需要 29GB 磁盘 | 用 `df -h .` 检查项目所在挂载点；显卡的 24GB 是显存，不是硬盘空间 |
| 缺少 FFmpeg／中文字体 | 安装系统依赖，确认 `ffmpeg`、`ffprobe` 和中文字体文件存在 |
| CUDA 验证失败 | 检查驱动与两个模型的 PyTorch CUDA 环境兼容性 |
| 下载失败／Windows wheel 缺失 | 查看最早的网络异常；检查镜像、代理及离线变量，必要时自定义源或切回官方源；wheel 缺失需检查平台兼容性 |
| `LocalEntryNotFoundError` | 查看此异常之前的原始错误，确认不是离线模式；按“已有服务器继续安装”保存日志并重跑 |
| `No module named 'yaml'` | 完成整次安装，并使用 `workspace_data/runtime/app/bin/python` 启动 |
| 安装后仍使用旧引擎 | 核对配置后使用 `--activate`，或在 Web 选择 VoxCPM 和 MuseTalk 本地并保存 |
| Web 访问不到 | 检查启动进程、`--host 0.0.0.0`、端口、防火墙与安全组 |
| 配置保存后未生效 | 检查 `.env` 和进程环境覆盖，调整后重启 |
| 首次字幕识别等待较久 | 确认 Whisper 权重下载网络正常 |
| 生成时显存不足 | 停止其他占用显存的任务，先串行生成短视频 |
| 提示已有安装任务 | 等待当前任务结束，查看日志；不要删除锁文件绕过互斥 |
