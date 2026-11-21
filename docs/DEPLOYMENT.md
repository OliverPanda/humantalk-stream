# Pro 部署与排查

固定链路：DeepSeek Chat API → 本地 VoxCPM-0.5B → 本地 MuseTalk 1.5 → 本地 Whisper small → FFmpeg。
DeepSeek API 是唯一需要配置的模型服务；数字人无需启动 8010 端口的额外 API。
不使用 LatentSync、EchoMimic、Wav2Lip、Edge TTS 或云端视频生成。

## 环境与安装

Linux x86_64，第一张 NVIDIA 显卡总显存至少 18GB，多卡显存不相加。
驱动须兼容 PyTorch cu121 / cu118；建议 40GiB 以上可用磁盘。
安装和首次取得源码、权重需要联网。视频生成期间仅文案与标题调用 DeepSeek API。

```bash
cd /root/gpufree-data/freevideoai
nvidia-smi
df -h .
sudo bash scripts/install_system_deps.sh
bash install.sh --profile pro --install --activate
```

root 账户可省略 `sudo`。安装器依次安装主程序、VoxCPM、MuseTalk、Whisper，
先执行 `pip install modelscope==1.34.0`，全部权重通过该 SDK 下载。
源码仍通过 Git 获取，Python 依赖仍通过 pip/uv 安装，这些不是模型权重。

三个模型都加载验证通过后才写 `config/user_config.yaml`；Pro 会应用固定模型选型及路径，
保留已有 DeepSeek Key 和用户素材目录。旧配置备份为 `config/user_config.yaml.bak`。
失败后修复提示的问题，重跑同一命令；无需删除 runtime 或清空已下载文件。

## 模型与目录

全部路径以项目根目录为基准，配置支持相对路径。

| 模型组 | ModelScope 仓库 | 本地目录 |
|---|---|---|
| VoxCPM-0.5B | `OpenBMB/VoxCPM-0.5B` | `workspace_data/runtime/models/VoxCPM-0.5B` |
| MuseTalk 1.5 | `AI-ModelScope/MuseTalk` | `workspace_data/runtime/MuseTalk/models/musetalkV15` |
| SD VAE | `stabilityai/sd-vae-ft-mse` | `…/MuseTalk/models/sd-vae` |
| MuseTalk 音频编码器 | `openai-mirror/whisper-tiny` | `…/MuseTalk/models/whisper` |
| DWPose | `zhangjin/DWPose` | `…/MuseTalk/models/dwpose` |
| 人脸解析、ResNet18 | `Kedreamix/Linly-Talker`（仅指定文件） | `…/MuseTalk/models/face-parse-bisent` |
| S3FD 人脸检测 | `HaveAnApplePie/s3fd-619a316812` | `…/MuseTalk/models/face-detection`，同步到官方源码要求的位置 |
| Whisper small 多语言 | `Systran/faster-whisper-small` | `workspace_data/runtime/models/faster-whisper-small` |

完整下载约 6.13GiB；确切文件、大小与 SHA256 在 `krvoiceai/core/pro_models.json`。
同名仓库的文件更新若与清单不一致，校验会失败，避免混用版本。
MuseTalk 的 Whisper tiny 为 Transformers 格式；字幕模型为 CTranslate2 格式，不能互换。
Pro 关闭额外 VAD 模型，只使用这份本地 Whisper small 完成识别与词级时间轴。

补齐模型并自动配置（主程序环境已安装）：

```bash
workspace_data/runtime/app/bin/python scripts/download_pro_models.py
# 也可以只补一组
workspace_data/runtime/app/bin/python scripts/download_pro_models.py --models musetalk
```

下载器在所有选中模型下载并校验成功后，将实际目录及对应引擎写入
`config/user_config.yaml`，保留 DeepSeek Key、兼容服务地址和素材目录；
单独下载一组时只更新该阶段和 DeepSeek 选型。已有配置会备份，下载失败不改配置。

自定义运行目录使用 `--runtime /实际/runtime`；自定义单组模型目录使用
`--destination /实际/模型目录`，无需再手动修改模型路径。
MuseTalk 目标目录必须为 `/实际/MuseTalk源码目录/models`。
这些相对路径都以项目根目录为基准。自定义 runtime 时，对应的 VoxCPM/MuseTalk
独立环境也应位于该 runtime 下；下载器只下载模型，缺少源码或依赖时仍需运行安装器。
仅下载而不写配置可加 `--no-config`；完整安装器使用此模式，加载验证通过后才统一激活配置。
下载器校验通过才替换目标文件，不删除用户已有模型或素材。

## 配置与启动

安装器会写入实际路径。核心配置如下，合并到已有节点，不要重复创建同名 YAML 节点：

```yaml
llm:
  provider: deepseek
  base_url: https://api.deepseek.com/v1
  model: deepseek-chat
  api_key: "填写密钥，或使用网页保存"
tts:
  provider: voxcpm
  fallback_chain: [voxcpm]
avatar:
  provider: musetalk_local
  musetalk_local:
    repo_dir: ./workspace_data/runtime/MuseTalk
    env_python: ./workspace_data/runtime/musetalk/bin/python
    worker_script: ./scripts/musetalk_worker.py
    version: v15
    batch_size: 4
    use_float16: true
    timeout: 1800
asr:
  provider: whisper_local
  whisper:
    model_size: small
    model_dir: ./workspace_data/runtime/models/faster-whisper-small
    device: cpu
    compute_type: int8
    vad_filter: false
```

环境变量（包括 `.env` 中的 `KRVOICEAI_*`）优先于 YAML；旧的 provider、模型路径变量会覆盖新配置，
请移除冲突项后重启。安装器会在应用配置前检查冲突，并保留下载进度；不会删除或改写 `.env`。

```bash
workspace_data/runtime/app/bin/python start.py --host 0.0.0.0 --port 8000
```

保持终端运行，访问 `http://服务器IP:8000`。重启时先在旧终端按 `Ctrl+C`，再执行同一命令。
启动终端会打印生效的引擎、MuseTalk 版本、模型绝对路径、独立 Python/worker、FFmpeg、
缺失项及修复命令；DeepSeek Key 只显示是否设置。启动检查仅检查文件和大小，
实际 CUDA/模型加载仍以上面的安装验证或下面的 `verify_musetalk.py --cuda` 为准。
下载脚本更新配置后，需要重启正在运行的服务。

## Pro 探针

在部署的那台云电脑上执行：

```bash
workspace_data/runtime/app/bin/python scripts/probe_pro.py
# 进一步验证 DeepSeek 请求和真人参考素材驱动出片
workspace_data/runtime/app/bin/python scripts/probe_pro.py \
  --check-api --face /实际/真人参考视频.mp4 --timeout 600
```

探针读取生效配置（含 `.env` 覆盖），检查 Pro 选型、第一张显卡、模型大小及 SHA256、
独立 Python 与源码；复用现有 worker 合成短音频，加载 MuseTalk 模型栈，
运行 Whisper 推理，并实际生成一秒 H.264/AAC 视频后用 ffprobe 检查。
本地模型子进程禁止网络连接，不会下载、修复模型或修改配置。

每次结果保存在 `workspace_data/probes/时间-随机后缀/`：
`report.json` 标记每项 PASS / FAIL / SKIP，执行过的子进程保留 `.log`。
失败返回退出码 1，并继续收集其他阶段问题；密钥不会写入终端或报告。
可用 `--output /新的空目录` 指定保存位置，`--config /用户配置.yaml` 指定配置文件。

默认只检查 DeepSeek Key 是否已设置，不调用 API；默认 MuseTalk 只验证 CUDA 和模型加载，
不代表真人出片通过。`--check-api` 会发出一次短 Chat 请求；
提供 `--face` 才会使用探针生成的配音验证真实数字人视频。
缺少配音时 Whisper 使用一秒静音做运行检查，报告会注明未验证真实语音识别。
探针通过也不代表长视频性能、声音相似度或口型质量已达标，仍需短任务人工确认。

## avatar 失败时

先检查模型与 CUDA 是否能加载：

```bash
workspace_data/runtime/musetalk/bin/python scripts/verify_musetalk.py --repo workspace_data/runtime/MuseTalk --cuda
```

必须看到“依赖与权重加载验证通过”。网页的本地引擎检查也会实际检查环境，不再仅返回“无需连接测试”。
选择已上传的真实形象，并确认任务有配音文件。失败原因会保存在任务详情和主日志；
推理的完整 stdout/stderr 写入该任务的 `musetalk.log`，超时也保留日志。

```bash
tail -n 100 workspace_data/logs/krvoiceai.log
# 用任务详情中的实际路径查看对应任务日志
tail -n 100 workspace_data/jobs/你的任务ID/musetalk.log
```

优先处理日志中第一处 CUDA、依赖、缺失权重或参考素材错误。
不要为 MuseTalk 故障额外安装其他数字人引擎。下载完成但文件无法加载时，重跑模型下载命令修复后再验证。

## Windows

安装 Git、可用的 NVIDIA 驱动、FFmpeg 和中文字体后，在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1 --profile pro --install --activate
.\workspace_data\runtime\app\Scripts\python.exe start.py --host 127.0.0.1 --port 8000
```

Windows 的独立环境 Python 使用 `Scripts/python.exe`，Linux 使用 `bin/python`。
Linux 云电脑是当前主要验证目标；真实 GPU 推理需在目标机器运行上面的加载检查和短任务验证。
