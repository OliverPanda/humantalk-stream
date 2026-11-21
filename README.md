# FreeVideoAI 口播工坊

输入文案，结合自己的声音与真人参考视频，生成带字幕、背景音乐和封面的口播成片。

当前只维护一套 **Pro** 链路：

| 阶段 | 固定选型 | 运行方式 |
|---|---|---|
| 文案润色、标题 | DeepSeek Chat | API，需要密钥 |
| 配音与声音克隆 | VoxCPM-0.5B | 本地 GPU |
| 数字人演绎 | MuseTalk 1.5 | 本地 GPU |
| 参考音频转写、字幕时间轴 | Whisper small / faster-whisper 1.2.1 | 本地 CPU int8 |
| 视频、字幕、BGM 合成 | FFmpeg | 本地 |
| 封面 | Pillow + 中文字体 | 本地 |
| 发布清单 | 内置 Publisher | 本地生成，手动发布 |

全部模型权重（含 MuseTalk 的辅助模型）由 **ModelScope SDK** 下载并校验大小和 SHA256。
安装器先安装 `modelscope==1.34.0`，再下载模型；推理时只读取本地文件。
缺少模型或推理失败会报告原始错误，不切换到云端配音、其他数字人引擎或占位输出。

## 安装与启动

优先支持 Linux x86_64。第一张 NVIDIA 显卡总显存至少 18GB，建议 24GB；
驱动需支持 VoxCPM 的 cu121 与 MuseTalk 的 cu118。建议数据盘至少保留 40GiB 可用空间。

在项目根目录执行：

```bash
sudo bash scripts/install_system_deps.sh
bash install.sh --profile pro --install --activate
workspace_data/runtime/app/bin/python start.py --host 0.0.0.0 --port 8000
```

使用 root 账户时可省略 `sudo`。首次安装需联网下载软件包、源码和约 6.13GiB 模型权重。
Python 包与源码使用 `config/install-mirrors.env` 中的下载源；模型文件统一从 ModelScope 下载。
重复执行会复用校验通过的模型，修复缺失或损坏的文件。安装器实际加载三个模型验证成功后，
才应用固定 Pro 配置，保留已有 DeepSeek 密钥、素材路径和用户资产；旧配置备份为 `config/user_config.yaml.bak`。

打开 `http://服务器IP:8000`：

1. 在“模型配置 → LLM”填写 DeepSeek API Key 并测试连接。
2. 在“我的声音”上传 5～20 秒干净人声，注册音色。
3. 在“我的形象”上传清晰、正面、单人的参考视频，注册形象。
4. 选择自己的声音与形象，先用 5～10 秒短文案验证。

关闭服务时在原终端按 `Ctrl+C`；重新执行上述启动命令即可重启。
启动终端会显示生效的引擎、模型路径、独立环境和缺失项，DeepSeek Key 只显示是否设置。
单独运行 `scripts/download_pro_models.py` 也会在校验成功后自动保存对应模型路径及引擎配置。
部署后可执行 `workspace_data/runtime/app/bin/python scripts/probe_pro.py` 做 Pro 探针，
结果保存在 `workspace_data/probes/`；加 `--check-api --face /实际/真人素材.mp4` 可验证 API 和数字人出片。

完整模型目录、模型单独下载、Windows 命令与 avatar 排查见 [Pro 部署指南](docs/DEPLOYMENT.md)。
