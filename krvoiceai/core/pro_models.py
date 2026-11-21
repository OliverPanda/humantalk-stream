"""固定 Pro 权重清单；下载使用 ModelScope SDK，推理阶段只检查本地文件。"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

MANIFEST = json.loads(Path(__file__).with_suffix(".json").read_text(encoding="utf-8"))
MODEL_NAMES = ("voxcpm", "musetalk", "whisper")


def model_directory(runtime: Path, model: str) -> Path:
    return {
        "voxcpm": runtime / "models/VoxCPM-0.5B",
        "musetalk": runtime / "MuseTalk/models",
        "whisper": runtime / "models/faster-whisper-small",
    }[model]


def _matches(path: Path, item: dict, *, checksum: bool) -> bool:
    if not path.is_file() or path.stat().st_size != item["size"]:
        return False
    if not checksum:
        return True
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == item["sha256"]


def model_status(model: str, directory: Path, *, checksum: bool = False) -> dict:
    directory = Path(directory).resolve()
    missing = [item["path"] for item in MANIFEST[model]
               if not _matches(directory / item["path"], item, checksum=checksum)]
    return {"ready": not missing, "missing": missing,
            "bytes": sum(item["size"] for item in MANIFEST[model])}


def require_model(model: str, directory: Path) -> Path:
    directory = Path(directory).resolve()
    status = model_status(model, directory)
    if not status["ready"]:
        raise RuntimeError(
            f"{model} 本地模型缺失或文件不完整: {directory / status['missing'][0]}。"
            f"请先用主程序 Python 执行 scripts/download_pro_models.py --models {model}，"
            "或重新运行 Pro 安装器；推理阶段不会在线补下载。"
        )
    return directory


def require_musetalk_assets(repo: Path, *, checksum: bool = False) -> None:
    repo = Path(repo).resolve()
    require_model("musetalk", repo / "models")
    if checksum and not model_status("musetalk", repo / "models", checksum=True)["ready"]:
        raise RuntimeError(f"MuseTalk 本地模型 SHA256 校验失败: {repo / 'models'}")
    item = next(f for f in MANIFEST["musetalk"] if f["path"] == "face-detection/s3fd.pth")
    detector = repo / "musetalk/utils/face_detection/detection/sfd/s3fd.pth"
    if not _matches(detector, item, checksum=checksum):
        raise RuntimeError(
            f"MuseTalk 本地 S3FD 检测权重缺失或不完整: {detector}。"
            "请用主程序 Python 执行 scripts/download_pro_models.py --models musetalk，"
            "不要让官方脚本自行联网下载。"
        )


def download_model(model: str, directory: Path) -> None:
    """复用已校验的文件；下载失败保留原目标，校验通过后原子替换。"""
    from filelock import FileLock
    from modelscope.hub.file_download import model_file_download

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / ".download.lock"), timeout=0):
        for item in MANIFEST[model]:
            target = directory / item["path"]
            if _matches(target, item, checksum=True):
                print(f"[复用] {model}/{item['path']}", flush=True)
                continue
            staging = directory / ".modelscope" / item["repo"].replace("/", "--")
            cached = staging / item["source"]
            # SDK 的 local_dir 可能保留损坏文件，先清掉已知损坏的这一项。
            if cached.is_file() and not _matches(cached, item, checksum=True):
                cached.unlink()
            print(f"[ModelScope] {item['repo']}/{item['source']}", flush=True)
            downloaded = Path(model_file_download(
                item["repo"], item["source"], revision=item["revision"],
                local_dir=str(staging),
            ))
            if not _matches(downloaded, item, checksum=True):
                raise RuntimeError(f"模型校验失败（大小/SHA256）: {downloaded}")
            target.parent.mkdir(parents=True, exist_ok=True)
            downloaded.replace(target)
        require_model(model, directory)
        if model == "musetalk":
            # 官方 S3FD 会在这个源码相对位置查找权重；缺失会触发外网下载。
            detector = directory.parent / "musetalk/utils/face_detection/detection/sfd/s3fd.pth"
            detector.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(directory / "face-detection/s3fd.pth", detector)
    print(f"[完成] {model}: {directory}", flush=True)
