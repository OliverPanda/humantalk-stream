"""Download inference weights into the image; fail on incomplete downloads."""
import argparse
import shutil
from pathlib import Path

import gdown
from huggingface_hub import snapshot_download
import requests

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=Path("/opt/MuseTalk/models"))
root = parser.parse_args().root.resolve()
models = [
    ("TMElyralab/MuseTalk", "", ["musetalkV15/musetalk.json", "musetalkV15/unet.pth"]),
    ("stabilityai/sd-vae-ft-mse", "sd-vae", ["config.json", "diffusion_pytorch_model.bin"]),
    ("openai/whisper-tiny", "whisper",
     ["config.json", "pytorch_model.bin", "preprocessor_config.json"]),
    ("yzd-v/DWPose", "dwpose", ["dw-ll_ucoco_384.pth"]),
]
for repo, folder, files in models:
    destination = root / folder
    snapshot_download(repo, local_dir=str(destination), allow_patterns=files)
    for name in files:
        path = destination / name
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f"Missing model: {path}")

face_dir = root / "face-parse-bisent"
face_dir.mkdir(parents=True, exist_ok=True)
face_model = face_dir / "79999_iter.pth"
if not face_model.is_file() or face_model.stat().st_size < 1024 * 1024:
    temporary = face_model.with_suffix(".pth.part")
    gdown.download(id="154JgKpzCPW82qINcVieuPH3fZ2e0P812", output=str(temporary),
                   quiet=False, resume=True)
    if not temporary.is_file() or temporary.stat().st_size < 1024 * 1024:
        raise RuntimeError(f"Invalid model download: {temporary}")
    temporary.replace(face_model)
resnet = face_dir / "resnet18-5c106cde.pth"
if not resnet.is_file() or resnet.stat().st_size < 1024 * 1024:
    temporary = resnet.with_suffix(".pth.part")
    with requests.get("https://download.pytorch.org/models/resnet18-5c106cde.pth",
                      stream=True, timeout=120) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                stream.write(chunk)
    if temporary.stat().st_size < 1024 * 1024:
        raise RuntimeError(f"Invalid model download: {temporary}")
    temporary.replace(resnet)
for path in (face_model, resnet):
    if not path.is_file() or path.stat().st_size < 1024 * 1024:
        raise RuntimeError(f"Invalid model download: {path}")

# Upstream SFDDetector otherwise downloads this on the first inference.
sfd = root / "face-detection/s3fd.pth"
sfd.parent.mkdir(parents=True, exist_ok=True)
if not sfd.is_file() or sfd.stat().st_size < 1024 * 1024:
    temporary = sfd.with_suffix(".pth.part")
    with requests.get(
        "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
        stream=True, timeout=120,
    ) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(1024 * 1024):
                stream.write(chunk)
    if temporary.stat().st_size < 1024 * 1024:
        raise RuntimeError(f"Invalid model download: {temporary}")
    temporary.replace(sfd)
detector = root.parent / "musetalk/utils/face_detection/detection/sfd/s3fd.pth"
detector.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(sfd, detector)
