"""Download MuseTalk weights through the selected Hugging Face endpoint."""
import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from krvoiceai.core.downloads import download_environment

# Hub constants are read at import time; configure before importing the library.
environment = download_environment()
os.environ.update(environment)
from huggingface_hub import hf_hub_download, snapshot_download

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=Path("/opt/MuseTalk/models"))
root = parser.parse_args().root.resolve()
options = {"endpoint": environment.get("HF_ENDPOINT", "https://huggingface.co")}
models = [
    ("TMElyralab/MuseTalk", "", ["musetalkV15/musetalk.json", "musetalkV15/unet.pth"], None),
    ("stabilityai/sd-vae-ft-mse", "sd-vae", ["config.json", "diffusion_pytorch_model.bin"], None),
    ("openai/whisper-tiny", "whisper",
     ["config.json", "pytorch_model.bin", "preprocessor_config.json"], None),
    ("yzd-v/DWPose", "dwpose", ["dw-ll_ucoco_384.pth"], None),
    # Same face-parsing assets, pinned to a known Hub snapshot instead of Google Drive.
    ("camenduru/MuseTalk", "", ["face-parse-bisent/79999_iter.pth",
     "face-parse-bisent/resnet18-5c106cde.pth"], "5cfe77e50e1dc10ac599e5340d610d69a5086af2"),
]
for repo, folder, files, revision in models:
    destination = root / folder
    snapshot_download(repo, local_dir=str(destination), allow_patterns=files,
                      revision=revision, **options)
    for name in files:
        path = destination / name
        minimum = 1024 * 1024 if name.endswith(".pth") else 0
        if not path.is_file() or path.stat().st_size <= minimum:
            raise RuntimeError(f"Missing or incomplete model: {path}")

# This is the upstream s3fd-619a316812.pth, hosted on the Hub as well.
sfd = root / "face-detection/s3fd.pth"
sfd.parent.mkdir(parents=True, exist_ok=True)
if not sfd.is_file() or sfd.stat().st_size < 1024 * 1024:
    cached = hf_hub_download(
        "vinthony/SadTalker", "hub/checkpoints/s3fd-619a316812.pth",
        revision="4aedd064359e623398a2d73eb8c253ebb2bd516c", **options,
    )
    temporary = sfd.with_suffix(".pth.part")
    shutil.copyfile(cached, temporary)
    if temporary.stat().st_size < 1024 * 1024:
        raise RuntimeError(f"Invalid model download: {temporary}")
    temporary.replace(sfd)
detector = root.parent / "musetalk/utils/face_detection/detection/sfd/s3fd.pth"
detector.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(sfd, detector)
