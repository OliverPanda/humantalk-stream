"""Shared pinned MuseTalk environment recipe for native Linux/Windows and Docker."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()

    def pip(*packages):
        subprocess.run([sys.executable, "-m", "pip", "install", *packages], check=True)

    pip("torch==2.0.1", "torchvision==0.15.2", "torchaudio==2.0.2",
        "--index-url", "https://download.pytorch.org/whl/cu118")
    pip("-r", str(args.repo / "requirements.txt"))
    pip("mmengine==0.10.7")
    pip("--only-binary=mmcv", "mmcv==2.0.1",
        "-f", "https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html")
    pip("--no-build-isolation", "chumpy==0.70")
    pip("mmdet==3.1.0", "mmpose==1.1.0")


if __name__ == "__main__":
    main()
