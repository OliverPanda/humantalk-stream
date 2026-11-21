"""Shared pinned MuseTalk environment recipe for native Linux/Windows and Docker."""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from krvoiceai.core.downloads import download_environment, torch_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()

    def pip(*packages):
        subprocess.run([sys.executable, "-m", "pip", "install", *packages],
                       check=True, env=download_environment())

    pip("torch==2.0.1", "torchvision==0.15.2", "torchaudio==2.0.2",
        "--index-url", torch_index("cu118"))
    pip("-r", str(args.repo / "requirements.txt"))
    pip("mmengine==0.10.7")
    pip("--only-binary=mmcv", "mmcv==2.0.1",
        "-f", download_environment().get(
            "FREEVIDEOAI_MMCV_INDEX",
            "https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html"))
    pip("--no-build-isolation", "chumpy==0.70")
    pip("mmdet==3.1.0", "mmpose==1.1.0")


if __name__ == "__main__":
    main()
