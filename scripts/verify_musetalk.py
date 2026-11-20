"""Load the actual MuseTalk model stack; optionally require working CUDA."""
import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    import torch
    if args.cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，请检查 NVIDIA 驱动和容器 GPU 透传。")
        # Test kernels before importing modules that load heavy model weights.
        torch.zeros(1, device="cuda").add_(1)
    from mmcv.ops import nms
    from musetalk.utils.utils import load_all_model
    from musetalk.utils.audio_processor import AudioProcessor
    from musetalk.utils.face_parsing import FaceParsing
    from musetalk.utils.preprocessing import get_landmark_and_bbox
    from transformers import WhisperModel
    # These constructors load weights; import-only checks miss incomplete installations.
    load_all_model(
        unet_model_path=str(repo / "models/musetalkV15/unet.pth"),
        unet_config=str(repo / "models/musetalkV15/musetalk.json"),
        device="cuda" if args.cuda else "cpu",
    )
    AudioProcessor(feature_extractor_path=str(repo / "models/whisper"))
    WhisperModel.from_pretrained(str(repo / "models/whisper"))
    FaceParsing()
    print("MuseTalk 依赖与权重加载验证通过")


if __name__ == "__main__":
    main()
