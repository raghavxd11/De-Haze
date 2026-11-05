"""
main.py -- Convenience wrapper and usage guide for De-Haze CLI entry points.

Replaces the legacy behavior of retraining on every run with a clean,
non-retraining usage overview directing users to train.py and infer.py.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import print_device_info


def print_usage():
    print("=" * 70)
    print("DE-HAZE: Deep Learning Image/Video Dehazing & Human Detection")
    print("=" * 70)
    print_device_info()
    print("\n[CLI Usage Overview]")
    print("\n1. Model Training (train.py):")
    print("   python train.py --model unet|pix2pix --epochs N --batch-size B")
    print("                   [--small] [--resume] [--preprocess MODE] [--rebuild-split]")
    print("   Examples:")
    print("     python train.py --model unet --small --epochs 1 --batch-size 4")
    print("     python train.py --model pix2pix --epochs 15 --batch-size 4 --preprocess clahe_hls")
    print("\n2. Model Inference (infer.py):")
    print("   python infer.py --model-path PATH --input IMG_OR_VIDEO --output outputs/")
    print("                   [--no-display] [--no-detect] [--detect-on dehazed|hazy] [--preprocess MODE]")
    print("   Examples:")
    print("     python infer.py --model-path outputs/models/unet/unet.keras --input sample_test_files/hazed_img.jpg --no-display")
    print("     python infer.py --model-path outputs/models/unet/unet.keras --input sample_test_files/test_vid.mp4 --detect-on dehazed --no-display")
    print("\nFor full argument details, run:")
    print("   python train.py --help")
    print("   python infer.py --help")
    print("=" * 70)


if __name__ == "__main__":
    print_usage()
