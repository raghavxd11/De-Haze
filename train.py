"""
train.py -- Unified command-line training entry point for De-Haze.

Usage:
    python train.py --model unet|pix2pix --epochs N --batch-size B
                    [--small] [--resume] [--preprocess MODE]
                    [--rebuild-split]
"""

import argparse
import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import config
from data import get_dehaze_datasets
from splits.split_builder import build_split
from unet_train import train_unet
from pix2pix_train import train_pix2pix


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train De-Haze models (U-Net or Pix2Pix) with tf.data datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["unet", "pix2pix"],
        default=config.default_model,
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: config.unet_epochs or config.pix2pix_epochs).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Mini-batch size (default: config.unet_batch_size or config.pix2pix_batch_size).",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Use the 55-pair outdoor subset (IDs 01-55) for fast smoke testing / debugging.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint (Pix2Pix).",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        choices=["clahe_hls", "clahe", "none"],
        default=config.preprocess_mode,
        help="Preprocessing mode for hazy images.",
    )
    parser.add_argument(
        "--rebuild-split",
        action="store_true",
        help="Force rebuild of splits/split_v1.json before training.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save the trained model files (default: outputs/models/<model>).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print(f"DE-HAZE TRAINING: {args.model.upper()}")
    print("=" * 70)

    # 1. Rebuild split if requested
    if args.rebuild_split:
        print("[Train] Rebuilding split_v1.json as requested by --rebuild-split...")
        build_split(
            hazy_dir=config.hazy_full_dir,
            gt_dir=config.gt_full_dir,
            rebuild=True,
        )

    # 2. Resolve hyperparameters
    if args.model == "unet":
        epochs = args.epochs if args.epochs is not None else config.unet_epochs
        batch_size = args.batch_size if args.batch_size is not None else config.unet_batch_size
    else:
        epochs = args.epochs if args.epochs is not None else config.pix2pix_epochs
        batch_size = args.batch_size if args.batch_size is not None else config.pix2pix_batch_size

    # Output directory defaults to outputs/models/<model> (gitignored)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = config.train_output_dir / args.model

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Model:           {args.model}")
    print(f"  Epochs:          {epochs}")
    print(f"  Batch size:      {batch_size}")
    print(f"  Preprocess mode: {args.preprocess}")
    print(f"  Small subset:    {args.small}")
    print(f"  Resume:          {args.resume}")
    print(f"  Output dir:      {output_dir}")
    print("=" * 70)

    # 3. Load datasets
    train_ds, val_ds = get_dehaze_datasets(
        batch_size=batch_size,
        preprocess_mode=args.preprocess,
        small=args.small,
    )

    # 4. Train selected model
    if args.model == "unet":
        if args.resume:
            print("[Train Note] --resume specified for U-Net; training fresh architecture for requested epochs.")
        train_unet(
            train_data=train_ds,
            val_data=val_ds,
            output_dir=output_dir,
            num_epochs=epochs,
            batch_size=batch_size,
            model_name="unet",
            preprocess_mode=args.preprocess,
        )
    elif args.model == "pix2pix":
        train_pix2pix(
            train_dataset=train_ds,
            val_dataset=val_ds,
            output_dir=output_dir,
            num_epochs=epochs,
            batch_size=batch_size,
            model_name="p2p",
            resume=args.resume,
            preprocess_mode=args.preprocess,
        )
    else:
        raise ValueError(f"Unknown model type '{args.model}'")

    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE: Artifacts saved to {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
