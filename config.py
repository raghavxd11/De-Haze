from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    # ── Image dimensions ─────────────────────────────────────────────────────
    image_size: int = 256
    target_size: tuple[int, int] = (256, 256)
    input_shape: tuple[int, int, int] = (256, 256, 3)

    # ── Training parameters ──────────────────────────────────────────────────
    batch_size: int = 8
    unet_batch_size: int = 3
    pix2pix_batch_size: int = 3
    unet_epochs: int = 60
    pix2pix_epochs: int = 15
    seed: int = 42

    # ── Preprocessing ────────────────────────────────────────────────────────
    # Supported modes: "none" | "clahe" | "clahe_hls"
    # clahe_hls = current behavior (CLAHE on L in LAB + HLS L-40/S+30 shift)
    # clahe     = CLAHE on L in LAB only
    # none      = raw image (no preprocessing)
    preprocess_mode: str = "clahe_hls"

    # ── Directory paths (relative to REPO_ROOT) ──────────────────────────────
    dataset_dir: Path = REPO_ROOT / "dataset"
    hazy_full_dir: Path = REPO_ROOT / "dataset" / "Hazy_Full"
    gt_full_dir: Path = REPO_ROOT / "dataset" / "GT_Full"
    hazy_small_dir: Path = REPO_ROOT / "dataset" / "Hazy_Small"
    gt_small_dir: Path = REPO_ROOT / "dataset" / "GT_Small"

    models_dir: Path = REPO_ROOT / "models"
    pix2pix_model_dir: Path = REPO_ROOT / "models" / "pix2pix"
    unet_model_dir: Path = REPO_ROOT / "models" / "U-Net"
    outputs_dir: Path = REPO_ROOT / "outputs"
    cache_dir: Path = REPO_ROOT / "outputs" / "cache"
    train_output_dir: Path = REPO_ROOT / "outputs" / "models"
    infer_output_dir: Path = REPO_ROOT / "outputs"
    sample_test_dir: Path = REPO_ROOT / "sample_test_files"

    # ── CLI & Inference defaults ─────────────────────────────────────────────
    default_model: str = "unet"
    default_detect_on: str = "dehazed"

    # ── Split settings (CG2) ─────────────────────────────────────────────────
    split_file: Path = REPO_ROOT / "splits" / "split_v1.json"

    # Scene grouping (MobileNetV2 cosine + AgglomerativeClustering)
    embedding_model: str = "MobileNetV2"          # logged in split_v1.json
    embedding_weights: str = "imagenet"
    grouping_linkage: str = "average"
    # Threshold 0.20 chosen after leakage sweep (0.15 left ~58% < 0.14; 0.20 has ~10% combined, ~4% in test)
    grouping_distance_threshold: float = 0.20
    grouping_embed_batch: int = 32

    # Split ratios and caps
    split_val_fraction: float = 0.10
    split_test_fraction: float = 0.10
    split_max_group_size_val_test: int = 30       # groups larger than this go to train
    split_seed: int = 42                           # = config.seed

    # Checkpoint settings (CG5)
    checkpoint_dir: Path = REPO_ROOT / "outputs" / "checkpoints"
    checkpoint_every_n_epochs: int = 5


config = Config()
