"""
data.py — Efficient data loading and tf.data dataset pipeline for De-Haze.

Key design principles (Phase 3 Change Group 2):
1. cv2.imread loads all images in native BGR format (handles JPEGs disguised as .png).
2. Images are resized directly to target size (256x256) as uint8 using cv2.INTER_AREA;
   full-resolution originals are never kept in memory.
3. Preprocessing (preprocess_mode: "clahe_hls" | "clahe" | "none") is applied ONLY to
   hazy images at load time on uint8 BGR. Ground truth is NEVER preprocessed.
4. Resized uint8 arrays are cached on disk under outputs/cache/ (gitignored),
   keyed by file-list hash + image dimensions + preprocess mode.
5. Undecodable files are logged with names and count, stopping with a clear RuntimeError.
6. tf.data.Dataset pipeline applies shuffle, batch, and prefetch; no sklearn train_test_split.
7. Supports full dataset (via splits/split_v1.json) and --small subset (IDs 01-55, outdoor only,
   deterministic 80/20 split, with explicit debug/smoke-test logging).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Sequence, Tuple

import cv2
import numpy as np
import tensorflow as tf

from config import config
from preprocess import preprocess, prepare_input

logger = logging.getLogger(__name__)


def _compute_cache_key(
    pairs: Sequence[Tuple[Path, Path]],
    target_size: Tuple[int, int],
    preprocess_mode: str,
) -> str:
    """Deterministic hash based on pair filenames, target dimensions, and preprocess mode."""
    h = hashlib.sha256()
    h.update(f"{target_size[0]}x{target_size[1]}_{preprocess_mode}".encode("utf-8"))
    for h_path, g_path in pairs:
        h.update(h_path.name.encode("utf-8"))
        h.update(g_path.name.encode("utf-8"))
    return h.hexdigest()[:16]


def load_and_cache_pairs(
    pairs: Sequence[Tuple[Path, Path]],
    target_size: Tuple[int, int] = config.target_size,
    preprocess_mode: str = config.preprocess_mode,
    cache_dir: Path = config.cache_dir,
    split_name: str = "data",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load, preprocess hazy images, resize directly to uint8, and cache on disk.

    Args:
        pairs: List of (hazy_path, gt_path) tuples.
        target_size: (height, width) tuple, defaults to (256, 256).
        preprocess_mode: "none" | "clahe" | "clahe_hls".
        cache_dir: Path to directory for cache files.
        split_name: Name prefix for cache identification.

    Returns:
        (hazy_array, gt_array) as uint8 BGR numpy arrays of shape (N, H, W, 3).

    Raises:
        RuntimeError: If any image cannot be decoded by cv2.imread.
    """
    if not pairs:
        empty = np.empty((0, target_size[0], target_size[1], 3), dtype=np.uint8)
        return empty, empty

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_key = _compute_cache_key(pairs, target_size, preprocess_mode)
    cache_file = cache_dir / f"{split_name}_{cache_key}_{target_size[0]}x{target_size[1]}_{preprocess_mode}.npz"

    if cache_file.exists():
        print(f"[Data Cache] Loading cached {split_name} ({len(pairs)} pairs) from {cache_file.name} …")
        with np.load(str(cache_file)) as data:
            return data["hazy"], data["gt"]

    print(f"[Data] Decoding, preprocessing ({preprocess_mode}), and caching {len(pairs)} pairs for {split_name} …")

    hazy_list = []
    gt_list = []
    undecodable = []

    for h_path, g_path in pairs:
        # Load hazy image in native BGR
        h_img = cv2.imread(str(h_path))
        if h_img is None:
            undecodable.append(str(h_path))
            continue

        # Prepare hazy image: preprocess at original resolution then resize to uint8
        h_resized = prepare_input(h_img, mode=preprocess_mode, target_size=target_size)

        # Load ground truth image in native BGR (never preprocessed!)
        g_img = cv2.imread(str(g_path))
        if g_img is None:
            undecodable.append(str(g_path))
            continue

        # Direct stretch resize to uint8 (cv2.INTER_AREA)
        g_resized = cv2.resize(g_img, target_size, interpolation=cv2.INTER_AREA)

        hazy_list.append(h_resized)
        gt_list.append(g_resized)

    if undecodable:
        err_msg = (
            f"[FATAL] cv2.imread failed to decode {len(undecodable)} files:\n"
            + "\n".join(f"  - {f}" for f in undecodable)
        )
        print(err_msg)
        raise RuntimeError(err_msg)

    hazy_arr = np.array(hazy_list, dtype=np.uint8)
    gt_arr = np.array(gt_list, dtype=np.uint8)

    # Save to disk cache
    np.savez_compressed(str(cache_file), hazy=hazy_arr, gt=gt_arr)
    print(f"[Data Cache] Saved {len(pairs)} pairs to {cache_file.name}")

    return hazy_arr, gt_arr


def get_small_pairs(
    hazy_dir: Path = config.hazy_small_dir,
    gt_dir: Path = config.gt_small_dir,
) -> Tuple[list[Tuple[Path, Path]], list[Tuple[Path, Path]]]:
    """Deterministic 80/20 train/val split of the 55 outdoor pairs (IDs 01-55)."""
    hazy_files = {p.name: p for p in sorted(hazy_dir.iterdir())
                  if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    gt_files = {p.name: p for p in sorted(gt_dir.iterdir())
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}

    matched: list[Tuple[Path, Path]] = []
    unmatched_count = 0
    for hname, hpath in sorted(hazy_files.items()):
        gt_name = hname.replace("_hazy.png", "_GT.png").replace("_hazy.jpg", "_GT.jpg")
        if gt_name in gt_files:
            matched.append((hpath, gt_files[gt_name]))
        else:
            print(f"  [WARNING] Unmatched hazy file: {hname}")
            unmatched_count += 1

    if unmatched_count:
        print(f"  [WARNING] Total unmatched small files: {unmatched_count}")

    matched.sort(key=lambda t: t[0].name)

    # Deterministic 80/20 split (44 train / 11 val)
    n_train = int(len(matched) * 0.8)
    train_pairs = matched[:n_train]
    val_pairs = matched[n_train:]
    return train_pairs, val_pairs


def get_split_pairs(
    split_file: Path = config.split_file,
    hazy_dir: Path = config.hazy_full_dir,
    gt_dir: Path = config.gt_full_dir,
) -> Tuple[list[Tuple[Path, Path]], list[Tuple[Path, Path]], list[Tuple[Path, Path]]]:
    """Read train, val, and test pairs defined in splits/split_v1.json."""
    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file '{split_file}' does not exist. "
            f"Run 'python splits/split_builder.py' to generate it."
        )

    with open(split_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    splits = data["splits"]
    train_pairs = [(hazy_dir / item["hazy"], gt_dir / item["gt"]) for item in splits["train"]]
    val_pairs = [(hazy_dir / item["hazy"], gt_dir / item["gt"]) for item in splits["val"]]
    test_pairs = [(hazy_dir / item["hazy"], gt_dir / item["gt"]) for item in splits["test"]]

    return train_pairs, val_pairs, test_pairs


def make_tf_dataset(
    hazy_arr: np.ndarray,
    gt_arr: np.ndarray,
    batch_size: int,
    is_training: bool = False,
    shuffle_buffer: int = 1000,
    seed: int = config.seed,
) -> tf.data.Dataset:
    """Create an efficient tf.data.Dataset from uint8 numpy arrays.

    Normalizes pixel values to [0.0, 1.0] float32 on-the-fly.
    Applies shuffle (if training), batch, and prefetch(AUTOTUNE).
    """
    dataset = tf.data.Dataset.from_tensor_slices((hazy_arr, gt_arr))

    if is_training:
        dataset = dataset.shuffle(
            buffer_size=min(len(hazy_arr), shuffle_buffer),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    # Cast to float32 and normalize [0, 255] -> [0.0, 1.0]
    def _normalize(h_u8, g_u8):
        h_f32 = tf.cast(h_u8, tf.float32) / 255.0
        g_f32 = tf.cast(g_u8, tf.float32) / 255.0
        return h_f32, g_f32

    dataset = dataset.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def get_dehaze_datasets(
    split_file: Path = config.split_file,
    hazy_dir: Path | None = None,
    gt_dir: Path | None = None,
    batch_size: int = config.batch_size,
    preprocess_mode: str = config.preprocess_mode,
    small: bool = False,
    shuffle_buffer: int = 1000,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """High-level function providing train and val tf.data datasets.

    Args:
        split_file: Path to split_v1.json (used when small=False).
        hazy_dir: Directory containing hazy images.
        gt_dir: Directory containing ground truth images.
        batch_size: Batch size for both train and val datasets.
        preprocess_mode: "clahe_hls" (default) | "clahe" | "none".
        small: If True, uses the 55-pair outdoor subset (IDs 01-55).
        shuffle_buffer: Buffer size for training shuffle.

    Returns:
        (train_dataset, val_dataset) as tf.data.Dataset instances.
    """
    if small:
        print("=" * 70)
        print("[Data] Using --small subset (IDs 01-55, outdoor only, 44 train / 11 val).")
        print("[Data] NOTE: This subset is outdoor only and is a debug/smoke-test mode")
        print("[Data] that must NOT be used for formal benchmark numbers.")
        print("=" * 70)
        h_dir = hazy_dir or config.hazy_small_dir
        g_dir = gt_dir or config.gt_small_dir
        train_pairs, val_pairs = get_small_pairs(h_dir, g_dir)
        train_prefix, val_prefix = "small_train", "small_val"
    else:
        h_dir = hazy_dir or config.hazy_full_dir
        g_dir = gt_dir or config.gt_full_dir
        train_pairs, val_pairs, _ = get_split_pairs(split_file, h_dir, g_dir)
        train_prefix, val_prefix = "full_train", "full_val"

    train_hazy, train_gt = load_and_cache_pairs(
        train_pairs,
        target_size=config.target_size,
        preprocess_mode=preprocess_mode,
        cache_dir=config.cache_dir,
        split_name=train_prefix,
    )
    val_hazy, val_gt = load_and_cache_pairs(
        val_pairs,
        target_size=config.target_size,
        preprocess_mode=preprocess_mode,
        cache_dir=config.cache_dir,
        split_name=val_prefix,
    )

    train_ds = make_tf_dataset(
        train_hazy, train_gt,
        batch_size=batch_size,
        is_training=True,
        shuffle_buffer=shuffle_buffer,
    )
    val_ds = make_tf_dataset(
        val_hazy, val_gt,
        batch_size=batch_size,
        is_training=False,
    )
    return train_ds, val_ds


def load_and_split(
    hazy_dir: Path | str = config.hazy_small_dir,
    gt_dir: Path | str = config.gt_small_dir,
    preprocess_mode: str = config.preprocess_mode,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int]]:
    """Deterministic loader returning float32 numpy arrays [0, 1] without sklearn train_test_split.

    Provided for backward compatibility with scripts expecting numpy arrays.
    """
    h_dir = Path(hazy_dir)
    g_dir = Path(gt_dir)
    train_pairs, val_pairs = get_small_pairs(h_dir, g_dir)

    X_train_u8, y_train_u8 = load_and_cache_pairs(
        train_pairs, config.target_size, preprocess_mode, config.cache_dir, "compat_train"
    )
    X_val_u8, y_val_u8 = load_and_cache_pairs(
        val_pairs, config.target_size, preprocess_mode, config.cache_dir, "compat_val"
    )

    X_train = X_train_u8.astype(np.float32) / 255.0
    y_train = y_train_u8.astype(np.float32) / 255.0
    X_val = X_val_u8.astype(np.float32) / 255.0
    y_val = y_val_u8.astype(np.float32) / 255.0

    return X_train, X_val, y_train, y_val, config.input_shape
