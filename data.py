import hashlib
import json

import cv2
import numpy as np
import tensorflow as tf

import config
from preprocess import prepare_input

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def load_pairs(pairs, mode, name, cache_dir=config.CACHE_DIR):
    """Load (hazy, gt) image pairs as uint8 BGR arrays of shape (N, 256, 256, 3).

    Only hazy images are preprocessed. Results are cached as .npz in outputs/cache,
    keyed by the file names, image size and preprocess mode.
    """
    w, h = config.IMG_SIZE
    if not pairs:
        empty = np.empty((0, h, w, 3), dtype=np.uint8)
        return empty, empty

    key = hashlib.sha256(f"{w}x{h}_{mode}".encode())
    for hazy_path, gt_path in pairs:
        key.update(hazy_path.name.encode())
        key.update(gt_path.name.encode())
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{name}_{key.hexdigest()[:16]}_{w}x{h}_{mode}.npz"

    if cache_file.exists():
        print(f"[Data] Loading {name} ({len(pairs)} pairs) from cache {cache_file.name}")
        with np.load(cache_file) as d:
            return d["hazy"], d["gt"]

    print(f"[Data] Reading {len(pairs)} pairs for {name} (preprocess: {mode})")
    hazy_list, gt_list, bad = [], [], []
    for hazy_path, gt_path in pairs:
        # cv2.imread reads the real format, so JPEGs saved as .png still work
        hazy = cv2.imread(str(hazy_path))
        if hazy is None:
            bad.append(str(hazy_path))
            continue
        gt = cv2.imread(str(gt_path))
        if gt is None:
            bad.append(str(gt_path))
            continue
        hazy_list.append(prepare_input(hazy, mode))
        gt_list.append(cv2.resize(gt, config.IMG_SIZE, interpolation=cv2.INTER_AREA))

    if bad:
        raise RuntimeError(f"Could not read {len(bad)} files:\n" + "\n".join("  - " + f for f in bad))

    hazy_arr = np.array(hazy_list, dtype=np.uint8)
    gt_arr = np.array(gt_list, dtype=np.uint8)
    np.savez_compressed(cache_file, hazy=hazy_arr, gt=gt_arr)
    print(f"[Data] Saved cache {cache_file.name}")
    return hazy_arr, gt_arr


def small_pairs(hazy_dir=config.HAZY_SMALL_DIR, gt_dir=config.GT_SMALL_DIR):
    """The 55 outdoor pairs (IDs 01-55), split 44 train / 11 val by file name."""
    gt_names = {p.name for p in gt_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    pairs = []
    for hazy_path in sorted(hazy_dir.iterdir()):
        if hazy_path.suffix.lower() not in IMAGE_EXTS:
            continue
        gt_name = hazy_path.name.replace("_hazy.png", "_GT.png").replace("_hazy.jpg", "_GT.jpg")
        if gt_name in gt_names:
            pairs.append((hazy_path, gt_dir / gt_name))
        else:
            print(f"[Data] WARNING: no GT image for {hazy_path.name}")

    n_train = int(len(pairs) * 0.8)
    return pairs[:n_train], pairs[n_train:]


def split_pairs(split_file=config.SPLIT_FILE, hazy_dir=config.HAZY_FULL_DIR, gt_dir=config.GT_FULL_DIR):
    """Train, val and test pairs listed in splits/split_v1.json."""
    if not split_file.exists():
        raise FileNotFoundError(f"{split_file} not found. Run 'python split.py' to create it.")
    splits = json.loads(split_file.read_text(encoding="utf-8"))["splits"]
    print(f"[Data] Split from {split_file.name}: " + ", ".join(f"{k} {len(v)}" for k, v in splits.items()))
    return [[(hazy_dir / item["hazy"], gt_dir / item["gt"]) for item in splits[name]]
            for name in ("train", "val", "test")]


def make_dataset(hazy, gt, batch_size, training=False):
    """tf.data pipeline: shuffle (train only), scale to [0, 1], batch, prefetch."""
    ds = tf.data.Dataset.from_tensor_slices((hazy, gt))
    if training:
        ds = ds.shuffle(min(len(hazy), 1000), seed=config.SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda h, g: (tf.cast(h, tf.float32) / 255.0, tf.cast(g, tf.float32) / 255.0),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def get_datasets(batch_size, mode=config.PREPROCESS_MODE, small=False):
    """Return (train_ds, val_ds). small=True is for quick debug runs only."""
    if small:
        print("[Data] --small: 55 outdoor pairs (44 train / 11 val). Debug only, not for reported numbers.")
        train, val = small_pairs()
        prefix = "small"
    else:
        train, val, _ = split_pairs()
        prefix = "full"

    train_hazy, train_gt = load_pairs(train, mode, f"{prefix}_train")
    val_hazy, val_gt = load_pairs(val, mode, f"{prefix}_val")
    return (make_dataset(train_hazy, train_gt, batch_size, training=True),
            make_dataset(val_hazy, val_gt, batch_size))
