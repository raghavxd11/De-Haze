"""Build the scene-grouped train/val/test split and save it to splits/split_v1.json.

The dataset has many near-duplicate indoor scenes. A random split would put
copies of the same room in train and test, so similar GT images are grouped
first (MobileNetV2 features + agglomerative clustering) and whole groups are
assigned to one split. Grouping is approximate; a few repeated rooms may still
end up in two splits.

    python split.py                    # show the existing split
    python split.py --rebuild-split    # build it again
"""
import argparse
import hashlib
import json

import cv2
import numpy as np

import config


def is_outdoor(path):
    # Files 01-55 are the outdoor images; 56+ are indoor
    prefix = path.stem.split("_")[0]
    return prefix.isdigit() and 1 <= int(prefix) <= 55


def matched_pairs(hazy_dir, gt_dir):
    gt_names = {p.name for p in gt_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    pairs = []
    for hazy_path in sorted(hazy_dir.iterdir()):
        if hazy_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        gt_name = hazy_path.name.replace("_hazy.png", "_GT.png").replace("_hazy.jpg", "_GT.jpg")
        if gt_name in gt_names:
            pairs.append((hazy_path, gt_dir / gt_name))
        else:
            print(f"  WARNING: no GT image for {hazy_path.name}")
    return pairs


def embeddings(gt_paths):
    """L2-normalised MobileNetV2 features of the GT images, cached in outputs/cache."""
    key = hashlib.sha256()
    for p in gt_paths:
        key.update(p.name.encode())
    cache_file = config.CACHE_DIR / f"embeddings_{key.hexdigest()[:16]}.npy"
    if cache_file.exists():
        print(f"  Loading embeddings from {cache_file.name}")
        return np.load(cache_file)

    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    print(f"  Computing MobileNetV2 embeddings for {len(gt_paths)} images ...")
    model = MobileNetV2(weights=config.EMBED_WEIGHTS, include_top=False, pooling="avg")
    feats = []
    for i in range(0, len(gt_paths), config.EMBED_BATCH):
        batch = []
        for p in gt_paths[i:i + config.EMBED_BATCH]:
            img = cv2.resize(cv2.imread(str(p)), (224, 224), interpolation=cv2.INTER_LINEAR)
            # MobileNetV2 expects RGB; this is the only RGB conversion in the project
            batch.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32))
        feats.append(model(preprocess_input(np.array(batch, dtype=np.float32)), training=False).numpy())

    feats = np.vstack(feats).astype(np.float32)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    feats = feats / np.where(norms == 0, 1.0, norms)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, feats)
    return feats


def group_scenes(emb, gt_paths):
    """Cluster by cosine distance. Group ids are numbered by their first file name."""
    from sklearn.cluster import AgglomerativeClustering

    dist = np.clip(1.0 - emb @ emb.T, 0.0, 2.0).astype(np.float32)
    labels = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage=config.LINKAGE,
                                     distance_threshold=config.DISTANCE_THRESHOLD).fit_predict(dist)
    first_name = {}
    for label, p in zip(labels, gt_paths):
        if label not in first_name or p.name < first_name[label]:
            first_name[label] = p.name
    new_id = {label: i for i, label in enumerate(sorted(first_name, key=first_name.get))}
    return np.array([new_id[label] for label in labels], dtype=np.int32)


def assign_groups(group_ids, outdoor, rng):
    """Give whole groups to val/test (about 10% each), separately for outdoor and indoor."""
    assignment = {}
    for domain in [True, False]:
        groups = {}
        for i in np.where(outdoor == domain)[0]:
            groups.setdefault(int(group_ids[i]), []).append(int(i))
        n_images = sum(len(m) for m in groups.values())
        if n_images == 0:
            continue
        val_target = max(1, round(n_images * config.VAL_FRACTION))
        test_target = max(1, round(n_images * config.TEST_FRACTION))

        # Very large groups always stay in train
        eligible = [(g, m) for g, m in sorted(groups.items()) if len(m) <= config.MAX_GROUP_SIZE_VAL_TEST]
        rng.shuffle(eligible)

        split_of_group, n_val, n_test = {}, 0, 0
        for g, members in eligible:
            if n_val < val_target:
                split_of_group[g] = "val"
                n_val += len(members)
            elif n_test < test_target:
                split_of_group[g] = "test"
                n_test += len(members)
        for g, members in groups.items():
            for i in members:
                assignment[i] = split_of_group.get(g, "train")
    return assignment


def leakage(emb, assignment, rng):
    """Distance from each val/test image to its closest train image (higher = less leakage)."""
    def stats(eval_idx, train_idx):
        d = 1.0 - np.clip((emb[eval_idx] @ emb[train_idx].T).max(axis=1), -1.0, 1.0)
        d = d.astype(np.float64)  # percentiles in float64, as in the saved split file
        return {"median": float(np.median(d)), "p5": float(np.percentile(d, 5)),
                "frac_lt_010": float(np.mean(d < 0.10)), "frac_lt_0125": float(np.mean(d < 0.125)),
                "frac_lt_014": float(np.mean(d < 0.14)), "frac_lt_015": float(np.mean(d < 0.15))}

    idx = {s: [i for i in sorted(assignment) if assignment[i] == s] for s in ("train", "val", "test")}
    eval_idx = sorted(idx["val"] + idx["test"])
    shuffled = np.arange(len(emb))
    rng.shuffle(shuffled)
    return {
        "grouped_split": stats(eval_idx, idx["train"]),
        "val_split": stats(idx["val"], idx["train"]),
        "test_split": stats(idx["test"], idx["train"]),
        # Same sizes but a plain random split, for comparison
        "random_split": stats(shuffled[:len(eval_idx)], shuffled[len(eval_idx):]),
    }


def build_split(rebuild=False, out=config.SPLIT_FILE):
    if out.exists() and not rebuild:
        print(f"[Split] Using existing {out}")
        return json.loads(out.read_text(encoding="utf-8"))

    import keras
    import sklearn
    import tensorflow as tf

    print("[Split] Building split ...")
    pairs = matched_pairs(config.HAZY_FULL_DIR, config.GT_FULL_DIR)
    gt_paths = [gt for _, gt in pairs]
    outdoor = np.array([is_outdoor(p) for p in gt_paths])
    emb = embeddings(gt_paths)
    group_ids = group_scenes(emb, gt_paths)
    print(f"  {len(pairs)} pairs, {outdoor.sum()} outdoor, {len(np.unique(group_ids))} groups")

    assignment = assign_groups(group_ids, outdoor, np.random.default_rng(config.SEED))
    diagnostics = leakage(emb, assignment, np.random.default_rng(config.SEED + 1))
    g = diagnostics["grouped_split"]
    print(f"  val/test -> nearest train: median {g['median']:.3f}, below 0.14: {g['frac_lt_014']:.1%}")

    order = sorted(assignment, key=lambda i: gt_paths[i].name)
    result = {
        "method": {
            "embedding_model": config.EMBED_MODEL, "embedding_weights": config.EMBED_WEIGHTS,
            "linkage": config.LINKAGE, "distance_threshold": config.DISTANCE_THRESHOLD,
            "cap": config.MAX_GROUP_SIZE_VAL_TEST, "seed": config.SEED, "split_mode": "grouped",
            "note": "Grouping is approximate. Do not recompute on Kaggle/Colab -- read this file.",
        },
        "versions": {"tensorflow": tf.__version__, "keras": keras.__version__,
                     "sklearn": sklearn.__version__, "opencv": cv2.__version__},
        "diagnostics": diagnostics,
        "splits": {s: [{"hazy": pairs[i][0].name, "gt": gt_paths[i].name, "group_id": int(group_ids[i]),
                        "domain": "outdoor" if outdoor[i] else "indoor"}
                       for i in order if assignment[i] == s]
                   for s in ("train", "val", "test")},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[Split] Saved {out}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build or show the train/val/test split.")
    parser.add_argument("--rebuild-split", action="store_true", help="Build again even if the file exists.")
    split = build_split(rebuild=parser.parse_args().rebuild_split)
    print({name: len(items) for name, items in split["splits"].items()})
