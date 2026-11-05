"""
split_builder.py -- Build (or load) the deterministic scene-grouped train/val/test split.

Grouping uses MobileNetV2 cosine distance with average linkage clustering at
threshold 0.20 (chosen after a leakage sweep: 0.15 left about 58% of val/test
images with a train image closer than 0.14; 0.20 leaves about 10% combined,
about 4% in test).

Grouping is approximate: some repeated rooms may still straddle splits because
MobileNetV2 cosine similarity cannot guarantee perfect recall for all near-duplicate scenes.

Usage (from repo root):
    python splits/split_builder.py                   # build or load cached split
    python splits/split_builder.py --rebuild-split   # force rebuild
    python splits/split_builder.py --split-mode random  # random baseline (WARNING printed)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

# -- Lazy TF/sklearn imports so the module is importable without TF installed --
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import config  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (all in config; repeated here as local aliases for readability)
# ---------------------------------------------------------------------------
DISTANCE_THRESHOLD: float = config.grouping_distance_threshold
LINKAGE: str = config.grouping_linkage
EMBED_BATCH: int = config.grouping_embed_batch
CAP: int = config.split_max_group_size_val_test
SEED: int = config.split_seed


# ---------------------------------------------------------------------------
# Helper: detect outdoor files
# ---------------------------------------------------------------------------

def _is_outdoor(path: Path) -> bool:
    """True if the file is one of the 55 true-PNG outdoor images (IDs 01-55).

    Determined by inspected format and size: files 01-55 are genuine PNG
    (1600x1200); files 56+ are JPEG data with a .png extension.
    We check only the numeric prefix rather than re-reading every file.
    """
    stem = path.stem  # e.g. "01_GT" or "200_GT"
    try:
        idx = int(stem.split("_")[0])
        return 1 <= idx <= 55
    except (ValueError, IndexError):
        return False


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _sorted_pairs(hazy_dir: Path, gt_dir: Path) -> list[tuple[Path, Path]]:
    """Return sorted (hazy, gt) path pairs. Log warnings for unmatched files."""
    hazy_files = {p.name: p for p in sorted(hazy_dir.iterdir())
                  if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    gt_files = {p.name: p for p in sorted(gt_dir.iterdir())
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}

    pairs: list[tuple[Path, Path]] = []
    warned = 0
    for hname, hpath in sorted(hazy_files.items()):
        gt_name = hname.replace("_hazy.png", "_GT.png").replace("_hazy.jpg", "_GT.jpg")
        if gt_name in gt_files:
            pairs.append((hpath, gt_files[gt_name]))
        else:
            print(f"  [WARNING] No GT partner for: {hname}")
            warned += 1
    for gname in sorted(gt_files):
        hazy_name = gname.replace("_GT.png", "_hazy.png").replace("_GT.jpg", "_hazy.jpg")
        if hazy_name not in hazy_files:
            print(f"  [WARNING] No hazy partner for: {gname}")
            warned += 1
    if warned:
        print(f"  [WARNING] Total unmatched files: {warned}")
    pairs.sort(key=lambda t: t[0].name)
    return pairs


def _load_uint8(path: Path, target_hw: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Read image with cv2.imread (detects real format from bytes), resize to uint8."""
    img = cv2.imread(str(path))  # BGR, keeps native channel order
    if img is None:
        raise RuntimeError(f"[ERROR] cv2.imread could not decode: {path}")
    img = cv2.resize(img, target_hw, interpolation=cv2.INTER_AREA)
    return img  # uint8 BGR


# ---------------------------------------------------------------------------
# Embedding + grouping
# ---------------------------------------------------------------------------

def _file_list_hash(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode())
    return h.hexdigest()[:16]


def _load_or_compute_embeddings(gt_paths: list[Path], cache_dir: Path) -> np.ndarray:
    """Return L2-normalized MobileNetV2 embeddings for GT images (N, D).

    RGB conversion is done ONLY inside this function. Everything else in the
    pipeline stays in BGR (OpenCV native order).
    """
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    cache_dir.mkdir(parents=True, exist_ok=True)
    fhash = _file_list_hash(gt_paths)
    cache_path = cache_dir / f"embeddings_{fhash}.npy"

    if cache_path.exists():
        print(f"  [Cache] Loading embeddings from {cache_path}")
        return np.load(str(cache_path))

    print(f"  [Embed] Building MobileNetV2 embeddings for {len(gt_paths)} GT images ...")
    t0 = time.time()

    model = MobileNetV2(
        weights=config.embedding_weights,
        include_top=False,
        pooling="avg",
    )

    embeddings = []
    for i in range(0, len(gt_paths), EMBED_BATCH):
        batch_paths = gt_paths[i: i + EMBED_BATCH]
        batch = []
        for p in batch_paths:
            img = cv2.imread(str(p))
            if img is None:
                raise RuntimeError(f"[ERROR] Could not decode GT image for embedding: {p}")
            # -- Resize BGR then convert to RGB ONLY here; not elsewhere in pipeline --
            img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            batch.append(img_rgb.astype(np.float32))
        batch_arr = np.array(batch, dtype=np.float32)
        batch_arr = preprocess_input(batch_arr)
        feats = model(batch_arr, training=False).numpy()
        embeddings.append(feats)
        if (i // EMBED_BATCH) % 10 == 0:
            pct = min(100, int(100 * (i + len(batch_paths)) / len(gt_paths)))
            print(f"    {pct}% ({i + len(batch_paths)}/{len(gt_paths)})")

    embeddings = np.vstack(embeddings).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings = embeddings / norms

    elapsed = time.time() - t0
    print(f"  [Embed] Done in {elapsed:.1f}s -- shape {embeddings.shape}")
    np.save(str(cache_path), embeddings)
    return embeddings


def _compute_cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Return (N, N) cosine distance matrix. Embeddings must be L2-normalized."""
    # dot product of unit vectors = cosine similarity; distance = 1 - similarity
    sim = embeddings @ embeddings.T
    dist = 1.0 - sim
    dist = np.clip(dist, 0.0, 2.0).astype(np.float32)
    return dist


def _cluster(dist_matrix: np.ndarray) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage=LINKAGE,
        distance_threshold=DISTANCE_THRESHOLD,
    )
    labels = clustering.fit_predict(dist_matrix)
    return labels


def _make_group_ids(labels: np.ndarray, gt_paths: list[Path]) -> np.ndarray:
    """Re-number groups by smallest sorted file name they contain (deterministic)."""
    n = len(labels)
    raw_to_min_name = {}
    for i, lbl in enumerate(labels):
        name = gt_paths[i].name
        if lbl not in raw_to_min_name or name < raw_to_min_name[lbl]:
            raw_to_min_name[lbl] = name
    # Sort unique raw labels by min-name to assign canonical group ids
    sorted_raw = sorted(raw_to_min_name, key=lambda l: raw_to_min_name[l])
    raw_to_gid = {raw: gid for gid, raw in enumerate(sorted_raw)}
    return np.array([raw_to_gid[lbl] for lbl in labels], dtype=np.int32)


# ---------------------------------------------------------------------------
# Split builder
# ---------------------------------------------------------------------------

def _size_aware_split(
    group_ids: np.ndarray,
    is_outdoor: np.ndarray,
    rng: np.random.Generator,
) -> dict[int, str]:  # image_index -> "train"|"val"|"test"
    """Assign whole groups to train/val/test, stratified by domain."""
    assignment: dict[int, str] = {}

    for domain_flag in [True, False]:
        domain_name = "outdoor" if domain_flag else "indoor"
        idx_domain = np.where(is_outdoor == domain_flag)[0]
        if len(idx_domain) == 0:
            continue

        # Build per-group info for this domain
        groups_in_domain: dict[int, list[int]] = {}
        for i in idx_domain:
            gid = int(group_ids[i])
            groups_in_domain.setdefault(gid, []).append(int(i))

        n_images = len(idx_domain)
        val_target = max(1, round(n_images * config.split_val_fraction))
        test_target = max(1, round(n_images * config.split_test_fraction))

        # Groups larger than CAP always go to train
        eligible: list[tuple[int, list[int]]] = []
        forced_train: list[tuple[int, list[int]]] = []
        for gid, members in sorted(groups_in_domain.items()):
            if len(members) > CAP:
                forced_train.append((gid, members))
            else:
                eligible.append((gid, members))

        # Shuffle eligible groups (deterministic via seeded rng)
        eligible_arr = list(eligible)
        rng.shuffle(eligible_arr)  # in-place shuffle

        val_assigned: list[tuple[int, list[int]]] = []
        test_assigned: list[tuple[int, list[int]]] = []
        val_count = 0
        test_count = 0

        for gid, members in eligible_arr:
            size = len(members)
            if val_count < val_target:
                val_assigned.append((gid, members))
                val_count += size
            elif test_count < test_target:
                test_assigned.append((gid, members))
                test_count += size

        # Everything else -> train
        val_gids = {g for g, _ in val_assigned}
        test_gids = {g for g, _ in test_assigned}

        for gid, members in groups_in_domain.items():
            if gid in val_gids:
                split = "val"
            elif gid in test_gids:
                split = "test"
            else:
                split = "train"
            for i in members:
                assignment[i] = split

        # Report
        train_count = sum(1 for i in idx_domain if assignment[i] == "train")
        val_count2  = sum(1 for i in idx_domain if assignment[i] == "val")
        test_count2 = sum(1 for i in idx_domain if assignment[i] == "test")
        print(f"  [{domain_name}] train={train_count}  val={val_count2}  test={test_count2}  "
              f"(groups: total={len(groups_in_domain)}, "
              f"val={len(val_assigned)}, test={len(test_assigned)})")

    return assignment


# ---------------------------------------------------------------------------
# Leakage diagnostics
# ---------------------------------------------------------------------------

def _leakage_diagnostics(
    embeddings: np.ndarray,
    pairs: list[tuple[Path, Path]],
    assignment: dict[int, str],
    rng: np.random.Generator,
) -> dict:
    """
    For every val and test image, compute min cosine distance to any train image.
    Also compute same for a random per-image split of the same size.
    """
    n = len(pairs)
    train_idx = [i for i in range(n) if assignment[i] == "train"]
    val_idx   = [i for i in range(n) if assignment[i] == "val"]
    test_idx  = [i for i in range(n) if assignment[i] == "test"]
    eval_idx  = [i for i in range(n) if assignment[i] in ("val", "test")]

    if not train_idx or not eval_idx:
        return {}

    def _stats(dists):
        arr = np.array(dists)
        return {
            "median":       float(np.median(arr)),
            "p5":           float(np.percentile(arr, 5)),
            "frac_lt_010":  float(np.mean(arr < 0.10)),
            "frac_lt_0125": float(np.mean(arr < 0.125)),
            "frac_lt_014":  float(np.mean(arr < 0.14)),
            "frac_lt_015":  float(np.mean(arr < 0.15)),
        }

    # Grouped split distances
    train_emb = embeddings[train_idx]
    eval_emb  = embeddings[eval_idx]
    val_emb   = embeddings[val_idx]
    test_emb  = embeddings[test_idx]

    sim_eval = eval_emb @ train_emb.T
    min_dist_eval = (1.0 - np.clip(sim_eval.max(axis=1), -1.0, 1.0)).tolist()

    sim_val = val_emb @ train_emb.T
    min_dist_val = (1.0 - np.clip(sim_val.max(axis=1), -1.0, 1.0)).tolist()

    sim_test = test_emb @ train_emb.T
    min_dist_test = (1.0 - np.clip(sim_test.max(axis=1), -1.0, 1.0)).tolist()

    # Random per-image split of same size
    rand_idx = np.arange(n)
    rng.shuffle(rand_idx)
    n_eval = len(eval_idx)
    rand_eval  = rand_idx[:n_eval].tolist()
    rand_train = rand_idx[n_eval:].tolist()
    if rand_train:
        rt_emb = embeddings[rand_train]
        re_emb = embeddings[rand_eval]
        sim2 = re_emb @ rt_emb.T
        min_dist_random = (1.0 - np.clip(sim2.max(axis=1), -1.0, 1.0)).tolist()
    else:
        min_dist_random = [0.0] * n_eval

    return {
        "grouped_split": _stats(min_dist_eval),
        "val_split":     _stats(min_dist_val),
        "test_split":    _stats(min_dist_test),
        "random_split":  _stats(min_dist_random),
        "note": (
            "Higher min-distance to train -> less leakage. Threshold 0.20 chosen "
            "after sweep (0.15 left ~58% < 0.14; 0.20 has ~10% combined, ~4% in test)."
        ),
    }


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

SAME_GROUP_PAIRS  = [(2000, 2036), (1000, 1800), (800, 1900)]
DIFF_GROUP_PAIRS  = [(100, 200), (300, 400), (700, 800)]
REPORT_ONLY_PAIRS = [(1400, 1500)]


def _index_for_id(file_id: int, gt_paths: list[Path]) -> int | None:
    """Find index in gt_paths for a file whose stem starts with <file_id>_GT."""
    for i, p in enumerate(gt_paths):
        stem = p.stem  # e.g. "2000_GT" or "01_GT"
        prefix = stem.split("_")[0]
        try:
            if int(prefix) == file_id:
                return i
        except ValueError:
            continue
    return None


def _sanity_checks(group_ids: np.ndarray, gt_paths: list[Path]) -> list[str]:
    results: list[str] = []

    def _check(a, b, expect_same):
        ia = _index_for_id(a, gt_paths)
        ib = _index_for_id(b, gt_paths)
        if ia is None or ib is None:
            return f"  ({a},{b}): SKIP - file not found"
        ga, gb = int(group_ids[ia]), int(group_ids[ib])
        same = ga == gb
        mark = "OK" if same == expect_same else "FAIL"
        expectation = "SAME" if expect_same else "DIFF"
        return f"  ({a},{b}): groups {ga}/{gb} -> [{mark}] (expected {expectation})"

    results.append("-- Must be SAME group:")
    for a, b in SAME_GROUP_PAIRS:
        results.append(_check(a, b, expect_same=True))
    results.append("-- Must be DIFFERENT groups:")
    for a, b in DIFF_GROUP_PAIRS:
        results.append(_check(a, b, expect_same=False))
    results.append("-- Reported (not required to match):")
    for a, b in REPORT_ONLY_PAIRS:
        ia = _index_for_id(a, gt_paths)
        ib = _index_for_id(b, gt_paths)
        if ia is None or ib is None:
            results.append(f"  ({a},{b}): SKIP - file not found")
        else:
            ga, gb = int(group_ids[ia]), int(group_ids[ib])
            results.append(f"  ({a},{b}): groups {ga}/{gb} -> {'SAME' if ga==gb else 'DIFF'}")
    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_split(
    hazy_dir: Path,
    gt_dir: Path,
    rebuild: bool = False,
    split_mode: str = "grouped",
) -> dict:
    """Build or load the split. Returns the split dict (also written to JSON)."""
    import tensorflow as tf
    import keras
    import sklearn
    import cv2 as cv

    split_path = config.split_file
    split_path.parent.mkdir(parents=True, exist_ok=True)

    if split_path.exists() and not rebuild:
        print(f"[Split] Loading existing split from {split_path}")
        return json.loads(split_path.read_text(encoding="utf-8"))

    if split_mode == "random":
        print("=" * 60)
        print("WARNING: --split-mode random is for diagnostic comparison ONLY.")
        print("Results can be inflated by near-duplicate scenes in the same split.")
        print("Do NOT use for formal benchmark numbers.")
        print("=" * 60)

    t_start = time.time()
    print(f"\n[Split] Building split from {gt_dir} ...")
    pairs = _sorted_pairs(hazy_dir, gt_dir)
    gt_paths = [p for _, p in pairs]
    hazy_paths = [p for p, _ in pairs]
    print(f"  Matched pairs: {len(pairs)}")

    is_outdoor = np.array([_is_outdoor(p) for p in gt_paths], dtype=bool)
    n_outdoor = is_outdoor.sum()
    print(f"  Outdoor (IDs 01-55): {n_outdoor}   Indoor: {(~is_outdoor).sum()}")

    # -- Embeddings ---------------------------------------------------------
    embeddings = _load_or_compute_embeddings(gt_paths, config.cache_dir)

    # -- Group or random labels ----------------------------------------------
    if split_mode == "grouped":
        dist_matrix = _compute_cosine_distance_matrix(embeddings)
        raw_labels  = _cluster(dist_matrix)
        group_ids   = _make_group_ids(raw_labels, gt_paths)
    else:
        # Random: each image is its own "group" for split purposes
        rng_r = np.random.default_rng(SEED)
        group_ids = np.arange(len(gt_paths), dtype=np.int32)
        rng_r.shuffle(group_ids)

    # -- Group statistics ----------------------------------------------------
    unique_gids, group_sizes = np.unique(group_ids, return_counts=True)
    n_groups = len(unique_gids)
    sizes_sorted = np.sort(group_sizes)[::-1]
    print(f"\n[Groups] Total: {n_groups}")
    print(f"  Size distribution: min={sizes_sorted.min()}  "
          f"median={int(np.median(sizes_sorted))}  max={sizes_sorted.max()}")
    top10 = sizes_sorted[:10]
    print(f"  Top-10 group sizes: {top10.tolist()}")
    if sizes_sorted.max() > 0.05 * len(gt_paths):
        print(f"  [WARNING] Largest group ({sizes_sorted.max()}) exceeds 5% of all images "
              f"({len(gt_paths)}).")

    # Check outdoor groups (55 outdoor files should be 55 separate groups)
    outdoor_gids = group_ids[is_outdoor]
    n_outdoor_groups = len(np.unique(outdoor_gids))
    if n_outdoor_groups != n_outdoor:
        print(f"  [WARNING] {n_outdoor} outdoor files yielded only {n_outdoor_groups} groups "
              f"(expected {n_outdoor} singletons).")
    else:
        print(f"  [OK] All {n_outdoor} outdoor files are in separate groups.")

    # -- Sanity checks -------------------------------------------------------
    print("\n[Sanity checks]")
    for line in _sanity_checks(group_ids, gt_paths):
        print(line)

    # -- Split assignment ----------------------------------------------------
    rng = np.random.default_rng(SEED)
    assignment = _size_aware_split(group_ids, is_outdoor, rng)

    # Assert no group in more than one split
    gid_splits: dict[int, set] = {}
    for i, split_name in assignment.items():
        gid = int(group_ids[i])
        gid_splits.setdefault(gid, set()).add(split_name)
    violations = {gid: splits for gid, splits in gid_splits.items() if len(splits) > 1}
    if violations:
        print(f"\n[ERROR] {len(violations)} groups appear in multiple splits: {violations}")
    else:
        print("\n[OK] No group appears in more than one split.")

    # Per-split summary
    for split_name in ("train", "val", "test"):
        idx_s = [i for i, s in assignment.items() if s == split_name]
        out_s  = sum(1 for i in idx_s if is_outdoor[i])
        ind_s  = sum(1 for i in idx_s if not is_outdoor[i])
        gids_s = {int(group_ids[i]) for i in idx_s}
        ind_g  = len({int(group_ids[i]) for i in idx_s if not is_outdoor[i]})
        out_g  = len({int(group_ids[i]) for i in idx_s if is_outdoor[i]})
        sizes_s = sorted([group_sizes[np.where(unique_gids == g)[0][0]] for g in gids_s], reverse=True)
        max_g = sizes_s[0] if sizes_s else 0
        print(f"  {split_name:5s}: total={len(idx_s):5d}  outdoor={out_s:3d} (groups: {out_g:2d})  indoor={ind_s:4d} (groups: {ind_g:2d})  "
              f"total_groups={len(gids_s):4d}  largest_group={max_g}")

    # -- Leakage diagnostics -------------------------------------------------
    print("\n[Leakage diagnostics] Computing min cosine distance val/test -> train ...")
    leakage = _leakage_diagnostics(embeddings, pairs, assignment, np.random.default_rng(SEED + 1))
    if leakage:
        g = leakage.get("grouped_split", {})
        v = leakage.get("val_split", {})
        t = leakage.get("test_split", {})
        r = leakage.get("random_split", {})
        print(f"  Grouped (eval): median dist={g.get('median', 0):.4f}  p5={g.get('p5', 0):.4f}  "
              f"<0.10: {g.get('frac_lt_010', 0)*100:.2f}%  <0.125: {g.get('frac_lt_0125', 0)*100:.2f}%  "
              f"<0.14: {g.get('frac_lt_014', 0)*100:.2f}%  <0.15: {g.get('frac_lt_015', 0)*100:.2f}%")
        print(f"  Val split:      median dist={v.get('median', 0):.4f}  p5={v.get('p5', 0):.4f}  "
              f"<0.10: {v.get('frac_lt_010', 0)*100:.2f}%  <0.125: {v.get('frac_lt_0125', 0)*100:.2f}%  "
              f"<0.14: {v.get('frac_lt_014', 0)*100:.2f}%  <0.15: {v.get('frac_lt_015', 0)*100:.2f}%")
        print(f"  Test split:     median dist={t.get('median', 0):.4f}  p5={t.get('p5', 0):.4f}  "
              f"<0.10: {t.get('frac_lt_010', 0)*100:.2f}%  <0.125: {t.get('frac_lt_0125', 0)*100:.2f}%  "
              f"<0.14: {t.get('frac_lt_014', 0)*100:.2f}%  <0.15: {t.get('frac_lt_015', 0)*100:.2f}%")
        print(f"  Random split:   median dist={r.get('median', 0):.4f}  p5={r.get('p5', 0):.4f}")
        print(f"  {leakage.get('note', '')}")

    # -- Write JSON ----------------------------------------------------------
    split_data = {
        "method": {
            "embedding_model":       config.embedding_model,
            "embedding_weights":     config.embedding_weights,
            "linkage":               config.grouping_linkage,
            "distance_threshold":    config.grouping_distance_threshold,
            "cap":                   config.split_max_group_size_val_test,
            "seed":                  config.split_seed,
            "split_mode":            split_mode,
            "note": (
                "Grouping uses MobileNetV2 cosine distance with average linkage "
                "clustering at threshold 0.20, chosen after a leakage sweep (0.15 left "
                "about 58% of val/test images with a train image closer than 0.14; "
                "0.20 leaves about 10% combined, about 4% in test). Grouping is approximate; "
                "some repeated rooms may still straddle splits. Do not recompute on "
                "Kaggle/Colab -- read this file."
            ),
        },
        "versions": {
            "tensorflow": tf.__version__,
            "keras":      keras.__version__,
            "sklearn":    sklearn.__version__,
            "opencv":     cv.__version__,
        },
        "diagnostics": leakage,
        "splits": {
            split_name: [
                {
                    "hazy": hazy_paths[i].name,
                    "gt":   gt_paths[i].name,
                    "group_id": int(group_ids[i]),
                    "domain": "outdoor" if is_outdoor[i] else "indoor",
                }
                for i in sorted(assignment, key=lambda k: gt_paths[k].name)
                if assignment[i] == split_name
            ]
            for split_name in ("train", "val", "test")
        },
    }
    split_path.write_text(json.dumps(split_data, indent=2), encoding="utf-8")
    elapsed_total = time.time() - t_start
    print(f"\n[Split] Written to {split_path}  (total time: {elapsed_total:.1f}s)")
    return split_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build train/val/test split for the dehazing dataset.")
    parser.add_argument("--rebuild-split", action="store_true", help="Force rebuild even if split_v1.json exists.")
    parser.add_argument("--split-mode", choices=["grouped", "random"], default="grouped",
                        help="'grouped' (default) or 'random' (diagnostic only).")
    args = parser.parse_args()

    build_split(
        hazy_dir=config.hazy_full_dir,
        gt_dir=config.gt_full_dir,
        rebuild=args.rebuild_split,
        split_mode=args.split_mode,
    )
