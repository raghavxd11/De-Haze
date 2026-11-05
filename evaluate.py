"""Score dehazing methods on the fixed split in splits/split_v1.json.

    python evaluate.py --model-path outputs/models/unet/unet.keras --compare-dcp
    python evaluate.py --export-test-inputs exported/test_inputs
    python evaluate.py --external-dir results/aodnet --external-name AOD-Net

Every method is scored the same way: its 256x256 uint8 output against the
ground truth stretched to 256x256 (INTER_AREA), with tf.image PSNR/SSIM.
Results go to outputs/results_per_image.csv, outputs/results.csv and outputs/results.md.
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

import config
from baselines.dcp import dehaze as dcp_dehaze
from data import load_pairs
from infer import load_model, preprocess_mode

N_BOOTSTRAP = 2000
WARMUP = 3
PSNR_CAP = 100.0
SUBSETS = ["overall", "indoor", "outdoor"]


def load_items(split, limit=None):
    items = json.loads(config.SPLIT_FILE.read_text(encoding="utf-8"))["splits"][split]
    if limit and limit < len(items):
        # Evenly spaced, so a short run still has both indoor and outdoor images
        idx = np.linspace(0, len(items) - 1, limit).round().astype(int)
        items = [items[i] for i in idx]
    return items


def score(pred, gt):
    """PSNR and SSIM of two uint8 images in [0, 1] space. Identical images give PSNR 100."""
    p = tf.constant(pred.astype(np.float32) / 255.0)
    g = tf.constant(gt.astype(np.float32) / 255.0)
    psnr = float(tf.image.psnr(p, g, max_val=1.0))
    ssim = float(tf.image.ssim(p, g, max_val=1.0))
    return (PSNR_CAP if math.isinf(psnr) else psnr), ssim


def bootstrap_ci(values, groups, n=N_BOOTSTRAP, seed=config.SEED):
    """95% CI of the mean of per-group means, resampling whole scene groups."""
    values, groups = np.asarray(values, dtype=np.float64), np.asarray(groups)
    group_means = np.array([values[groups == g].mean() for g in np.unique(groups)])
    rng = np.random.default_rng(seed)
    samples = rng.choice(group_means, size=(n, len(group_means)), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def run_timed(fn, inputs):
    """Run fn on each image (batch size 1). Returns outputs and mean ms/image after WARMUP calls."""
    for _ in range(WARMUP):
        fn(inputs[0])
    outputs, total = [], 0.0
    for x in inputs:
        t0 = time.perf_counter()
        outputs.append(fn(x))
        total += time.perf_counter() - t0
    return outputs, 1000.0 * total / len(inputs)


def model_runner(model):
    def run(hazy):
        x = tf.convert_to_tensor(hazy[None].astype(np.float32) / 255.0)
        y = model(x, training=False)
        if isinstance(y, (list, tuple)):
            y = y[0]
        # Same uint8 conversion as infer.py, so we score what would be saved
        return np.clip(np.array(y[0]) * 255.0, 0, 255).astype(np.uint8)
    return run


def load_external(folder, items, allow_missing):
    """Read <hazy stem>.png outputs of another model. Missing files become None."""
    w, h = config.IMG_SIZE
    outputs, missing, resized = [], [], 0
    for item in items:
        path = folder / (Path(item["hazy"]).stem + ".png")
        img = cv2.imread(str(path)) if path.exists() else None
        if img is None:
            missing.append(path.name)
        elif img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            resized += 1
        outputs.append(img)
    if missing and not allow_missing:
        raise SystemExit(f"[ERROR] {len(missing)} files missing in {folder}, e.g. {missing[:3]}. "
                         f"Use --allow-missing to score the rest.")
    if missing:
        print(f"[WARNING] {len(missing)} files missing in {folder}; scoring the other {len(items) - len(missing)}")
    if resized:
        print(f"[WARNING] {resized} images in {folder} were not 256x256 and were resized with INTER_AREA")
    return outputs, len(missing)


def summarize(method, rows, params, ms, device):
    out = []
    for subset in SUBSETS:
        r = [x for x in rows if subset == "overall" or x["domain"] == subset]
        line = {"method": method, "subset": subset, "images": len(r),
                "groups": len({x["group_id"] for x in r}), "params": params,
                "ms_per_image": ms, "device": device}
        for metric in ["psnr", "ssim"]:
            if not r:
                continue
            vals = np.array([x[metric] for x in r])
            groups = np.array([x["group_id"] for x in r])
            line[f"{metric}_image_mean"] = float(vals.mean())
            line[f"{metric}_group_mean"] = float(np.mean([vals[groups == g].mean() for g in np.unique(groups)]))
            line[f"{metric}_ci_low"], line[f"{metric}_ci_high"] = bootstrap_ci(vals, groups)
        out.append(line)
    return out


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, summary, notes, split):
    def fmt(row, metric, digits):
        if f"{metric}_image_mean" not in row:
            return "-", "-"
        return (f"{row[f'{metric}_image_mean']:.{digits}f}",
                f"{row[f'{metric}_group_mean']:.{digits}f} [{row[f'{metric}_ci_low']:.{digits}f}, "
                f"{row[f'{metric}_ci_high']:.{digits}f}]")

    lines = [f"# Results ({split} split)", ""]
    lines += [f"- {n}" for n in notes] + [""]
    lines.append("| Method | Subset | Images | Groups | PSNR image mean | PSNR group mean [95% CI] "
                 "| SSIM image mean | SSIM group mean [95% CI] | Params | ms/image | Device |")
    lines.append("|" + "---|" * 11)
    for row in summary:
        psnr_img, psnr_grp = fmt(row, "psnr", 2)
        ssim_img, ssim_grp = fmt(row, "ssim", 4)
        params = f"{row['params']:,}" if isinstance(row["params"], int) else row["params"]
        ms = f"{row['ms_per_image']:.1f}" if isinstance(row["ms_per_image"], float) else row["ms_per_image"]
        lines.append(f"| {row['method']} | {row['subset']} | {row['images']} | {row['groups']} | {psnr_img} "
                     f"| {psnr_grp} | {ssim_img} | {ssim_grp} | {params} | {ms} | {row['device']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate dehazing methods on splits/split_v1.json.")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--model-path", type=Path, action="append", default=[],
                        help=".keras or .h5 model; can be given more than once")
    parser.add_argument("--compare-dcp", action="store_true", help="also score Dark Channel Prior")
    parser.add_argument("--limit", type=int, help="only N images, for quick checks (numbers are not final)")
    parser.add_argument("--export-test-inputs", type=Path, metavar="DIR",
                        help="save the raw 256x256 hazy inputs as <hazy stem>.png for external models")
    parser.add_argument("--external-dir", type=Path, help="folder with <hazy stem>.png outputs of another model")
    parser.add_argument("--external-name", help="name for the external method (default: folder name)")
    parser.add_argument("--allow-missing", action="store_true", help="score external outputs even if some are missing")
    args = parser.parse_args()

    if not (args.model_path or args.compare_dcp or args.external_dir or args.export_test_inputs):
        parser.error("nothing to do: give --model-path, --compare-dcp, --external-dir or --export-test-inputs")

    items = load_items(args.split, args.limit)
    total = len(load_items(args.split))
    pairs = [(config.HAZY_FULL_DIR / it["hazy"], config.GT_FULL_DIR / it["gt"]) for it in items]
    # Mode "none" gives the raw hazy image, only resized with INTER_AREA
    raw_hazy, gt = load_pairs(pairs, "none", f"eval_{args.split}")

    gpus = tf.config.list_physical_devices("GPU")
    tf_device = "GPU (" + ", ".join(g.name for g in gpus) + ")" if gpus else "CPU"
    print(f"[Eval] {args.split} split: {len(items)} of {total} images. TensorFlow device: {tf_device}")

    if args.export_test_inputs:
        args.export_test_inputs.mkdir(parents=True, exist_ok=True)
        for item, img in zip(items, raw_hazy):
            cv2.imwrite(str(args.export_test_inputs / (Path(item["hazy"]).stem + ".png")), img)
        print(f"[Eval] Saved {len(items)} raw 256x256 hazy inputs to {args.export_test_inputs}")

    methods = []  # (name, outputs, params, ms, device)
    for path in args.model_path:
        mode = preprocess_mode(path)
        model = load_model(path)
        hazy, _ = load_pairs(pairs, mode, f"eval_{args.split}")
        outputs, ms = run_timed(model_runner(model), hazy)
        name = path.stem if path.stem not in [m[0] for m in methods] else f"{path.parent.name}/{path.stem}"
        methods.append((name, outputs, model.count_params(), ms, tf_device))
        print(f"[Eval] {name}: preprocess '{mode}', {ms:.1f} ms/image")
    if args.compare_dcp:
        outputs, ms = run_timed(dcp_dehaze, raw_hazy)
        methods.append(("DCP", outputs, 0, ms, "CPU (NumPy/OpenCV)"))
        print(f"[Eval] DCP: {ms:.1f} ms/image")
    missing = {}
    if args.external_dir:
        name = args.external_name or args.external_dir.name
        outputs, missing[name] = load_external(args.external_dir, items, args.allow_missing)
        methods.append((name, outputs, "n/a", "n/a", "n/a (run elsewhere)"))

    if not methods:
        return

    per_image, summary = [], []
    for name, outputs, params, ms, device in methods:
        rows = []
        for item, out, target in zip(items, outputs, gt):
            if out is None:
                continue
            psnr, ssim = score(out, target)
            rows.append({"method": name, "split": args.split, "hazy": item["hazy"], "gt": item["gt"],
                         "group_id": item["group_id"], "domain": item["domain"], "psnr": psnr, "ssim": ssim})
        per_image += rows
        summary += summarize(name, rows, params, ms, device)

    n_out = sum(it["domain"] == "outdoor" for it in items)
    notes = [f"{len(items)} of {total} {args.split} images, "
             f"{len({it['group_id'] for it in items})} scene groups."]
    if args.limit and args.limit < total:
        notes.append(f"**NOT FINAL: --limit {args.limit} was used, so only part of the split was scored.**")
    if n_out < 10:
        notes.append(f"**Outdoor rows are noisy: only {n_out} outdoor images.**")
    if args.compare_dcp:
        notes.append("DCP is run on the raw hazy image resized with INTER_AREA, without the CLAHE/HLS "
                     "preprocessing used by our models.")
    for name, n_missing in missing.items():
        if n_missing:
            notes.append(f"**{name} is missing {n_missing} images; its rows use fewer images "
                         f"and are not directly comparable.**")
    notes += ["PSNR/SSIM: tf.image, max_val 1.0, on uint8 outputs vs the ground truth stretched to "
              f"256x256 (INTER_AREA). Infinite PSNR is capped at {PSNR_CAP:.0f} dB.",
              f"Group mean = mean of per-scene-group means. 95% CI: bootstrap over scene groups "
              f"({N_BOOTSTRAP} resamples, seed {config.SEED}).",
              f"ms/image: batch size 1, after {WARMUP} warm-up runs, includes NumPy conversion."]

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(config.OUTPUTS_DIR / "results_per_image.csv", per_image)
    write_csv(config.OUTPUTS_DIR / "results.csv", summary)
    write_markdown(config.OUTPUTS_DIR / "results.md", summary, notes, args.split)
    for n in notes[:-3]:
        print("[Eval] " + n.replace("**", ""))
    print(f"[Eval] Saved results_per_image.csv, results.csv and results.md in {config.OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
