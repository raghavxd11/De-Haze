"""Dark Channel Prior dehazing (He, Sun and Tang, CVPR 2009) for 256x256 BGR uint8 images.

    python baselines/dcp.py     # runs on 3 sample images, saves to outputs/dcp_samples/

Settings follow the paper (15x15 patch, omega 0.95, t0 0.1, brightest 0.1%).
The guided filter radius 24 and eps 1e-3 are fixed values; they were NOT tuned
on the test split (or on any split).
"""
import cv2
import numpy as np

PATCH = 15
OMEGA = 0.95
T0 = 0.1
TOP_FRACTION = 0.001
GF_RADIUS = 24
GF_EPS = 1e-3


def dark_channel(img):
    """Minimum over the colour channels, then over a PATCH x PATCH window."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (PATCH, PATCH))
    return cv2.erode(img.min(axis=2), kernel)


def atmospheric_light(img, dark):
    """Among the brightest 0.1% dark-channel pixels, take the colour of the brightest input pixel."""
    flat_img = img.reshape(-1, 3)
    n = max(1, int(dark.size * TOP_FRACTION))
    candidates = np.argsort(dark.ravel())[-n:]
    best = candidates[np.argmax(flat_img[candidates].sum(axis=1))]
    return flat_img[best]


def guided_filter(guide, src, radius=GF_RADIUS, eps=GF_EPS):
    """Edge-preserving smoothing of src using a grey guide image (He et al., 2010)."""
    size = (2 * radius + 1, 2 * radius + 1)
    mean = lambda x: cv2.boxFilter(x, cv2.CV_64F, size)
    mean_i, mean_p = mean(guide), mean(src)
    var_i = mean(guide * guide) - mean_i * mean_i
    cov_ip = mean(guide * src) - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return mean(a) * guide + mean(b)


def dehaze(bgr):
    """bgr: uint8 image (H, W, 3). Returns the dehazed uint8 BGR image."""
    img = bgr.astype(np.float64) / 255.0
    A = atmospheric_light(img, dark_channel(img))
    t = 1.0 - OMEGA * dark_channel(img / np.maximum(A, 1e-6))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    t = np.maximum(guided_filter(gray, t), T0)
    out = (img - A) / t[:, :, None] + A
    return (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    import config

    out_dir = config.OUTPUTS_DIR / "dcp_samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in ["sample_test_files/hazed_img.jpg", "dataset/Hazy_Small/01_hazy.png",
                 "dataset/Hazy_Full/200_hazy.png"]:
        hazy = cv2.resize(cv2.imread(str(root / path)), config.IMG_SIZE, interpolation=cv2.INTER_AREA)
        result = dehaze(hazy)
        name = Path(path).stem
        cv2.imwrite(str(out_dir / f"{name}_dcp.png"), result)
        cv2.imwrite(str(out_dir / f"{name}_side_by_side.png"), np.hstack([hazy, result]))
        print(f"{path}: mean BGR in {hazy.reshape(-1, 3).mean(0).round(1)} -> out "
              f"{result.reshape(-1, 3).mean(0).round(1)}, saved {out_dir / name}_*.png")
