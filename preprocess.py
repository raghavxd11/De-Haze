import cv2
import config


# Supported preprocessing modes:
#   "clahe_hls" : CLAHE on L in LAB, then HLS L-40 / S+30 shift  [DEFAULT]
#                 Identical to the original behavior (proved by np.array_equal).
#   "clahe"     : CLAHE on L in LAB only (no HLS adjustment).
#   "none"      : raw image, no processing.
#
# The mode is selected at call time via the `mode` argument. The default is
# "clahe_hls" so that all existing weights continue to work unchanged.
# Training and inference must use the same mode; the mode is stored as a
# sidecar JSON alongside saved model files (see unet_train.py / pix2pix_train.py).


def preprocess(frame, mode: str = "clahe_hls"):
    """Preprocess a BGR uint8 image for dehazing.

    Args:
        frame: BGR uint8 numpy array (H, W, 3).
        mode:  One of "clahe_hls" (default), "clahe", or "none".

    Returns:
        Preprocessed BGR uint8 numpy array at original resolution.
    """
    if mode == "none":
        return frame

    # ── CLAHE on L channel (LAB) ─────────────────────────────────────────────
    # Increase contrast using Contrast Limited Adaptive Histogram Equalization
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(3, 3))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    dehazed_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    if mode == "clahe":
        return dehazed_frame

    # ── HLS adjustment (clahe_hls only) ──────────────────────────────────────
    # Further adjustments in the HLS color space
    hls = cv2.cvtColor(dehazed_frame, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)
    l = cv2.subtract(l, 40)
    s = cv2.add(s, 30)
    hls = cv2.merge((h, l, s))
    final_frame = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)

    return final_frame


def prepare_input(
    frame,
    mode: str = config.PREPROCESS_MODE,
    target_size: tuple[int, int] = config.IMG_SIZE,
):
    """Prepare a BGR image identically for training and inference.

    Flow:
      1. preprocess(frame, mode) at original resolution
      2. cv2.resize to target_size (from config) with cv2.INTER_AREA

    Args:
        frame: BGR uint8 image at original resolution.
        mode: Preprocessing mode ("clahe_hls" | "clahe" | "none").
        target_size: (width, height) tuple for model input.

    Returns:
        uint8 BGR array of shape (target_size[1], target_size[0], 3).
    """
    prep = preprocess(frame, mode=mode)
    resized = cv2.resize(prep, target_size, interpolation=cv2.INTER_AREA)
    return resized
