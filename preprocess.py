import cv2

import config


def preprocess(frame, mode="clahe_hls"):
    """Contrast boost for a BGR uint8 image. mode: "clahe_hls", "clahe" or "none"."""
    if mode == "none":
        return frame

    # CLAHE on the L channel in LAB
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(3, 3))
    frame = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
    if mode == "clahe":
        return frame

    # Original repo behaviour: darken (L-40) and saturate (S+30) in HLS
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)
    hls = cv2.merge((h, cv2.subtract(l, 40), cv2.add(s, 30)))
    return cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)


def prepare_input(frame, mode=config.PREPROCESS_MODE, size=config.IMG_SIZE):
    """Same input for training and inference: preprocess at full size, then resize."""
    return cv2.resize(preprocess(frame, mode), size, interpolation=cv2.INTER_AREA)
