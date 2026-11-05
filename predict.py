import json
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import tensorflow as tf

import config
from detect import draw_boxes, hog_human_detection
from preprocess import prepare_input


def load_preprocess_mode_from_sidecar(model_path: str | Path) -> str:
    """Read the preprocess mode from a model's sidecar JSON file if present."""
    p = Path(model_path)
    sidecar = p.with_suffix(".json")
    if sidecar.exists():
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                data = json.load(f)
            mode = data.get("preprocess_mode")
            if mode:
                print(f"[Inference] Loaded preprocess mode '{mode}' from {sidecar.name}")
                return mode
        except Exception as e:
            print(f"[Inference WARNING] Could not read sidecar {sidecar}: {e}")
    return config.PREPROCESS_MODE


def predict_and_visualize(
    model,
    img: np.ndarray,
    mode: str | None = None,
    model_path: str | Path | None = None,
    detect_on: Literal["dehazed", "hazy"] = "dehazed",
    show: bool = True,
) -> tuple[np.ndarray, list]:
    """Dehaze an image, detect humans, and visualize/return the result.

    Flow per frame:
      original frame -> prepare_input -> model -> dehazed output (256x256)
      -> resize back to original frame size -> detection -> draw boxes and human count.
      Nothing is ever drawn on arrays fed into the model.

    Args:
        model: Trained Keras model (or callable).
        img: Input BGR image (numpy array, original resolution).
        mode: Preprocessing mode ("clahe_hls" | "clahe" | "none").
              If None and model_path provided, reads from sidecar JSON.
              Defaults to config.PREPROCESS_MODE ("clahe_hls").
        model_path: Optional path to model file for sidecar lookup.
        detect_on: "dehazed" (default) or "hazy".
        show: If True, display windows using cv2.imshow.

    Returns:
        (dehazed_annotated, rectangles): Output image (same size as input) and detected boxes.
    """
    if mode is None and model_path is not None:
        mode = load_preprocess_mode_from_sidecar(model_path)
    elif mode is None:
        mode = config.PREPROCESS_MODE

    orig_h, orig_w = img.shape[:2]

    # 1. Prepare input: preprocess at original resolution then resize to 256x256
    model_in_uint8 = prepare_input(img, mode=mode, target_size=config.IMG_SIZE)
    model_in_norm = model_in_uint8.astype(np.float32) / 255.0
    model_input = tf.convert_to_tensor(np.expand_dims(model_in_norm, axis=0), dtype=tf.float32)

    # 2. Call model directly (fast inference, no training graph updates)
    pred_tensor = model(model_input, training=False)
    if isinstance(pred_tensor, (list, tuple)):
        pred_tensor = pred_tensor[0]
    pred_arr = pred_tensor[0].numpy() if hasattr(pred_tensor[0], "numpy") else np.array(pred_tensor[0])

    # Convert to uint8 before drawing or writing
    dehazed_256_u8 = np.clip(pred_arr * 255.0, 0, 255).astype(np.uint8)

    # 3. Resize dehazed output back to original frame size
    dehazed_full = cv2.resize(dehazed_256_u8, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # 4. Human detection
    if detect_on == "dehazed":
        rectangles = hog_human_detection(dehazed_full)
    elif detect_on == "hazy":
        # Detect on a clean copy of the original frame; never mutate input
        rectangles = hog_human_detection(img.copy())
    else:
        raise ValueError(f"Unknown detect_on mode '{detect_on}'. Use 'dehazed' or 'hazy'.")

    # 5. Draw boxes and human count onto the dehazed output ONLY
    out_img = draw_boxes(dehazed_full, rectangles)
    cv2.putText(
        out_img,
        f"Total Humans: {len(rectangles)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
    )

    if show:
        cv2.imshow("Raw", img)
        cv2.imshow("Prediction", out_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return out_img, rectangles


def vid_read(
    model,
    directory: str | Path,
    mode: str | None = None,
    model_path: str | Path | None = None,
    detect_on: Literal["dehazed", "hazy"] = "dehazed",
    show: bool = True,
    max_frames: int | None = None,
) -> int:
    """Read a video, apply dehazing first, detect humans, and annotate output.

    Flow per frame:
      original frame -> prepare_input -> model -> dehazed output (256x256)
      -> resize back to original frame size -> detection -> draw boxes and human count.
      Nothing is ever drawn on arrays fed into the model.

    Args:
        model: Trained Keras model (or callable).
        directory: Path to video file.
        mode: Preprocessing mode ("clahe_hls" | "clahe" | "none").
        model_path: Optional path to model file for sidecar lookup.
        detect_on: "dehazed" (default) or "hazy".
        show: If True, display frames via cv2.imshow.
        max_frames: Optional frame limit (useful for testing).

    Returns:
        Total frames processed.
    """
    if mode is None and model_path is not None:
        mode = load_preprocess_mode_from_sidecar(model_path)
    elif mode is None:
        mode = config.PREPROCESS_MODE

    cap = cv2.VideoCapture(str(directory))
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if frame is None or not ret:
            break

        orig_h, orig_w = frame.shape[:2]

        # 1. Prepare input: preprocess at original resolution then resize to 256x256
        model_in_uint8 = prepare_input(frame, mode=mode, target_size=config.IMG_SIZE)
        model_in_norm = model_in_uint8.astype(np.float32) / 255.0
        model_input = tf.convert_to_tensor(np.expand_dims(model_in_norm, axis=0), dtype=tf.float32)

        # 2. Call model directly
        pred_tensor = model(model_input, training=False)
        if isinstance(pred_tensor, (list, tuple)):
            pred_tensor = pred_tensor[0]
        pred_arr = pred_tensor[0].numpy() if hasattr(pred_tensor[0], "numpy") else np.array(pred_tensor[0])

        # Convert to uint8 before drawing
        dehazed_256_u8 = np.clip(pred_arr * 255.0, 0, 255).astype(np.uint8)

        # 3. Resize back to original frame size
        dehazed_full = cv2.resize(dehazed_256_u8, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # 4. Human detection
        if detect_on == "dehazed":
            rectangles = hog_human_detection(dehazed_full)
        elif detect_on == "hazy":
            rectangles = hog_human_detection(frame.copy())
        else:
            raise ValueError(f"Unknown detect_on mode '{detect_on}'. Use 'dehazed' or 'hazy'.")

        # 5. Draw boxes and human count onto dehazed output ONLY
        out_frame = draw_boxes(dehazed_full, rectangles)
        cv2.putText(
            out_frame,
            f"Total Humans: {len(rectangles)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )

        frame_count += 1

        if show:
            cv2.imshow("frame", out_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if max_frames is not None and frame_count >= max_frames:
            break

    cap.release()
    if show:
        cv2.destroyAllWindows()

    return frame_count
