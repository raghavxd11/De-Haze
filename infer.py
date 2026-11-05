"""Dehaze an image or a video and count people with HOG.

    python infer.py --model-path outputs/models/unet/unet.keras --input sample_test_files/hazed_img.jpg
    python infer.py --model-path outputs/models/unet/unet.keras --input sample_test_files/test_vid.mp4 --no-display
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

import config
from detect import draw_boxes, hog_human_detection
from preprocess import prepare_input

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"}


def fail(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def load_model(model_path):
    if not model_path.exists():
        fail(f"Model file not found: {model_path}")
    if model_path.suffix.lower() not in {".keras", ".h5"}:
        fail(f"Model file must be .keras or .h5, got '{model_path.suffix}'")
    try:
        # compile=False: we only predict, and it avoids needing the custom metrics
        model = tf.keras.models.load_model(str(model_path), compile=False)
    except Exception as e:
        fail(f"Could not load {model_path}: {e}")
    print(f"[Infer] Loaded {model_path}")
    return model


def preprocess_mode(model_path, cli_mode=None):
    """Use the mode the model was trained with (from its .json sidecar) unless --preprocess is given."""
    sidecar = model_path.with_suffix(".json")
    saved_mode = None
    if sidecar.exists():
        try:
            saved_mode = json.loads(sidecar.read_text(encoding="utf-8")).get("preprocess_mode")
            print(f"[Infer] {sidecar.name} says preprocess mode '{saved_mode}'")
        except Exception as e:
            print(f"[WARNING] Could not read {sidecar}: {e}", file=sys.stderr)

    if cli_mode is not None:
        if saved_mode is not None and cli_mode != saved_mode:
            print(f"[WARNING] --preprocess '{cli_mode}' differs from the model's '{saved_mode}'; "
                  f"using '{cli_mode}'.", file=sys.stderr)
        return cli_mode
    if saved_mode is not None:
        return saved_mode
    print(f"[Infer] No sidecar found; using default '{config.PREPROCESS_MODE}'")
    return config.PREPROCESS_MODE


def dehaze_frame(model, frame, mode, detect_on="dehazed", detect=True):
    """Dehaze one BGR frame. Returns (output at the input's size, people boxes)."""
    h, w = frame.shape[:2]
    x = prepare_input(frame, mode).astype(np.float32) / 255.0
    pred = model(tf.convert_to_tensor(x[None], dtype=tf.float32), training=False)
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    pred = np.array(pred[0])
    dehazed = cv2.resize(np.clip(pred * 255.0, 0, 255).astype(np.uint8), (w, h),
                         interpolation=cv2.INTER_LINEAR)

    if not detect:
        return dehazed, []
    # Boxes are drawn only on the output, never on the frame given to the model
    rects = hog_human_detection(dehazed if detect_on == "dehazed" else frame.copy())
    return draw_boxes(dehazed, rects), rects


def output_file(output, input_path, ext, tag):
    # A path without an extension is treated as a folder
    if output.is_dir() or output.suffix == "":
        output.mkdir(parents=True, exist_ok=True)
        return output / f"dehazed_{input_path.stem}{tag}{ext}"
    output.parent.mkdir(parents=True, exist_ok=True)
    return output.with_suffix(ext)


def infer_image(model, args, mode, tag):
    img = cv2.imread(str(args.input))
    if img is None:
        fail(f"Could not read image: {args.input}")

    t0 = time.perf_counter()
    out, rects = dehaze_frame(model, img, mode, args.detect_on, not args.no_detect)
    ms = (time.perf_counter() - t0) * 1000
    out_file = output_file(args.output, args.input, ".png", tag)
    cv2.imwrite(str(out_file), out)

    if not args.no_display:
        cv2.imshow("Original Input", img)
        cv2.imshow("Dehazed Output", out)
        print("[Infer] Press any key to close the windows")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    humans = "detection off" if args.no_detect else len(rects)
    print(f"[Infer] Saved {out_file}")
    print(f"[Infer] 1 frame, {ms:.2f} ms, humans: {humans}, "
          f"size {img.shape[1]}x{img.shape[0]} -> {out.shape[1]}x{out.shape[0]}")


def infer_video(model, args, mode, tag):
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        fail(f"Could not open video: {args.input}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0
    out_file = output_file(args.output, args.input, ".mp4", tag)
    writer = cv2.VideoWriter(str(out_file), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frames, total_ms, counts = 0, 0.0, []
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            t0 = time.perf_counter()
            out, rects = dehaze_frame(model, frame, mode, args.detect_on, not args.no_detect)
            total_ms += (time.perf_counter() - t0) * 1000
            writer.write(out)
            frames += 1
            counts.append(len(rects))
            if not args.no_display:
                cv2.imshow("Dehazed Video Output", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[Infer] Stopped by user")
                    break
    finally:
        cap.release()
        writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    if args.no_detect:
        humans = "detection off"
    else:
        humans = f"latest {counts[-1] if counts else 0}, mean {sum(counts) / max(len(counts), 1):.2f}"
    print(f"[Infer] Saved {out_file}")
    print(f"[Infer] {frames} frames, {total_ms / max(frames, 1):.2f} ms/frame, humans: {humans}, size {w}x{h}")


def main():
    parser = argparse.ArgumentParser(description="Dehaze an image or video and detect people.")
    parser.add_argument("--model-path", type=Path, required=True, help=".keras or .h5 file")
    parser.add_argument("--input", type=Path, required=True, help="image or video file")
    parser.add_argument("--output", type=Path, default=config.OUTPUTS_DIR, help="folder or file path")
    parser.add_argument("--no-display", action="store_true", help="do not open windows")
    parser.add_argument("--no-detect", action="store_true", help="skip human detection")
    parser.add_argument("--detect-on", choices=["dehazed", "hazy"], default="dehazed",
                        help="run HOG on the dehazed output or on the original input")
    parser.add_argument("--preprocess", choices=config.PREPROCESS_MODES,
                        help="override the mode stored in the model's .json sidecar")
    args = parser.parse_args()

    if not args.input.exists():
        fail(f"Input file not found: {args.input}")
    mode = preprocess_mode(args.model_path, args.preprocess)
    model = load_model(args.model_path)
    tag = "" if args.no_detect else f"_{args.detect_on}"

    ext = args.input.suffix.lower()
    if ext in VIDEO_EXTS:
        infer_video(model, args, mode, tag)
    elif ext in IMAGE_EXTS or cv2.imread(str(args.input)) is not None:
        infer_image(model, args, mode, tag)
    else:
        infer_video(model, args, mode, tag)


if __name__ == "__main__":
    main()
