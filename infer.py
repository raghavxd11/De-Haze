"""
infer.py -- Unified command-line inference entry point for De-Haze.

Usage:
    python infer.py --model-path PATH --input IMG_OR_VIDEO
                    --output outputs/ [--no-display] [--no-detect]
                    [--detect-on dehazed|hazy] [--preprocess MODE]
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import tensorflow as tf

from config import config
from detect import draw_boxes, hog_human_detection
from preprocess import prepare_input

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run De-Haze model inference on an image or video with human detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to trained model (.keras or .h5 file).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input image or video file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.infer_output_dir,
        help="Directory or file path to save dehazed outputs.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable interactive display window (cv2.imshow).",
    )
    parser.add_argument(
        "--no-detect",
        action="store_true",
        help="Disable HOG human detection.",
    )
    parser.add_argument(
        "--detect-on",
        type=str,
        choices=["dehazed", "hazy"],
        default=config.default_detect_on,
        help="Detect humans on 'dehazed' output or on clean copy of 'hazy' input.",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        choices=["clahe_hls", "clahe", "none"],
        default=None,
        help="Preprocessing mode override. If omitted, read from model's sidecar JSON.",
    )
    return parser.parse_args()


def load_dehaze_model(model_path: Path) -> tf.keras.Model:
    """Load model with compile=False for both .keras and .h5 formats."""
    if not model_path.exists():
        print(f"[FATAL ERROR] Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    suffix = model_path.suffix.lower()
    if suffix not in {".keras", ".h5"}:
        print(
            f"[FATAL ERROR] Unsupported model format '{suffix}'. Model file must be .keras or .h5.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        model = tf.keras.models.load_model(str(model_path), compile=False)
        print(f"[Inference] Successfully loaded model from {model_path} (compile=False)")
        return model
    except Exception as e:
        print(f"[FATAL ERROR] Failed to load model {model_path}: {e}", file=sys.stderr)
        sys.exit(1)


def resolve_preprocess_mode(model_path: Path, cli_mode: str | None) -> str:
    """Read preprocess mode from sidecar JSON or CLI override, with warning on mismatch."""
    sidecar_path = model_path.with_suffix(".json")
    sidecar_mode = None

    if sidecar_path.exists():
        try:
            meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar_mode = meta.get("preprocess_mode")
            if sidecar_mode:
                print(f"[Inference] Model sidecar ({sidecar_path.name}) specified mode: '{sidecar_mode}'")
        except Exception as e:
            print(f"[WARNING] Could not read sidecar {sidecar_path}: {e}", file=sys.stderr)

    if cli_mode is not None:
        if sidecar_mode is not None and cli_mode != sidecar_mode:
            print(
                f"[WARNING] Explicit --preprocess '{cli_mode}' differs from model sidecar "
                f"mode '{sidecar_mode}'. Explicit flag takes precedence.",
                file=sys.stderr,
            )
        return cli_mode

    if sidecar_mode is not None:
        return sidecar_mode

    print(f"[Inference] No sidecar mode found; falling back to config default '{config.preprocess_mode}'")
    return config.preprocess_mode


def process_frame(
    model: tf.keras.Model,
    frame: np.ndarray,
    mode: str,
    detect_on: str,
    no_detect: bool,
) -> tuple[np.ndarray, list]:
    """Execute single-frame dehaze and detection pipeline."""
    orig_h, orig_w = frame.shape[:2]

    # 1. Prepare input: preprocess at original resolution then resize to 256x256
    model_in_uint8 = prepare_input(frame, mode=mode, target_size=config.target_size)
    model_in_norm = model_in_uint8.astype(np.float32) / 255.0
    model_input = tf.convert_to_tensor(np.expand_dims(model_in_norm, axis=0), dtype=tf.float32)

    # 2. Call model directly
    pred_tensor = model(model_input, training=False)
    if isinstance(pred_tensor, (list, tuple)):
        pred_tensor = pred_tensor[0]
    pred_arr = pred_tensor[0].numpy() if hasattr(pred_tensor[0], "numpy") else np.array(pred_tensor[0])

    # Convert to uint8 before drawing or writing
    dehazed_256_u8 = np.clip(pred_arr * 255.0, 0, 255).astype(np.uint8)

    # 3. Resize dehazed output back to original frame size
    dehazed_full = cv2.resize(dehazed_256_u8, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # 4. Human detection
    rectangles = []
    if not no_detect:
        if detect_on == "dehazed":
            rectangles = hog_human_detection(dehazed_full)
        elif detect_on == "hazy":
            # Detect on clean copy of original input
            rectangles = hog_human_detection(frame.copy())
        else:
            raise ValueError(f"Unknown detect_on mode '{detect_on}'")

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
    else:
        out_frame = dehazed_full

    return out_frame, rectangles


def infer_image(
    model: tf.keras.Model,
    input_path: Path,
    output_dest: Path,
    mode: str,
    detect_on: str,
    no_display: bool,
    no_detect: bool,
):
    """Run inference on a single image and save as PNG."""
    raw_img = cv2.imread(str(input_path))
    if raw_img is None:
        print(f"[FATAL ERROR] Could not decode image file: {input_path}", file=sys.stderr)
        sys.exit(1)

    orig_h, orig_w = raw_img.shape[:2]
    t0 = time.perf_counter()

    out_img, rects = process_frame(
        model=model,
        frame=raw_img,
        mode=mode,
        detect_on=detect_on,
        no_detect=no_detect,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Determine output file path
    if output_dest.is_dir() or output_dest.suffix == "":
        output_dest.mkdir(parents=True, exist_ok=True)
        detect_tag = f"_{detect_on}" if not no_detect else ""
        out_file = output_dest / f"dehazed_{input_path.stem}{detect_tag}.png"
    else:
        output_dest.parent.mkdir(parents=True, exist_ok=True)
        out_file = output_dest.with_suffix(".png")

    cv2.imwrite(str(out_file), out_img)

    # Optional display
    if not no_display:
        cv2.imshow("Original Input", raw_img)
        cv2.imshow("Dehazed Output", out_img)
        print("[Inference] Displaying output window (press any key to close)...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    out_h, out_w = out_img.shape[:2]
    size_equal = (orig_h, orig_w) == (out_h, out_w)

    print("\n" + "=" * 70)
    print("IMAGE INFERENCE SUMMARY")
    print("=" * 70)
    print(f"  Input file:             {input_path}")
    print(f"  Output file:            {out_file}")
    print(f"  Frames processed:       1")
    print(f"  Average ms per frame:   {elapsed_ms:.2f} ms")
    print(f"  Human count:            {len(rects) if not no_detect else 'Detection disabled'}")
    print(f"  Input size:             {orig_w}x{orig_h}")
    print(f"  Output size:            {out_w}x{out_h}")
    print(f"  Output size == Input:   {size_equal}")
    print("=" * 70)


def infer_video(
    model: tf.keras.Model,
    input_path: Path,
    output_dest: Path,
    mode: str,
    detect_on: str,
    no_display: bool,
    no_detect: bool,
):
    """Run inference on a video file and save with cv2.VideoWriter (mp4v)."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"[FATAL ERROR] Could not open video file: {input_path}", file=sys.stderr)
        sys.exit(1)

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    # Determine destination video path
    if output_dest.is_dir() or output_dest.suffix == "":
        output_dest.mkdir(parents=True, exist_ok=True)
        detect_tag = f"_{detect_on}" if not no_detect else ""
        out_file = output_dest / f"dehazed_{input_path.stem}{detect_tag}.mp4"
    else:
        output_dest.parent.mkdir(parents=True, exist_ok=True)
        out_file = output_dest.with_suffix(".mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (orig_w, orig_h))

    frame_count = 0
    total_time_ms = 0.0
    human_counts = []

    try:
        # Loop handles unknown or streaming frame counts safely
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            t0 = time.perf_counter()
            out_frame, rects = process_frame(
                model=model,
                frame=frame,
                mode=mode,
                detect_on=detect_on,
                no_detect=no_detect,
            )
            elapsed = (time.perf_counter() - t0) * 1000.0

            writer.write(out_frame)
            frame_count += 1
            total_time_ms += elapsed
            if not no_detect:
                human_counts.append(len(rects))

            if not no_display:
                cv2.imshow("Dehazed Video Output", out_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[Inference] Video processing interrupted by user.")
                    break
    finally:
        cap.release()
        writer.release()
        if not no_display:
            cv2.destroyAllWindows()

    avg_ms = total_time_ms / frame_count if frame_count > 0 else 0.0
    avg_humans = sum(human_counts) / len(human_counts) if human_counts else 0.0
    latest_humans = human_counts[-1] if human_counts else (0 if not no_detect else "Detection disabled")

    print("\n" + "=" * 70)
    print("VIDEO INFERENCE SUMMARY")
    print("=" * 70)
    print(f"  Input file:             {input_path}")
    print(f"  Output file:            {out_file}")
    print(f"  Frames processed:       {frame_count}")
    print(f"  Average ms per frame:   {avg_ms:.2f} ms")
    print(f"  Human counts:           latest={latest_humans}, mean={avg_humans:.2f}")
    print(f"  Input size:             {orig_w}x{orig_h}")
    print(f"  Output size:            {orig_w}x{orig_h}")
    print(f"  Output size == Input:   True")
    print("=" * 70)


def main():
    args = parse_args()

    # 1. Validate input file
    input_path = args.input
    if not input_path.exists():
        print(f"[FATAL ERROR] Input file does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    # 2. Resolve preprocess mode from sidecar or CLI
    mode = resolve_preprocess_mode(args.model_path, args.preprocess)

    # 3. Load model (compile=False supports .keras and .h5)
    model = load_dehaze_model(args.model_path)

    # 4. Dispatch based on file type
    suffix = input_path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        infer_video(
            model=model,
            input_path=input_path,
            output_dest=args.output,
            mode=mode,
            detect_on=args.detect_on,
            no_display=args.no_display,
            no_detect=args.no_detect,
        )
    elif suffix in IMAGE_EXTENSIONS:
        infer_image(
            model=model,
            input_path=input_path,
            output_dest=args.output,
            mode=mode,
            detect_on=args.detect_on,
            no_display=args.no_display,
            no_detect=args.no_detect,
        )
    else:
        # Fallback probe
        probe_img = cv2.imread(str(input_path))
        if probe_img is not None:
            infer_image(
                model=model,
                input_path=input_path,
                output_dest=args.output,
                mode=mode,
                detect_on=args.detect_on,
                no_display=args.no_display,
                no_detect=args.no_detect,
            )
        else:
            infer_video(
                model=model,
                input_path=input_path,
                output_dest=args.output,
                mode=mode,
                detect_on=args.detect_on,
                no_display=args.no_display,
                no_detect=args.no_detect,
            )


if __name__ == "__main__":
    main()
