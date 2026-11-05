import cv2

# Initialize the HOG descriptor and SVM detector ONCE at module level
_HOG = cv2.HOGDescriptor()
_HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def hog_human_detection(frame):
    """Detect humans using the cached HOG descriptor.

    Args:
        frame: uint8 BGR image.

    Returns:
        rectangles: Array or sequence of (x, y, w, h) bounding boxes.
                    Does NOT modify frame or draw boxes.
    """
    rectangles, _ = _HOG.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )
    return rectangles


def draw_boxes(frame, rects, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes on a frame copy and return the annotated frame.

    Args:
        frame: uint8 BGR image to draw on.
        rects: Iterable of (x, y, w, h) bounding box coordinates.
        color: BGR tuple for box color (default green).
        thickness: Line thickness in pixels.

    Returns:
        Annotated uint8 BGR image.
    """
    out = frame.copy()
    for (x, y, w, h) in rects:
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)
    return out
