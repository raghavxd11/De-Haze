import cv2

# Build the HOG people detector once, not per frame
HOG = cv2.HOGDescriptor()
HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def hog_human_detection(frame):
    """Return (x, y, w, h) boxes of people found in a BGR image."""
    rects, _ = HOG.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
    return rects


def draw_boxes(frame, rects):
    """Draw green boxes and the human count on a copy of the frame."""
    out = frame.copy()
    for (x, y, w, h) in rects:
        cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
    cv2.putText(out, f"Total Humans: {len(rects)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return out
