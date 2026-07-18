"""
task2_pixel_manipulation.py
===========================
Live webcam feed with six pixel-manipulation modes, cycled with the
keyboard.

Modes (press 1-6 or ← / → arrows to cycle):
    1  – Raw feed
    2  – Channel-swapped (BGR → RGB display trick)
    3  – Brightness / Contrast adjustment  (trackbars)
    4  – Pixelated ROI (centre rectangle)
    5  – Sepia filter
    6  – Live histogram overlay

Other keys:
    q  – Quit

Performance note:
    All transforms use NumPy / OpenCV vectorized operations.
    See NOTES.md for a comparison with naive loop-based approaches.
"""

import cv2
import numpy as np
import time
import os

# ── Configuration ────────────────────────────────────────────────────
CAMERA_INDEX = 0
MODE_NAMES = {
    1: "Raw Feed",
    2: "Channel Swap (BGR → RGB)",
    3: "Brightness / Contrast",
    4: "Pixelated ROI",
    5: "Sepia Filter",
    6: "Live Histogram",
}

WINDOW = "Pixel Manipulation – Task 2"


# ── Filter implementations ──────────────────────────────────────────

def channel_swap(frame: np.ndarray) -> np.ndarray:
    """Swap B and R channels → appears as RGB when shown through imshow (BGR)."""
    return frame[:, :, ::-1].copy()


def adjust_brightness_contrast(
    frame: np.ndarray, brightness: int = 0, contrast: int = 0
) -> np.ndarray:
    """
    Vectorized brightness/contrast.
        alpha = contrast factor  (1.0 = neutral)
        beta  = brightness offset
    Uses cv2.convertScaleAbs which is a fast, saturating cast.
    """
    # Map contrast slider (0-200) → alpha 0.0-3.0
    alpha = 1.0 + contrast / 100.0
    beta = brightness
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


def pixelate_roi(frame: np.ndarray, block_size: int = 12) -> np.ndarray:
    """Pixelate a centred rectangle using vectorized resize-down-then-up."""
    h, w = frame.shape[:2]
    # Define ROI – centre 40 % of frame
    x1, y1 = int(w * 0.3), int(h * 0.3)
    x2, y2 = int(w * 0.7), int(h * 0.7)

    roi = frame[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    # Shrink then enlarge → pixelation
    small = cv2.resize(roi, (max(1, rw // block_size), max(1, rh // block_size)),
                       interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)

    out = frame.copy()
    out[y1:y2, x1:x2] = pixelated
    # Draw the ROI border
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
    return out


def sepia(frame: np.ndarray) -> np.ndarray:
    """Apply sepia tone using a vectorized matrix multiply."""
    # Standard sepia kernel
    kernel = np.array(
        [[0.272, 0.534, 0.131],
         [0.349, 0.686, 0.168],
         [0.393, 0.769, 0.189]],
        dtype=np.float32,
    )
    # cv2.transform applies the 3×3 matrix per pixel — fully vectorized
    sepia_img = cv2.transform(frame, kernel)
    return np.clip(sepia_img, 0, 255).astype(np.uint8)


def histogram_overlay(frame: np.ndarray) -> np.ndarray:
    """Draw live per-channel histograms in the top-right corner."""
    out = frame.copy()
    h, w = out.shape[:2]

    hist_h, hist_w = 160, 256
    # Semi-transparent black background
    overlay_region = out[10 : 10 + hist_h, w - hist_w - 10 : w - 10]
    out[10 : 10 + hist_h, w - hist_w - 10 : w - 10] = (
        overlay_region * 0.3
    ).astype(np.uint8)

    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # B, G, R
    for i, col in enumerate(colours):
        hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, hist_h - 4, cv2.NORM_MINMAX)
        pts = np.column_stack(
            (
                np.arange(256) + w - hist_w - 10,
                (10 + hist_h - 2) - hist.ravel().astype(int),
            )
        )
        cv2.polylines(out, [pts], isClosed=False, color=col, thickness=1,
                      lineType=cv2.LINE_AA)

    cv2.putText(out, "B  G  R", (w - hist_w, 10 + hist_h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


# ── Trackbar callback (no-op, value read in loop) ───────────────────
def _noop(_):
    pass


# ── Main loop ────────────────────────────────────────────────────────

def main() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Resolution: {width}x{height}")

    cv2.namedWindow(WINDOW)

    # Trackbars for brightness / contrast (visible in mode 3)
    cv2.createTrackbar("Brightness", WINDOW, 100, 200, _noop)  # 100 = neutral
    cv2.createTrackbar("Contrast", WINDOW, 0, 200, _noop)

    mode = 1
    frame_count = 0
    fps_start = time.perf_counter()
    live_fps = 0.0

    print(f"\n[MODE {mode}] {MODE_NAMES[mode]}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame grab failed.")
            break

        # ── Apply active mode ────────────────────────────────────────
        if mode == 1:
            display = frame.copy()
        elif mode == 2:
            display = channel_swap(frame)
        elif mode == 3:
            bright = cv2.getTrackbarPos("Brightness", WINDOW) - 100
            contr = cv2.getTrackbarPos("Contrast", WINDOW)
            display = adjust_brightness_contrast(frame, bright, contr)
        elif mode == 4:
            display = pixelate_roi(frame)
        elif mode == 5:
            display = sepia(frame)
        elif mode == 6:
            display = histogram_overlay(frame)
        else:
            display = frame.copy()

        # ── OSD ──────────────────────────────────────────────────────
        label = f"[{mode}] {MODE_NAMES[mode]}  |  FPS: {live_fps:.1f}"
        cv2.putText(display, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(display, "Keys: 1-6 or Arrows  |  q = quit", (10, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(WINDOW, display)

        # ── FPS ──────────────────────────────────────────────────────
        frame_count += 1
        elapsed = time.perf_counter() - fps_start
        if elapsed >= 1.0:
            live_fps = frame_count / elapsed
            frame_count = 0
            fps_start = time.perf_counter()

        # ── Key handling ─────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        new_mode = mode
        if key == ord("q"):
            break
        elif key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6")):
            new_mode = key - ord("0")
        elif key == 3 or key == ord("d"):  # right arrow
            new_mode = mode % 6 + 1
        elif key == 2 or key == ord("a"):  # left arrow
            new_mode = (mode - 2) % 6 + 1

        if new_mode != mode:
            mode = new_mode
            print(f"[MODE {mode}] {MODE_NAMES[mode]}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
