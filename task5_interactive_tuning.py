"""
task5_interactive_tuning.py
===========================
Live webcam feed with a trackbar control panel for real-time parameter
tuning across the image-processing pipeline.

Trackbar parameters (7 total, spanning 5 pipeline stages):
    ┌─ Brightness / Contrast ──────────────────────────────────┐
    │  Brightness   -100 … +100  (slider 0-200, 100 = neutral) │
    │  Contrast      0.0 … 3.0  (slider 0-300, 100 = 1.0×)    │
    ├─ Smoothing ──────────────────────────────────────────────┤
    │  Blur Radius   0-30  (kernel = 2r+1; 0 = off)            │
    ├─ Edge Detection ─────────────────────────────────────────┤
    │  Canny Lo      0-255                                      │
    │  Canny Hi      0-255                                      │
    │  (set both to 0 to disable edge overlay)                  │
    ├─ Pixelation ─────────────────────────────────────────────┤
    │  Pixelate      1-50  (1 = off; higher = coarser blocks)  │
    ├─ Colour Filter ──────────────────────────────────────────┤
    │  Sepia %       0-100  (0 = off; 100 = full sepia)         │
    └──────────────────────────────────────────────────────────┘

Keyboard shortcuts:
    s  – save current trackbar values to  tuning_config.json
    l  – reload values from  tuning_config.json
    r  – reset all trackbars to defaults
    q  – quit

The config file is auto-loaded on startup if it exists.
"""

import cv2
import numpy as np
import json
import os
import time

# ── Paths & constants ────────────────────────────────────────────────
CAMERA_INDEX = 0
CONFIG_FILE = "tuning_config.json"
WINDOW_FEED = "Tuned Feed - Task 5"
WINDOW_CTRL = "Controls"

# Trackbar definitions: (name, default, maximum)
# Actual parameter value = slider_value + offset  (computed in code)
TRACKBAR_DEFS: list[tuple[str, int, int]] = [
    ("Brightness",  100,  200),   # 100 → 0 offset (range -100..+100)
    ("Contrast",    100,  300),   # 100 → 1.0× factor (range 0.0..3.0)
    ("Blur Radius",   0,   30),   # kernel = 2r+1;  0 = disabled
    ("Canny Lo",      0,  255),
    ("Canny Hi",      0,  255),
    ("Pixelate",      1,   50),   # 1 = off
    ("Sepia %",       0,  100),
]

DEFAULTS = {name: default for name, default, _ in TRACKBAR_DEFS}

# Sepia colour-transform kernel
SEPIA_KERNEL = np.array(
    [[0.272, 0.534, 0.131],
     [0.349, 0.686, 0.168],
     [0.393, 0.769, 0.189]],
    dtype=np.float32,
)


# ── Config persistence ──────────────────────────────────────────────

def save_config(values: dict[str, int], path: str = CONFIG_FILE) -> None:
    """Write current trackbar values to a JSON file."""
    with open(path, "w") as f:
        json.dump(values, f, indent=2)
    print(f"[SAVE] Config written → {os.path.abspath(path)}")


def load_config(path: str = CONFIG_FILE) -> dict[str, int] | None:
    """Read trackbar values from a JSON file.  Returns None on failure."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        print(f"[LOAD] Config loaded ← {os.path.abspath(path)}")
        return data
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[WARN] Could not load config: {exc}")
        return None


# ── Trackbar helpers ─────────────────────────────────────────────────

def _noop(_: int) -> None:
    """No-op callback; values are polled in the main loop."""


def create_trackbars(window: str) -> None:
    """Create all trackbars on *window* with their default values."""
    for name, default, maximum in TRACKBAR_DEFS:
        cv2.createTrackbar(name, window, default, maximum, _noop)


def read_trackbars(window: str) -> dict[str, int]:
    """Return the current value of every trackbar as a dict."""
    return {name: cv2.getTrackbarPos(name, window)
            for name, _, _ in TRACKBAR_DEFS}


def apply_trackbars(window: str, values: dict[str, int]) -> None:
    """Push a dict of values back into the trackbars (e.g. after load)."""
    for name, _, _ in TRACKBAR_DEFS:
        if name in values:
            cv2.setTrackbarPos(name, window, int(values[name]))


# ── Image-processing pipeline ───────────────────────────────────────

def process_frame(frame: np.ndarray, p: dict[str, int]) -> np.ndarray:
    """
    Apply the full pipeline to *frame* using parameter dict *p*.
    Returns a new image; the original is not modified.
    """
    out = frame.copy()

    # 1 ── Brightness & Contrast ──────────────────────────────────
    beta = p["Brightness"] - 100          # -100 … +100
    alpha = p["Contrast"] / 100.0         #  0.0 …  3.0
    if beta != 0 or alpha != 1.0:
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    # 2 ── Gaussian blur ──────────────────────────────────────────
    r = p["Blur Radius"]
    if r > 0:
        k = 2 * r + 1  # must be odd
        out = cv2.GaussianBlur(out, (k, k), 0)

    # 3 ── Canny edge overlay ─────────────────────────────────────
    lo, hi = p["Canny Lo"], p["Canny Hi"]
    if lo > 0 or hi > 0:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, lo, hi)
        # Green edge overlay on top of the current image
        edge_bgr = np.zeros_like(out)
        edge_bgr[:, :, 1] = edges  # green channel
        out = cv2.addWeighted(out, 0.7, edge_bgr, 1.0, 0)

    # 4 ── Pixelation ─────────────────────────────────────────────
    blk = p["Pixelate"]
    if blk > 1:
        h, w = out.shape[:2]
        small = cv2.resize(
            out,
            (max(1, w // blk), max(1, h // blk)),
            interpolation=cv2.INTER_LINEAR,
        )
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    # 5 ── Sepia ──────────────────────────────────────────────────
    sp = p["Sepia %"]
    if sp > 0:
        sepia = cv2.transform(out, SEPIA_KERNEL)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)
        blend = sp / 100.0
        out = cv2.addWeighted(sepia, blend, out, 1.0 - blend, 0)

    return out


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Resolution: {width}×{height}")

    # ── Create windows ───────────────────────────────────────────
    cv2.namedWindow(WINDOW_FEED, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(WINDOW_CTRL, cv2.WINDOW_AUTOSIZE)

    # A small blank image for the control window (trackbars sit on it)
    ctrl_bg = np.zeros((1, 480, 3), dtype=np.uint8)
    cv2.imshow(WINDOW_CTRL, ctrl_bg)

    create_trackbars(WINDOW_CTRL)

    # Auto-load saved config if it exists
    saved = load_config()
    if saved:
        apply_trackbars(WINDOW_CTRL, saved)

    # ── FPS state ────────────────────────────────────────────────
    frame_count = 0
    fps_timer = time.perf_counter()
    live_fps = 0.0

    print("\n[CONTROLS]")
    print("  s – save config      l – load config")
    print("  r – reset defaults   q – quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame grab failed.")
            break

        # Read current trackbar values
        params = read_trackbars(WINDOW_CTRL)

        # Process
        display = process_frame(frame, params)

        # ── OSD ──────────────────────────────────────────────────
        h, w = display.shape[:2]
        info_lines = [
            f"FPS: {live_fps:.1f}",
            f"Bright {params['Brightness']-100:+d}  "
            f"Contr {params['Contrast']/100:.1f}x  "
            f"Blur {params['Blur Radius']}",
            f"Canny {params['Canny Lo']}/{params['Canny Hi']}  "
            f"Pix {params['Pixelate']}  "
            f"Sepia {params['Sepia %']}%",
        ]
        y0 = 24
        for i, line in enumerate(info_lines):
            cv2.putText(display, line, (10, y0 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1,
                        cv2.LINE_AA)

        cv2.putText(display, "s=save  l=load  r=reset  q=quit",
                    (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1,
                    cv2.LINE_AA)

        cv2.imshow(WINDOW_FEED, display)

        # ── FPS ──────────────────────────────────────────────────
        frame_count += 1
        elapsed = time.perf_counter() - fps_timer
        if elapsed >= 1.0:
            live_fps = frame_count / elapsed
            frame_count = 0
            fps_timer = time.perf_counter()

        # ── Key handling ─────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[INFO] Quitting.")
            break

        elif key == ord("s"):
            save_config(read_trackbars(WINDOW_CTRL))

        elif key == ord("l"):
            loaded = load_config()
            if loaded:
                apply_trackbars(WINDOW_CTRL, loaded)

        elif key == ord("r"):
            apply_trackbars(WINDOW_CTRL, DEFAULTS)
            print("[RESET] All trackbars restored to defaults.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
