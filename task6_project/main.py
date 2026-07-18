"""
main.py — Task 6 Integration Project
=====================================
End-to-end real-time application combining:
  • Webcam capture  (capture.py)
  • Face detection & forehead ROI  (detection.py)
  • Signal extraction & bandpass filtering  (signal_processing.py)
  • Side-by-side composite display  (webcam + signal plot)
  • Interactive trackbar tuning  (reused from Task 5)
  • On-screen UI & config persistence

Run:
    cd task6_project
    python main.py

Keyboard shortcuts:
    s  – save trackbar values to task6_config.json
    l  – load trackbar values from task6_config.json
    r  – reset all trackbars to defaults
    b  – toggle face-blur privacy mode
    q  – quit
"""

import cv2
import numpy as np
import json
import os
import time

import matplotlib
matplotlib.use("Agg")                       # off-screen rendering
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# ── Local modules ────────────────────────────────────────────────────
from capture import open_camera, get_camera_info, read_frame, release
from detection import (
    create_face_mesh,
    get_primary_face,
    face_bbox,
    forehead_roi_mean,
    draw_forehead_overlay,
    draw_face_bbox,
    pixelate_face,
)
from signal_processing import (
    SignalBuffer,
    bandpass_filter,
    CHANNEL_NAMES,
)

# ── Constants ────────────────────────────────────────────────────────
CAMERA_INDEX   = 0
BUFFER_SECONDS = 10
CONFIG_FILE    = "task6_config.json"
WINDOW_MAIN    = "CV Pipeline - Task 6"
WINDOW_CTRL    = "Controls"

# Plot dimensions (pixels) — matches the webcam feed height
PLOT_WIDTH  = 480
PLOT_HEIGHT = 360                            # overridden at runtime

# ── Trackbar definitions: (name, default, maximum) ───────────────────
TRACKBAR_DEFS: list[tuple[str, int, int]] = [
    ("Brightness",    100,  200),   # slider 0-200 → offset -100..+100
    ("Contrast",      100,  300),   # slider 0-300 → factor 0.0..3.0
    ("Blur Radius",     0,   30),   # kernel = 2r+1; 0 = off
    ("BP Low x10",      7,   50),   # ×0.1 → 0.1..5.0 Hz
    ("BP High x10",    40,   80),   # ×0.1 → 0.1..8.0 Hz
    ("Filter Order",    4,    6),   # Butterworth order 1..6
    ("Channel",         0,    3),   # 0=Green 1=Red 2=Blue 3=Lum
]
DEFAULTS = {name: default for name, default, _ in TRACKBAR_DEFS}


# ── Config persistence ──────────────────────────────────────────────

def save_config(values: dict[str, int], path: str = CONFIG_FILE) -> None:
    with open(path, "w") as f:
        json.dump(values, f, indent=2)
    print(f"[SAVE] Config → {os.path.abspath(path)}")


def load_config(path: str = CONFIG_FILE) -> dict[str, int] | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[WARN] Config load failed: {exc}")
        return None


# ── Trackbar helpers ─────────────────────────────────────────────────

def _noop(_: int) -> None:
    """No-op callback — values are polled each frame."""


def create_trackbars(window: str) -> None:
    for name, default, maximum in TRACKBAR_DEFS:
        cv2.createTrackbar(name, window, default, maximum, _noop)


def read_trackbars(window: str) -> dict[str, int]:
    return {name: cv2.getTrackbarPos(name, window)
            for name, _, _ in TRACKBAR_DEFS}


def apply_trackbars(window: str, values: dict[str, int]) -> None:
    for name, _, _ in TRACKBAR_DEFS:
        if name in values:
            cv2.setTrackbarPos(name, window, int(values[name]))


# ── Image processing (brightness / contrast / blur) ─────────────────

def process_frame(frame: np.ndarray, params: dict[str, int]) -> np.ndarray:
    """Apply brightness, contrast, and blur adjustments."""
    out = frame.copy()

    beta  = params["Brightness"] - 100
    alpha = params["Contrast"] / 100.0
    if beta != 0 or alpha != 1.0:
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    r = params["Blur Radius"]
    if r > 0:
        k = 2 * r + 1
        out = cv2.GaussianBlur(out, (k, k), 0)

    return out


# ── Signal plot rendering (Agg → NumPy) ─────────────────────────────

class PlotRenderer:
    """
    Renders raw-vs-filtered signal into a NumPy BGR image using the
    matplotlib Agg backend (no GUI window — purely off-screen).
    """

    def __init__(self, width_px: int, height_px: int, dpi: int = 100):
        self.dpi = dpi
        w_in = width_px / dpi
        h_in = height_px / dpi
        self.fig = Figure(figsize=(w_in, h_in), dpi=dpi)
        self.fig.patch.set_facecolor("#1e1e2e")
        self.canvas = FigureCanvasAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._style_ax()

        self.line_raw, = self.ax.plot(
            [], [], color="#7aa2f7", linewidth=1.0, alpha=0.5, label="Raw",
        )
        self.line_filt, = self.ax.plot(
            [], [], color="#f7768e", linewidth=1.8, label="Filtered",
        )
        self.ax.legend(
            loc="upper right", fontsize=7,
            facecolor="#1e1e2e", edgecolor="#414868", labelcolor="#c0caf5",
        )
        self.fig.tight_layout(pad=1.0)

    def _style_ax(self) -> None:
        ax = self.ax
        ax.set_facecolor("#1e1e2e")
        ax.set_xlabel("Time (s)", color="#c0caf5", fontsize=8)
        ax.set_ylabel("Intensity", color="#c0caf5", fontsize=8)
        ax.tick_params(colors="#565f89", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#414868")

    def render(self, times: np.ndarray, raw: np.ndarray,
               filtered: np.ndarray, channel: str) -> np.ndarray:
        """
        Update plot data and return a BGR image (H × W × 3, uint8).
        """
        if len(times) < 2:
            # Return a blank dark frame when there's no data yet
            self.canvas.draw()
            buf = np.frombuffer(self.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(self.canvas.get_width_height()[::-1] + (4,))
            return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)

        t = times - times[0]
        self.line_raw.set_data(t, raw)
        self.line_filt.set_data(t, filtered)

        self.ax.set_xlim(t[0], max(t[-1], 1.0))
        margin = 5
        lo = min(np.min(raw), np.min(filtered)) - margin
        hi = max(np.max(raw), np.max(filtered)) + margin
        self.ax.set_ylim(lo, hi)
        self.ax.set_title(
            f"Channel: {channel}", color="#c0caf5", fontsize=9,
        )

        self.canvas.draw()
        buf = np.frombuffer(self.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(self.canvas.get_width_height()[::-1] + (4,))
        return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)


# ── OSD (on-screen display) helpers ──────────────────────────────────

def draw_osd_top(frame: np.ndarray, text: str) -> None:
    """Draw a text bar at the top with a semi-transparent background."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (30, 30, 40), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, text, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 180), 1,
                cv2.LINE_AA)


def draw_osd_bottom(frame: np.ndarray, text: str) -> None:
    """Draw a text bar at the bottom with a semi-transparent background."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 28), (w, h), (30, 30, 40), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, text, (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1,
                cv2.LINE_AA)


def draw_no_face(frame: np.ndarray) -> None:
    """Centred 'No face detected' warning."""
    h, w = frame.shape[:2]
    cv2.putText(frame, "No face detected",
                (w // 2 - 120, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 0, 255), 2)


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    # ── Camera ───────────────────────────────────────────────────
    try:
        cap = open_camera(CAMERA_INDEX)
    except RuntimeError as e:
        print(e)
        return

    info = get_camera_info(cap)
    frame_w, frame_h = info["width"], info["height"]
    nominal_fps = info["fps"]
    print(f"[INFO] Resolution : {frame_w} × {frame_h}")
    print(f"[INFO] Nominal FPS: {nominal_fps:.1f}")

    # ── Face Mesh ────────────────────────────────────────────────
    mesh = create_face_mesh(max_faces=1)

    # ── Signal buffer ────────────────────────────────────────────
    max_samples = int(BUFFER_SECONDS * nominal_fps) + 1
    sig_buf = SignalBuffer(max_samples=max_samples)

    # ── Plot renderer (matches webcam height) ────────────────────
    plot_h = frame_h
    plot_w = PLOT_WIDTH
    renderer = PlotRenderer(plot_w, plot_h, dpi=100)

    # ── OpenCV windows ───────────────────────────────────────────
    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(WINDOW_CTRL, cv2.WINDOW_AUTOSIZE)

    ctrl_bg = np.zeros((1, 480, 3), dtype=np.uint8)
    cv2.imshow(WINDOW_CTRL, ctrl_bg)
    create_trackbars(WINDOW_CTRL)

    # Auto-load saved config
    saved = load_config()
    if saved:
        apply_trackbars(WINDOW_CTRL, saved)
        print("[LOAD] Config loaded on startup.")

    # ── State ────────────────────────────────────────────────────
    blur_mode    = False
    frame_count  = 0
    fps_timer    = time.perf_counter()
    measured_fps = nominal_fps

    print("\n[CONTROLS]")
    print("  s = save config   l = load config   r = reset defaults")
    print("  b = toggle blur   q = quit\n")

    while True:
        ok, frame = read_frame(cap)
        if not ok:
            print("[ERROR] Frame grab failed.")
            break

        now = time.perf_counter()
        h, w = frame.shape[:2]

        # ── Read trackbar state ──────────────────────────────────
        params = read_trackbars(WINDOW_CTRL)
        bp_lo    = params["BP Low x10"]  * 0.1
        bp_hi    = params["BP High x10"] * 0.1
        filt_ord = max(1, params["Filter Order"])
        ch_idx   = params["Channel"]
        ch_name  = CHANNEL_NAMES[min(ch_idx, len(CHANNEL_NAMES) - 1)]

        # ── Image pre-processing ─────────────────────────────────
        display = process_frame(frame, params)

        # ── Face detection ───────────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mesh.process(rgb)
        face_lm = get_primary_face(results, w, h)

        face_ok = False
        if face_lm is not None:
            face_ok = True
            bbox = face_bbox(face_lm, w, h, padding=15)

            if blur_mode:
                pixelate_face(display, bbox)
            else:
                draw_forehead_overlay(display, face_lm, w, h)

            draw_face_bbox(display, bbox, "Face")

            mean_bgr = forehead_roi_mean(frame, face_lm, w, h)
            if mean_bgr is not None:
                sig_buf.append(now, mean_bgr)

        if not face_ok:
            draw_no_face(display)

        # ── Signal & filtering ───────────────────────────────────
        raw_sig  = sig_buf.get_channel(ch_name)
        times    = sig_buf.get_times()
        filt_sig = bandpass_filter(raw_sig, measured_fps,
                                   lo=bp_lo, hi=bp_hi, order=filt_ord)

        # ── Render plot to image ─────────────────────────────────
        plot_img = renderer.render(times, raw_sig, filt_sig, ch_name)
        # Resize plot image to match webcam frame height exactly
        plot_img = cv2.resize(plot_img, (plot_w, h),
                              interpolation=cv2.INTER_LINEAR)

        # ── On-screen display ────────────────────────────────────
        face_tag = "Face OK" if face_ok else "No face"
        blur_tag = " | BLUR" if blur_mode else ""
        top_txt = (f"FPS: {measured_fps:.1f}  |  {face_tag}  |  "
                   f"Chan: {ch_name}  |  "
                   f"BP: {bp_lo:.1f}-{bp_hi:.1f} Hz  |  "
                   f"Buf: {len(sig_buf)}{blur_tag}")
        draw_osd_top(display, top_txt)
        draw_osd_bottom(display, "s=save  l=load  r=reset  b=blur  q=quit")

        # ── Composite side-by-side ───────────────────────────────
        composite = np.hstack([display, plot_img])
        cv2.imshow(WINDOW_MAIN, composite)

        # ── FPS ──────────────────────────────────────────────────
        frame_count += 1
        elapsed = time.perf_counter() - fps_timer
        if elapsed >= 1.0:
            measured_fps = frame_count / elapsed
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
            print("[RESET] Trackbars restored to defaults.")

        elif key == ord("b"):
            blur_mode = not blur_mode
            print(f"[BLUR] {'ON' if blur_mode else 'OFF'}")

    # ── Cleanup ──────────────────────────────────────────────────
    mesh.close()
    release(cap)
    cv2.destroyAllWindows()
    plt.close("all")


if __name__ == "__main__":
    main()
