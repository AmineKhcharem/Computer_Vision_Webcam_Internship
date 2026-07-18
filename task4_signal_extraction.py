"""
task4_signal_extraction.py
==========================
Real-time signal extraction from a facial ROI, with bandpass
filtering and live plotting of raw vs filtered signals.

The script detects the primary face (via MediaPipe Face Mesh),
extracts the average pixel intensity of the *forehead* region each
frame, stores it in a rolling buffer (default 10 s), applies a
Butterworth bandpass filter, and plots raw vs filtered signals.

Two plotting modes (toggle with 'p'):
    LIVE   – matplotlib figure updates every frame in the main loop
    END    – plot is shown only when you quit (faster main loop)

Keyboard shortcuts (press while the video window is focused):
    p  – toggle between LIVE plot and END-of-session plot
    c  – cycle signal channel  (Green → Red → Blue → Luminance)
    q  – quit  (shows end-of-session plot if in END mode)

Requirements:
    pip install opencv-python mediapipe matplotlib scipy numpy

Why the forehead?
    The forehead has relatively thin skin with superficial blood vessels
    and minimal muscle movement, making it one of the steadiest facial
    sub-regions for extracting a colour-change signal (e.g. rPPG).

Filtering:
    A 4th-order Butterworth bandpass filter is applied.  Default
    passband is 0.7 – 4.0 Hz (≈42 – 240 BPM), chosen to span the
    plausible range of a resting-to-exercise human heart rate.  The
    filter requires a known sample rate, which is estimated from the
    actual measured FPS.

Author notes:
    See NOTES.md for robustness observations (movement, lighting, etc.)
"""

import cv2
import mediapipe as mp
import numpy as np
import matplotlib
matplotlib.use("TkAgg")            # interactive backend (non-blocking)
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
import time
import collections

# ── Configuration ────────────────────────────────────────────────────
CAMERA_INDEX   = 0
BUFFER_SECONDS = 10                # rolling buffer length
MAX_FACES      = 1
BANDPASS_LO    = 0.7               # Hz  (lower cutoff)
BANDPASS_HI    = 4.0               # Hz  (upper cutoff)
FILTER_ORDER   = 4                 # Butterworth order

WINDOW_MAIN    = "Signal Extraction - Task 4"

# MediaPipe aliases
mp_face_mesh   = mp.solutions.face_mesh
mp_drawing     = mp.solutions.drawing_utils

# Forehead landmark indices (MediaPipe Face Mesh 468-point model)
# These trace a patch across the mid-forehead, avoiding eyebrows.
FOREHEAD_IDX = [10, 338, 297, 332, 284, 251, 389, 356,
                454, 323, 361, 288, 397, 365, 379, 378,
                400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21,
                54, 103, 67, 109]

# Channel labels for cycling
CHANNEL_NAMES = ["Green", "Red", "Blue", "Luminance"]
CHANNEL_MAP   = {"Green": 1, "Red": 2, "Blue": 0, "Luminance": -1}


# ── Helpers ──────────────────────────────────────────────────────────

def forehead_roi_mean(frame_bgr: np.ndarray, landmarks,
                      img_w: int, img_h: int) -> np.ndarray | None:
    """
    Return the mean (B, G, R) values inside the forehead polygon.

    Uses a convex-hull mask over the FOREHEAD_IDX landmark subset.
    Returns None if the ROI is degenerate.
    """
    pts = np.array(
        [(int(landmarks.landmark[i].x * img_w),
          int(landmarks.landmark[i].y * img_h))
         for i in FOREHEAD_IDX if i < len(landmarks.landmark)],
        dtype=np.int32,
    )
    if len(pts) < 3:
        return None

    hull = cv2.convexHull(pts)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    mean_bgr = cv2.mean(frame_bgr, mask=mask)[:3]  # (B, G, R)
    return np.array(mean_bgr)


def get_primary_face(results, img_w: int, img_h: int):
    """Return the landmarks of the largest detected face, or None."""
    if not results.multi_face_landmarks:
        return None

    best = None
    best_area = -1
    for face_lm in results.multi_face_landmarks:
        xs = [lm.x * img_w for lm in face_lm.landmark]
        ys = [lm.y * img_h for lm in face_lm.landmark]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area > best_area:
            best_area = area
            best = face_lm
    return best


def bandpass_filter(signal: np.ndarray, fs: float,
                    lo: float = BANDPASS_LO,
                    hi: float = BANDPASS_HI,
                    order: int = FILTER_ORDER) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter.

    Returns the filtered signal, or the original signal unchanged if
    the sample rate is too low for the requested passband, or if there
    are too few samples.
    """
    nyq = fs / 2.0
    if nyq <= hi or nyq <= lo or lo >= hi:
        return signal.copy()
    if len(signal) < 3 * (2 * order + 1):
        return signal.copy()

    sos = butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    try:
        return sosfiltfilt(sos, signal)
    except ValueError:
        return signal.copy()


def draw_forehead_overlay(frame: np.ndarray, landmarks,
                          w: int, h: int) -> None:
    """Draw a translucent green overlay on the forehead region."""
    pts = np.array(
        [(int(landmarks.landmark[i].x * w),
          int(landmarks.landmark[i].y * h))
         for i in FOREHEAD_IDX if i < len(landmarks.landmark)],
        dtype=np.int32,
    )
    if len(pts) < 3:
        return
    hull = cv2.convexHull(pts)
    overlay = frame.copy()
    cv2.fillConvexPoly(overlay, hull, (0, 200, 80))
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.polylines(frame, [hull], True, (0, 255, 0), 1, cv2.LINE_AA)


# ── Live plot setup ──────────────────────────────────────────────────

def init_live_plot():
    """Create the matplotlib figure for live plotting."""
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    fig.canvas.manager.set_window_title("Signal Plot - Task 4")
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    line_raw, = ax.plot([], [], color="#7aa2f7", linewidth=1.0,
                        alpha=0.5, label="Raw")
    line_filt, = ax.plot([], [], color="#f7768e", linewidth=1.8,
                         label="Filtered")

    ax.set_xlabel("Time (s)", color="#c0caf5", fontsize=9)
    ax.set_ylabel("Intensity", color="#c0caf5", fontsize=9)
    ax.tick_params(colors="#565f89", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#414868")
    ax.legend(loc="upper right", fontsize=8,
              facecolor="#1e1e2e", edgecolor="#414868",
              labelcolor="#c0caf5")
    fig.tight_layout()
    return fig, ax, line_raw, line_filt


def update_live_plot(fig, ax, line_raw, line_filt,
                     times, raw, filtered, channel_name):
    """Push new data into the live matplotlib plot."""
    if len(times) < 2:
        return
    t = np.array(times) - times[0]
    line_raw.set_data(t, raw)
    line_filt.set_data(t, filtered)
    ax.set_xlim(t[0], max(t[-1], 1.0))

    margin = 5
    lo = min(np.min(raw), np.min(filtered)) - margin
    hi = max(np.max(raw), np.max(filtered)) + margin
    ax.set_ylim(lo, hi)
    ax.set_title(f"Channel: {channel_name}",
                 color="#c0caf5", fontsize=10)

    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def show_end_plot(times, raw, filtered, channel_name):
    """Static plot shown after the recording session ends."""
    if len(times) < 2:
        print("[INFO] Not enough data to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.canvas.manager.set_window_title("Signal Plot (End) - Task 4")
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    t = np.array(times) - times[0]
    ax.plot(t, raw, color="#7aa2f7", linewidth=0.8,
            alpha=0.5, label="Raw")
    ax.plot(t, filtered, color="#f7768e", linewidth=1.6,
            label="Filtered")

    ax.set_xlabel("Time (s)", color="#c0caf5", fontsize=10)
    ax.set_ylabel("Intensity", color="#c0caf5", fontsize=10)
    ax.set_title(f"Recorded Signal — Channel: {channel_name}",
                 color="#c0caf5", fontsize=12)
    ax.tick_params(colors="#565f89", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#414868")
    ax.legend(fontsize=9, facecolor="#1e1e2e", edgecolor="#414868",
              labelcolor="#c0caf5")
    fig.tight_layout()
    plt.show()


# ── Main loop ────────────────────────────────────────────────────────

def main() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.  Check your camera connection.")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nominal_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[INFO] Resolution : {frame_w} × {frame_h}")
    print(f"[INFO] Nominal FPS: {nominal_fps:.1f}")

    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_AUTOSIZE)

    # ── MediaPipe Face Mesh (tracking mode for speed) ────────────
    mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=MAX_FACES,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # ── Rolling buffers ──────────────────────────────────────────
    max_samples = int(BUFFER_SECONDS * nominal_fps) + 1
    buf_time  = collections.deque(maxlen=max_samples)
    buf_bgr   = collections.deque(maxlen=max_samples)   # (B, G, R) tuples

    # ── State ────────────────────────────────────────────────────
    live_plot_mode = True          # True = live,  False = end-of-session
    channel_idx    = 0             # index into CHANNEL_NAMES
    frame_count    = 0
    fps_timer      = time.perf_counter()
    measured_fps   = nominal_fps

    # Live plot handles (created lazily)
    plot_handles = None

    print("\n[CONTROLS]")
    print("  p  – toggle LIVE / END-of-session plot")
    print("  c  – cycle signal channel (G → R → B → Lum)")
    print("  q  – quit\n")
    print(f"[MODE] Live plot: {'ON' if live_plot_mode else 'OFF'}")
    print(f"[CHAN] {CHANNEL_NAMES[channel_idx]}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame grab failed.")
            break

        h, w = frame.shape[:2]
        now = time.perf_counter()

        # ── Face detection ───────────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mesh.process(rgb)
        face_lm = get_primary_face(results, w, h)

        display = frame.copy()
        face_detected = False

        if face_lm is not None:
            face_detected = True

            # Draw forehead overlay on display frame
            draw_forehead_overlay(display, face_lm, w, h)

            # Extract signal
            mean_bgr = forehead_roi_mean(frame, face_lm, w, h)
            if mean_bgr is not None:
                buf_time.append(now)
                buf_bgr.append(mean_bgr)

        # ── Build the selected channel's raw signal ──────────────
        ch_name = CHANNEL_NAMES[channel_idx]
        raw_signal = np.array([], dtype=np.float64)
        filtered_signal = np.array([], dtype=np.float64)

        if len(buf_bgr) > 1:
            bgr_arr = np.array(buf_bgr)             # shape (N, 3)
            if ch_name == "Luminance":
                # ITU-R BT.601 luminance from BGR
                raw_signal = (0.114 * bgr_arr[:, 0] +
                              0.587 * bgr_arr[:, 1] +
                              0.299 * bgr_arr[:, 2])
            else:
                ch_idx = CHANNEL_MAP[ch_name]
                raw_signal = bgr_arr[:, ch_idx].astype(np.float64)

            # Filter using measured FPS
            filtered_signal = bandpass_filter(raw_signal, measured_fps)

        # ── Live plotting ────────────────────────────────────────
        if live_plot_mode and len(raw_signal) > 1:
            if plot_handles is None:
                plot_handles = init_live_plot()
            fig, ax, lr, lf = plot_handles
            update_live_plot(fig, ax, lr, lf,
                             list(buf_time), raw_signal,
                             filtered_signal, ch_name)

        # ── On-screen display ────────────────────────────────────
        plot_tag = "LIVE" if live_plot_mode else "END"
        face_tag = "Face OK" if face_detected else "No face"
        osd = (f"FPS: {measured_fps:.1f}  |  {face_tag}  |  "
               f"Chan: {ch_name}  |  Plot: {plot_tag}  |  "
               f"Buf: {len(buf_bgr)}")
        cv2.putText(display, osd, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1,
                    cv2.LINE_AA)

        if not face_detected:
            cv2.putText(display, "No face detected",
                        (w // 2 - 130, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)

        cv2.putText(display, "p=plot mode  c=channel  q=quit",
                    (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1,
                    cv2.LINE_AA)

        cv2.imshow(WINDOW_MAIN, display)

        # ── FPS measurement ──────────────────────────────────────
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

        elif key == ord("p"):
            live_plot_mode = not live_plot_mode
            print(f"[MODE] Plot: {'LIVE' if live_plot_mode else 'END-of-session'}")
            if not live_plot_mode and plot_handles is not None:
                plt.close(plot_handles[0])
                plot_handles = None

        elif key == ord("c"):
            channel_idx = (channel_idx + 1) % len(CHANNEL_NAMES)
            print(f"[CHAN] {CHANNEL_NAMES[channel_idx]}")

    # ── Cleanup ──────────────────────────────────────────────────
    mesh.close()
    cap.release()
    cv2.destroyAllWindows()

    if plot_handles is not None:
        plt.close(plot_handles[0])

    # End-of-session plot
    if not live_plot_mode and len(buf_bgr) > 1:
        ch_name = CHANNEL_NAMES[channel_idx]
        bgr_arr = np.array(buf_bgr)
        if ch_name == "Luminance":
            raw_signal = (0.114 * bgr_arr[:, 0] +
                          0.587 * bgr_arr[:, 1] +
                          0.299 * bgr_arr[:, 2])
        else:
            raw_signal = bgr_arr[:, CHANNEL_MAP[ch_name]].astype(np.float64)
        filtered_signal = bandpass_filter(raw_signal, measured_fps)
        show_end_plot(list(buf_time), raw_signal, filtered_signal, ch_name)


if __name__ == "__main__":
    main()
