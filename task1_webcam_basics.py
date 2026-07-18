"""
task1_webcam_basics.py
======================
Opens the default webcam and displays a live feed.

Keyboard shortcuts (press while the video window is focused):
    g  – toggle grayscale mode
    e  – toggle Canny edge-detection mode
    s  – save the current frame as a timestamped PNG snapshot
    q  – quit

Resolution and FPS are logged to the console on startup and
continuously updated every second.
"""

import cv2
import time
import os

# ── Configuration ────────────────────────────────────────────────────
SNAPSHOT_DIR = "snapshots"
CAMERA_INDEX = 0  # change if you have multiple cameras


def main() -> None:
    # Ensure snapshot directory exists
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check your camera connection.")
        return

    # Read and log native resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Resolution : {width} x {height}")
    print(f"[INFO] Camera FPS : {reported_fps:.1f}")
    print()
    print("[CONTROLS]")
    print("  g  – toggle grayscale")
    print("  e  – toggle edge detection")
    print("  s  – save snapshot")
    print("  q  – quit")
    print()

    # ── State flags ──────────────────────────────────────────────────
    grayscale_on = False
    edge_detect_on = False

    # ── FPS tracking ─────────────────────────────────────────────────
    frame_count = 0
    fps_start = time.perf_counter()
    live_fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        # ── Apply filters ────────────────────────────────────────────
        display = frame.copy()

        if grayscale_on:
            display = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
            # Convert back to 3-channel so edge overlay and OSD work uniformly
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

        if edge_detect_on:
            gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, threshold1=80, threshold2=160)
            display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # ── On-screen info overlay ───────────────────────────────────
        status_parts = []
        if grayscale_on:
            status_parts.append("GRAY")
        if edge_detect_on:
            status_parts.append("EDGE")
        status_text = " | ".join(status_parts) if status_parts else "NORMAL"

        cv2.putText(
            display,
            f"{width}x{height}  FPS: {live_fps:.1f}  [{status_text}]",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Webcam – Task 1", display)

        # ── FPS calculation (updated every second) ───────────────────
        frame_count += 1
        elapsed = time.perf_counter() - fps_start
        if elapsed >= 1.0:
            live_fps = frame_count / elapsed
            print(f"[FPS] {live_fps:.1f}  |  Resolution: {width}x{height}")
            frame_count = 0
            fps_start = time.perf_counter()

        # ── Key handling ─────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[INFO] Quitting.")
            break

        elif key == ord("g"):
            grayscale_on = not grayscale_on
            print(f"[TOGGLE] Grayscale {'ON' if grayscale_on else 'OFF'}")

        elif key == ord("e"):
            edge_detect_on = not edge_detect_on
            print(f"[TOGGLE] Edge detection {'ON' if edge_detect_on else 'OFF'}")

        elif key == ord("s"):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(SNAPSHOT_DIR, f"snapshot_{timestamp}.png")
            cv2.imwrite(filename, frame)  # always save the original frame
            print(f"[SNAPSHOT] Saved → {filename}")

    # ── Cleanup ──────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
