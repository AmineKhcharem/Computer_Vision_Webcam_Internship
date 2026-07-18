"""
task3_face_roi.py
=================
Real-time face detection with landmarks, cropped ROI, privacy blur,
and a detect-vs-track FPS comparison.

Requirements:
    pip install opencv-python mediapipe

Keyboard shortcuts (press while the video window is focused):
    m  – toggle between  DETECT-EVERY-FRAME  and  DETECT+TRACK  modes
    b  – toggle face-blur / pixelation privacy mode
    q  – quit

How the two modes work:
    DETECT/FRAME  – MediaPipe Face Mesh runs full detection on every
                    single frame (static_image_mode=True).  Accurate
                    but heavier.
    DETECT+TRACK  – MediaPipe uses its built-in lightweight tracking
                    between detection frames (static_image_mode=False).
                    Faster because full detection only fires when
                    tracking confidence drops below the threshold.

Both modes return 468 landmarks per face on every frame.  The FPS
overlay lets you compare throughput in real time.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os

# ── Configuration ────────────────────────────────────────────────────
CAMERA_INDEX = 0
MAX_FACES = 5
ROI_WINDOW_SIZE = 220          # px, square
BLUR_BLOCK_SIZE = 10           # pixelation coarseness

WINDOW_MAIN = "Face ROI - Task 3"
WINDOW_ROI = "Face ROI - Cropped"

# MediaPipe aliases
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ── Helpers ──────────────────────────────────────────────────────────

def faces_from_results(results, img_w: int, img_h: int) -> list[dict]:
    """
    Extract a list of face dicts from MediaPipe Face Mesh results.

    Each dict contains:
        bbox       – (x, y, w, h) in pixel coords
        landmarks  – NormalizedLandmarkList (for mp_drawing)
        area       – bounding-box area (used to pick the primary face)

    Returns an empty list when no faces are found.
    """
    faces = []
    if not results.multi_face_landmarks:
        return faces

    for face_lm in results.multi_face_landmarks:
        xs = [lm.x * img_w for lm in face_lm.landmark]
        ys = [lm.y * img_h for lm in face_lm.landmark]
        x1 = max(0, int(min(xs)))
        y1 = max(0, int(min(ys)))
        x2 = min(img_w, int(max(xs)))
        y2 = min(img_h, int(max(ys)))
        faces.append({
            "bbox": (x1, y1, x2 - x1, y2 - y1),
            "landmarks": face_lm,
            "area": (x2 - x1) * (y2 - y1),
        })

    # Largest face first → "primary" face is faces[0]
    faces.sort(key=lambda f: f["area"], reverse=True)
    return faces


def pixelate_region(frame: np.ndarray, x: int, y: int, w: int, h: int,
                    block: int = BLUR_BLOCK_SIZE) -> None:
    """Pixelate a rectangular region **in-place**."""
    roi = frame[y : y + h, x : x + w]
    if roi.size == 0:
        return
    small = cv2.resize(
        roi,
        (max(1, w // block), max(1, h // block)),
        interpolation=cv2.INTER_LINEAR,
    )
    frame[y : y + h, x : x + w] = cv2.resize(
        small, (w, h), interpolation=cv2.INTER_NEAREST
    )


def draw_face_mesh(frame: np.ndarray, landmarks) -> None:
    """Draw the full MediaPipe face-mesh (tesselation + contours)."""
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style(),
    )
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style(),
    )
    # Iris landmarks (refined)
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=landmarks,
        connections=mp_face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style(),
    )


# ── Main loop ────────────────────────────────────────────────────────

def main() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.  Check your camera connection.")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Resolution : {frame_w} × {frame_h}")

    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(WINDOW_ROI, cv2.WINDOW_AUTOSIZE)

    # ── Two Face Mesh instances – one for each mode ──────────────
    #  static_image_mode=True  → full detection on EVERY frame
    #  static_image_mode=False → built-in tracking between detections
    mesh_detect = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=MAX_FACES,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
    mesh_track = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=MAX_FACES,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # ── State ────────────────────────────────────────────────────
    track_mode = False          # False → detect / frame, True → detect + track
    blur_mode = False

    frame_count = 0
    fps_timer = time.perf_counter()
    live_fps = 0.0

    print("\n[CONTROLS]")
    print("  m  – toggle  DETECT/FRAME  ↔  DETECT+TRACK")
    print("  b  – toggle privacy blur")
    print("  q  – quit\n")
    print("[MODE] Detect every frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame grab failed.")
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Run the active pipeline ──────────────────────────────
        mesh = mesh_track if track_mode else mesh_detect
        results = mesh.process(rgb)
        faces = faces_from_results(results, w, h)

        display = frame.copy()
        primary_roi = None

        # ── No faces ─────────────────────────────────────────────
        if not faces:
            cv2.putText(
                display, "No face detected",
                (w // 2 - 120, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2,
            )
        else:
            for idx, face in enumerate(faces):
                bx, by, bw, bh = face["bbox"]
                is_primary = idx == 0

                # ── Privacy blur / pixelation ────────────────────
                if blur_mode:
                    pixelate_region(display, bx, by, bw, bh)
                else:
                    draw_face_mesh(display, face["landmarks"])

                # ── Bounding box + label ─────────────────────────
                colour = (0, 255, 0) if is_primary else (255, 200, 50)
                thickness = 2 if is_primary else 1
                label = "Primary" if is_primary else f"Face {idx + 1}"

                cv2.rectangle(display, (bx, by), (bx + bw, by + bh),
                              colour, thickness)
                cv2.putText(display, label, (bx, by - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, colour, 1)

                # ── Extract primary ROI ──────────────────────────
                if is_primary:
                    pad = 20
                    ry1 = max(0, by - pad)
                    rx1 = max(0, bx - pad)
                    ry2 = min(h, by + bh + pad)
                    rx2 = min(w, bx + bw + pad)
                    primary_roi = frame[ry1:ry2, rx1:rx2].copy()

        # ── ROI window ───────────────────────────────────────────
        if primary_roi is not None and primary_roi.size > 0:
            roi_show = cv2.resize(
                primary_roi,
                (ROI_WINDOW_SIZE, ROI_WINDOW_SIZE),
                interpolation=cv2.INTER_LINEAR,
            )
            if blur_mode:
                roi_show = cv2.GaussianBlur(roi_show, (31, 31), 30)

            n = len(faces)
            count_txt = f"{n} face{'s' if n != 1 else ''}"
            cv2.putText(roi_show, count_txt, (6, ROI_WINDOW_SIZE - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 1)
            cv2.imshow(WINDOW_ROI, roi_show)
        else:
            blank = np.zeros((ROI_WINDOW_SIZE, ROI_WINDOW_SIZE, 3), np.uint8)
            cv2.putText(blank, "No face", (55, ROI_WINDOW_SIZE // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 100, 100), 2)
            cv2.imshow(WINDOW_ROI, blank)

        # ── On-screen display ────────────────────────────────────
        mode_label = "DETECT+TRACK" if track_mode else "DETECT/FRAME"
        blur_label = "  |  BLUR" if blur_mode else ""
        osd_top = f"{mode_label}   FPS: {live_fps:.1f}   Faces: {len(faces)}{blur_label}"
        cv2.putText(display, osd_top, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2)
        cv2.putText(display, "m = mode   b = blur   q = quit",
                    (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        cv2.imshow(WINDOW_MAIN, display)

        # ── FPS counter (updated every second) ───────────────────
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

        elif key == ord("m"):
            track_mode = not track_mode
            tag = "Detect + Track" if track_mode else "Detect every frame"
            print(f"[MODE] {tag}")

        elif key == ord("b"):
            blur_mode = not blur_mode
            print(f"[BLUR] {'ON – faces pixelated' if blur_mode else 'OFF'}")

    # ── Cleanup ──────────────────────────────────────────────────
    mesh_detect.close()
    mesh_track.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
