"""
detection.py
============
Face detection and forehead-ROI extraction using MediaPipe Face Mesh.

Provides helpers to:
- Create a Face Mesh detector
- Find the primary (largest) face
- Compute a bounding box from mesh landmarks
- Extract mean BGR from the forehead polygon
- Draw overlays (forehead highlight, bounding box, privacy blur)
"""

import cv2
import mediapipe as mp
import numpy as np

# ── MediaPipe aliases ────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh

# Forehead landmark indices (MediaPipe 468-point model).
# These trace a patch across the mid-forehead, avoiding eyebrows.
FOREHEAD_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172,  58, 132,  93, 234, 127, 162,  21,
     54, 103,  67, 109,
]


# ── Factory ──────────────────────────────────────────────────────────

def create_face_mesh(
    max_faces: int = 1,
    detection_confidence: float = 0.5,
    tracking_confidence: float = 0.5,
) -> mp_face_mesh.FaceMesh:
    """Return a configured MediaPipe FaceMesh instance (tracking mode)."""
    return mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=max_faces,
        refine_landmarks=True,
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence,
    )


# ── Face queries ─────────────────────────────────────────────────────

def get_primary_face(results, img_w: int, img_h: int):
    """
    Return the landmarks of the largest detected face, or *None*.
    """
    if not results.multi_face_landmarks:
        return None

    best = None
    best_area = -1.0
    for face_lm in results.multi_face_landmarks:
        xs = [lm.x * img_w for lm in face_lm.landmark]
        ys = [lm.y * img_h for lm in face_lm.landmark]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area > best_area:
            best_area = area
            best = face_lm
    return best


def face_bbox(landmarks, img_w: int, img_h: int,
              padding: int = 20) -> tuple[int, int, int, int]:
    """
    Compute a padded bounding box (x, y, w, h) from mesh landmarks.
    """
    xs = [lm.x * img_w for lm in landmarks.landmark]
    ys = [lm.y * img_h for lm in landmarks.landmark]
    x1 = max(0, int(min(xs)) - padding)
    y1 = max(0, int(min(ys)) - padding)
    x2 = min(img_w, int(max(xs)) + padding)
    y2 = min(img_h, int(max(ys)) + padding)
    return x1, y1, x2 - x1, y2 - y1


# ── ROI extraction ──────────────────────────────────────────────────

def _forehead_hull(landmarks, img_w: int, img_h: int) -> np.ndarray | None:
    """Return the convex hull of the forehead landmarks, or None."""
    pts = np.array(
        [(int(landmarks.landmark[i].x * img_w),
          int(landmarks.landmark[i].y * img_h))
         for i in FOREHEAD_IDX if i < len(landmarks.landmark)],
        dtype=np.int32,
    )
    if len(pts) < 3:
        return None
    return cv2.convexHull(pts)


def forehead_roi_mean(
    frame_bgr: np.ndarray,
    landmarks,
    img_w: int,
    img_h: int,
) -> np.ndarray | None:
    """
    Return the mean (B, G, R) values inside the forehead polygon.

    Uses a convex-hull mask over FOREHEAD_IDX.
    Returns None if the ROI is degenerate.
    """
    hull = _forehead_hull(landmarks, img_w, img_h)
    if hull is None:
        return None

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    mean_bgr = cv2.mean(frame_bgr, mask=mask)[:3]
    return np.array(mean_bgr)


# ── Drawing helpers ──────────────────────────────────────────────────

def draw_forehead_overlay(frame: np.ndarray, landmarks,
                          w: int, h: int) -> None:
    """Draw a translucent green overlay on the forehead region."""
    hull = _forehead_hull(landmarks, w, h)
    if hull is None:
        return
    overlay = frame.copy()
    cv2.fillConvexPoly(overlay, hull, (0, 200, 80))
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.polylines(frame, [hull], True, (0, 255, 0), 1, cv2.LINE_AA)


def draw_face_bbox(frame: np.ndarray, bbox: tuple[int, int, int, int],
                   label: str = "Face",
                   colour: tuple[int, int, int] = (0, 255, 0)) -> None:
    """Draw a labelled rectangle around a face."""
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
    cv2.putText(frame, label, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, colour, 1, cv2.LINE_AA)


def pixelate_face(frame: np.ndarray, bbox: tuple[int, int, int, int],
                  block: int = 10) -> None:
    """Pixelate a rectangular region **in-place**."""
    x, y, w, h = bbox
    roi = frame[y:y + h, x:x + w]
    if roi.size == 0:
        return
    small = cv2.resize(
        roi,
        (max(1, w // block), max(1, h // block)),
        interpolation=cv2.INTER_LINEAR,
    )
    frame[y:y + h, x:x + w] = cv2.resize(
        small, (w, h), interpolation=cv2.INTER_NEAREST,
    )
