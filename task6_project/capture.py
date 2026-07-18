"""
capture.py
==========
Thin wrapper around cv2.VideoCapture for clean webcam management.

Provides open, read, info, and release helpers so the main loop
stays focused on application logic rather than camera plumbing.
"""

import cv2
import numpy as np


def open_camera(index: int = 0) -> cv2.VideoCapture:
    """
    Open a webcam by *index* and return the VideoCapture object.

    Raises RuntimeError if the camera cannot be opened.
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"[ERROR] Cannot open camera at index {index}. "
            "Check your camera connection."
        )
    return cap


def get_camera_info(cap: cv2.VideoCapture) -> dict:
    """
    Return a dict with width, height, and nominal FPS from *cap*.
    """
    return {
        "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":    cap.get(cv2.CAP_PROP_FPS) or 30.0,
    }


def read_frame(cap: cv2.VideoCapture) -> tuple[bool, np.ndarray | None]:
    """
    Read a single frame.  Returns (success, frame).
    """
    return cap.read()


def release(cap: cv2.VideoCapture) -> None:
    """Release the VideoCapture resource."""
    if cap is not None:
        cap.release()
