"""
signal_processing.py
====================
Rolling signal buffer and bandpass filtering for live ROI signals.

Provides:
- SignalBuffer  – deque-backed store for timestamped BGR samples
- bandpass_filter() – Butterworth zero-phase filter via sosfiltfilt
- Channel name/index constants
"""

import collections
import numpy as np
from scipy.signal import butter, sosfiltfilt

# ── Channel constants ────────────────────────────────────────────────
CHANNEL_NAMES = ["Green", "Red", "Blue", "Luminance"]
CHANNEL_MAP   = {"Green": 1, "Red": 2, "Blue": 0, "Luminance": -1}


# ── Signal buffer ───────────────────────────────────────────────────

class SignalBuffer:
    """
    Fixed-capacity rolling buffer for timestamped BGR-mean samples.

    Parameters
    ----------
    max_samples : int
        Maximum number of samples to retain (oldest are discarded).
    """

    def __init__(self, max_samples: int = 300):
        self._times = collections.deque(maxlen=max_samples)
        self._bgr   = collections.deque(maxlen=max_samples)

    # ── Mutators ─────────────────────────────────────────────────

    def append(self, timestamp: float, bgr_mean: np.ndarray) -> None:
        """Add a (timestamp, BGR-mean) sample to the buffer."""
        self._times.append(timestamp)
        self._bgr.append(bgr_mean)

    def clear(self) -> None:
        """Discard all samples."""
        self._times.clear()
        self._bgr.clear()

    # ── Accessors ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._times)

    def get_times(self) -> np.ndarray:
        """Return timestamps as a 1-D float64 array."""
        return np.array(self._times, dtype=np.float64)

    def get_channel(self, name: str) -> np.ndarray:
        """
        Extract a single-channel signal by *name*.

        Valid names: "Green", "Red", "Blue", "Luminance".
        Returns a 1-D float64 array of length len(self).
        """
        if len(self._bgr) == 0:
            return np.array([], dtype=np.float64)

        bgr = np.array(self._bgr)  # (N, 3)
        if name == "Luminance":
            # ITU-R BT.601
            return (0.114 * bgr[:, 0] +
                    0.587 * bgr[:, 1] +
                    0.299 * bgr[:, 2])

        idx = CHANNEL_MAP.get(name)
        if idx is None or idx < 0:
            return np.zeros(len(bgr), dtype=np.float64)
        return bgr[:, idx].astype(np.float64)


# ── Bandpass filter ──────────────────────────────────────────────────

def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    lo: float = 0.7,
    hi: float = 4.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter.

    Returns the filtered signal.  Falls back to a copy of the input
    when the sample rate is too low for the passband, or when there
    are too few samples for the filter length.
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
