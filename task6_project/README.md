# Task 6: Integrated CV Application

This is the capstone project for the internship, integrating webcam capture, face tracking, signal extraction, filtering, and live tuning into a single modular application.

## Modules

* **`main.py`**: The entry point. Handles the main loop, composites the side-by-side display (webcam feed + matplotlib plot), manages the trackbar UI, and handles config persistence.
* **`capture.py`**: A thin wrapper around `cv2.VideoCapture` for cleaner webcam management.
* **`detection.py`**: Encapsulates MediaPipe Face Mesh logic, primary face selection, ROI (forehead) extraction, and drawing helpers.
* **`signal_processing.py`**: Contains the `SignalBuffer` class for managing the rolling time/BGR data and the `bandpass_filter` function.

## How to Run

From the root of the repository:

```bash
cd task6_project
python main.py
```

## Interactive Controls

The application opens two windows: the main display and a "Controls" window.

### Trackbars

| Trackbar | Purpose |
| :--- | :--- |
| **Brightness** | Adjust overall image brightness (-100 to +100). |
| **Contrast** | Adjust image contrast multiplier (0.0x to 3.0x). |
| **Blur Radius** | Apply a Gaussian blur (0 = off). |
| **BP Low x10** | Lower cutoff for the bandpass filter (0.1 - 5.0 Hz). |
| **BP High x10** | Upper cutoff for the bandpass filter (0.1 - 8.0 Hz). |
| **Filter Order** | Butterworth filter order (1 - 6). |
| **Channel** | Select signal channel (0=Green, 1=Red, 2=Blue, 3=Luminance). |

### Keyboard Shortcuts (Active when main window is focused)

| Key | Action |
| :--- | :--- |
| `s` | Save current trackbar values to `task6_config.json`. |
| `l` | Load trackbar values from `task6_config.json`. |
| `r` | Reset trackbars to default values. |
| `b` | Toggle face-blur privacy mode. |
| `q` | Quit the application. |
