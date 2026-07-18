# Computer Vision & Webcam Internship Project

This repository contains my work for a 1-month Computer Vision and Webcam Manipulation internship. The project builds up from basic webcam access to a real-time signal extraction application.

## Project Structure

* **`task1_webcam_basics.py`**: Introduction to capturing video, exploring frame properties, and basic OpenCV processing.
* **`task2_pixel_manipulation.py`**: Direct numpy array manipulation for channel swapping, brightness/contrast, and color filtering.
* **`task3_face_roi.py`**: Real-time face detection and Region of Interest (ROI) extraction using MediaPipe.
* **`task4_signal_extraction.py`**: Extracting a live signal from the forehead ROI and applying a bandpass filter.
* **`task5_interactive_tuning.py`**: Live parameter tuning using OpenCV trackbars and JSON config persistence.
* **`task6_project/`**: The final integrated application combining all previous tasks into a single modular codebase.
* **`NOTES.md`**: Observations on signal robustness (motion artifacts, lighting, etc.).
* **`assistant_log.md`**: Log of interactions with the AI coding assistant.

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/<your-github-username>/cv-webcam-internship.git
    cd cv-webcam-internship
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv cv_env
    # Windows: cv_env\Scripts\activate
    # Mac/Linux: source cv_env/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Final App

Navigate to the `task6_project` directory and run `main.py`:

```bash
cd task6_project
python main.py
```

See `task6_project/README.md` for detailed usage instructions and keyboard shortcuts.

## Key Learnings
* **OpenCV Basics:** Capturing frames, color spaces (BGR vs RGB), and basic image processing.
* **NumPy for CV:** Why vectorized operations are crucial for real-time performance compared to nested loops.
* **Face Detection:** Using MediaPipe for fast, landmark-based tracking.
* **Signal Processing:** Extracting subtle signals (like a pulse) from video using ROI tracking, rolling buffers, and Butterworth bandpass filters.
* **Interactive UIs:** Building live-tuning interfaces with OpenCV trackbars.
* **Software Architecture:** Refactoring standalone scripts into a clean, modular application.
