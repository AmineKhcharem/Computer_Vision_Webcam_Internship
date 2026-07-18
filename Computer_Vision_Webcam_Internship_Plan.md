# Computer Vision & Webcam Manipulation — 1-Month Internship Program

Welcome! This document outlines your tasks for the next 4 weeks. It's designed to take you from webcam/CV basics to building a small working real-time application. The tasks are ordered so each one builds on the previous, so make sure you're comfortable with a task before moving to the next.

**Prerequisites:** basic Python knowledge (variables, functions, loops, classes)

---

## Working with a Coding Assistant (Claude Code)

You're expected to use an AI coding assistant (Claude Code, or similar) throughout this internship rather than writing everything unaided. This mirrors how real engineering teams work today, and it will speed up your learning curve. A few guidelines:

- Use the assistant to **scaffold** scripts quickly (boilerplate capture loops, argument parsing, file structure), then read and understand every line before moving on — the goal is to learn, not just to produce output.
- Ask the assistant to **explain** unfamiliar functions or concepts (e.g., "explain what `cv2.CAP_PROP_EXPOSURE` does and why it matters") rather than just copy-pasting.
- Use the assistant to **review** your code at the end of each task: ask it to point out bugs, inefficiencies, or bad practices.
- Use the assistant to help you **write commit messages, README files, and code comments**, so good documentation habits are built in from day one.
- Use the assistant to help **debug** errors: paste the traceback and ask for a root-cause explanation, not just a fix.
- Keep a short log (`assistant_log.md`) noting a few interesting prompts/answers as you go — this helps you reflect on what you learned, and gives your supervisor visibility into your process.

This isn't about the assistant doing the work for you — use it as a pair-programmer/tutor, but make sure you understand and can explain every task you deliver.

---

## Environment Setup (Day 0)

### 1. Install Python
Use Python 3.10 or 3.11 (best compatibility with the CV libraries below).

### 2. Create a virtual environment
```bash
python3 -m venv cv_env
source cv_env/bin/activate      # Mac/Linux
cv_env\Scripts\activate         # Windows
```

### 3. Create `requirements.txt`
```
opencv-python==4.10.0.84
opencv-contrib-python==4.10.0.84
numpy==1.26.4
matplotlib==3.9.0
mediapipe==0.10.14
scipy==1.13.1
imutils==0.5.4
```

Install with:
```bash
pip install -r requirements.txt
```

### 4. Verify webcam access
Write a small test script (`test_webcam.py`):
```python
import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
else:
    ret, frame = cap.read()
    print("Webcam OK, frame shape:", frame.shape)
cap.release()
```

---

## Task 1 — Webcam & OpenCV Fundamentals

**Goal:** Understand how to capture, display, and manipulate raw camera frames.

### Steps
1. Capture live video from your webcam and display it in a window (`cv2.VideoCapture`, `cv2.imshow`)
2. Print and understand frame properties: resolution, FPS, color format (BGR vs RGB)
3. Try switching resolution and FPS programmatically (`cap.set(cv2.CAP_PROP_FRAME_WIDTH, ...)`)
4. Explore camera controls: exposure, brightness, contrast, white balance
   - On Linux: `v4l2-ctl --list-ctrls -d /dev/video0`
   - On Windows: DirectShow properties via OpenCV (`cv2.CAP_PROP_EXPOSURE`, etc.)
5. Save frames to disk as images, and record a short video to `.mp4` using `cv2.VideoWriter`
6. Basic image processing: grayscale conversion, blurring, edge detection (Canny), thresholding

### Deliverable
A script `task1_webcam_basics.py` that:
- Opens the webcam
- Displays a live feed
- Lets you press keys to toggle grayscale / edge detection / save a snapshot
- Logs resolution and FPS to the console

---
## Task 2 — Pixel Manipulation

**Goal:** Build direct intuition for how a frame is really just a numpy array — this is essential before doing any signal extraction later.

### Steps
1. Access and print raw pixel values at a specific (x, y) coordinate, and understand the channel order (BGR in OpenCV, not RGB)
2. Manually split a frame into its B, G, R channels and display each channel separately as grayscale
3. Manually recombine/swap channels (e.g., turn a frame into "fake infrared" by swapping R and B) using pure numpy indexing, without built-in OpenCV color functions
4. Implement a simple brightness/contrast adjustment by directly manipulating the pixel array (`frame = alpha * frame + beta`, with proper clipping to 0–255)
5. Implement a basic pixelation/mosaic effect (downscale then upscale a region using nearest-neighbor) — useful conceptually for privacy-blurring a face region
6. Implement a simple color filter (e.g., sepia tone) using a manual matrix transform on pixel values
7. Compute and display a live histogram of pixel intensities (using numpy — first do it manually, then compare against `cv2.calcHist`)
8. Measure the performance difference between a naive Python for-loop pixel manipulation vs a vectorized numpy operation on the same effect, and explain why the difference exists

### Deliverable
A script `task2_pixel_manipulation.py` that:
- Lets you cycle (via key press) through: raw feed, channel-swapped feed, brightness/contrast adjusted feed, pixelated region, sepia filter, live histogram overlay
- Includes a short comment block or `NOTES.md` entry comparing loop-based vs vectorized performance

---

## Task 3 — Face Detection & Region of Interest (ROI)

**Goal:** Detect faces in real time and extract specific regions.

### Steps
1. Face detection using Haar Cascade (`cv2.CascadeClassifier`) — understand how it works at a high level (sliding window + trained classifier)
2. Face detection using a more modern method: MediaPipe Face Detection or Face Mesh — compare accuracy and speed vs Haar Cascade
3. Draw bounding boxes and landmarks on the live feed
4. Extract a specific ROI (e.g., forehead or cheek region) based on facial landmarks
5. Handle multiple faces in frame (draw a box + label/index for each detected face)
6. Measure and print FPS of the whole detection pipeline (be aware of the performance cost of detection)
7. Face tracking across frames: instead of re-running detection every frame, track a bounding box between detections (e.g., using `cv2.TrackerCSRT` or a simple centroid tracker) and compare smoothness/performance vs re-detecting every frame
8. Apply a privacy-preserving effect on detected faces: blur or pixelate the face region in real time (ties back into Task 2)
9. Basic face alignment: use landmarks (eyes) to rotate/crop the face so it's horizontally aligned, regardless of head tilt
10. Simple face counting/logging: log to console (or a CSV) how many faces were detected per second over a short session

### Deliverable
A script `task3_face_roi.py` that:
- Detects a face in real time
- Draws landmarks
- Extracts and displays a cropped ROI in a separate window
- Handles the case of no face / multiple faces gracefully
- Includes a toggle for a live face-blur/pixelation privacy mode
- Includes a toggle between "detect every frame" and "detect + track" modes, with FPS shown on screen for comparison

---

## Task 4 — Signal Extraction & Robustness

**Goal:** Extract a simple time-varying signal from video and deal with real-world noise.

### Steps
1. From the ROI, extract the average pixel intensity (per channel) over time
2. Store this in a rolling buffer (e.g., last 10 seconds of frames)
3. Plot the signal in real time (using `matplotlib` in a separate thread, or simple live plotting)
4. Apply a basic bandpass filter (using `scipy.signal`) to clean up the signal
5. Test robustness:
   - What happens with movement?
   - What happens with lighting changes?
   - What happens with different webcams if you have access to more than one?
6. Document your observations (a short markdown or text file — "what breaks the signal and why")

### Deliverable
A script `task4_signal_extraction.py` that:
- Extracts and buffers a live signal from the ROI
- Applies filtering
- Plots the raw vs filtered signal in real time or at the end of a recording session
- Includes a short `NOTES.md` documenting your robustness observations

---

## Task 5 — Interactive Parameter Tuning

**Goal:** Learn how to expose parameters as live, adjustable controls instead of hardcoded values — an important habit for any CV pipeline that needs tuning in different conditions.

### Steps
1. Add OpenCV trackbars (`cv2.createTrackbar`) to a window, letting you adjust parameters live (e.g., brightness/contrast values from Task 2, or blur/pixelation strength)
2. Add trackbars to tune face-detection parameters (e.g., `scaleFactor` and `minNeighbors` for Haar Cascade) and observe how detection quality changes in real time
3. Add trackbars to tune the bandpass filter's cutoff frequencies from Task 4, and observe the effect on the filtered signal live
4. Save and reload a chosen set of parameter values to/from a small config file (JSON or YAML), so tuned settings can persist between runs

### Deliverable
A script `task5_interactive_tuning.py` that:
- Opens a control window with trackbars for at least 3 different parameters across the pipeline (pick from brightness/contrast, detection thresholds, filter cutoffs, pixelation strength)
- Applies the tuned values live to the video feed
- Can save the current trackbar values to a config file and reload them on startup

---

## Task 6 — Integration Project & Documentation

**Goal:** Combine everything into a small end-to-end demo application.

### Steps
1. Build a single application that:
   - Captures webcam feed
   - Detects face and extracts ROI
   - Extracts and filters a signal
   - Displays a live overlay (webcam feed + signal plot side-by-side, or overlayed)
2. Reuse the trackbar controls from Task 5 so key parameters are adjustable live in the final app
3. Add a simple UI or on-screen instructions (can be OpenCV-drawn text, or a simple Streamlit app if you want an extra challenge)
4. Clean up your code: split into modules (e.g., `capture.py`, `detection.py`, `signal_processing.py`, `main.py`)
5. Write a `README.md` explaining what the project does, how to run it, and what you learned
6. Prepare a short (5–10 min) walkthrough/demo — either live or recorded

### Deliverable
A complete, structured mini-project ready to push to GitHub.

---

## GitHub Setup & Submission Instructions

### 1. Create the repository
Create a **new GitHub repository** named something like:
```
cv-webcam-internship
```

### 2. Initialize locally and push
Inside your project folder:
```bash
git init
git add .
git commit -m "Initial commit: internship project setup"
git branch -M main
git remote add origin https://github.com/<your-github-username>/cv-webcam-internship.git
git push -u origin main
```

### 3. Suggested repo structure
```
cv-webcam-internship/
├── README.md
├── requirements.txt
├── task1_webcam_basics.py
├── task2_pixel_manipulation.py
├── task3_face_roi.py
├── task4_signal_extraction.py
├── task5_interactive_tuning.py
├── NOTES.md
├── assistant_log.md
├── task6_project/
│   ├── main.py
│   ├── capture.py
│   ├── detection.py
│   ├── signal_processing.py
│   └── README.md
└── .gitignore
```

### 4. `.gitignore` suggestion
```
cv_env/
__pycache__/
*.pyc
*.mp4
*.avi
.DS_Store
```

### 5. Sharing your repo
Once pushed, please:
- Add your supervisor as a collaborator (Settings → Collaborators → Add people), using the GitHub username you'll be given separately, **or**
- If the repo is public, just share the link directly

### 6. Regular check-ins
Please commit and push after finishing each task, so your progress can be reviewed incrementally rather than only at the end of the month:
```bash
git add .
git commit -m "Task 3: face detection and ROI extraction"
git push
```

---

## Optional Stretch Goals (if you finish early)
- Try running the pipeline on a different/external USB webcam and compare quality
- Add a simple GUI using Streamlit or Tkinter
- Experiment with a different face landmark model (e.g., dlib's 68-point predictor) and compare accuracy/speed vs MediaPipe
- Add unit tests for the signal processing functions
- Add simple motion detection (frame differencing) as a trigger to start/stop signal recording automatically

If you finish the core tasks ahead of schedule, more advanced tasks can be assigned, for example:
- Optical flow: implement Lucas-Kanade or Farneback optical flow to track motion between frames, and visualize the motion vectors live
- Simple object tracking with a Kalman filter, comparing its smoothness vs the basic tracker used in Task 3
- Basic camera calibration (checkerboard method) to estimate and correct for lens distortion
- A lightweight custom face/object detector: train a small classifier (e.g., HOG + SVM, or a small CNN) on a custom dataset and compare it against MediaPipe
- Multi-threading or multiprocessing the pipeline (capture, processing, display on separate threads) and measuring the FPS improvement
- Package the final app into a standalone executable or a simple Docker container

We'll discuss together which of these makes sense to take on, based on how the first tasks go.

---

Good luck, and don't hesitate to ask questions along the way!