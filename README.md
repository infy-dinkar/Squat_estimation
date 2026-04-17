#  AI Squat Tracker

A real-time AI-powered squat analysis system built with **Streamlit**, **MediaPipe**, and **OpenCV**. The app detects your body posture using a webcam or an uploaded video, tracks your squat movement phases, counts correct and incorrect squats, and gives you plain-English coaching feedback — all in real time.

---

##  Demo

| Live Webcam Mode | Upload Video Mode |
|:---:|:---:|
| Stream directly from your browser | Upload `.mp4`, `.mov`, or `.avi` |
| Real-time skeleton overlay | Frame-by-frame analysis |
| Instant posture feedback | Downloadable annotated output video |

---

##  Features

-  **Live Webcam Analysis** — browser-based, no local camera driver needed (powered by WebRTC)
-  **Uploaded Video Analysis** — process pre-recorded squat videos and download annotated output
-  **Pose Detection** — uses Google MediaPipe Pose to track 33 body landmarks
-  **Angle Computation** — calculates hip, knee, and ankle angles relative to a vertical reference
-  **Squat Phase Detection** — maps movement into three states: Standing (S1), Transition (S2), Bottom (S3)
-  **Correct /  Incorrect Squat Counting** — based on full range of motion and posture quality
-  **Real-time Coaching Feedback** — human-readable messages like "Your knee is going past your toes"
-  **Inactivity Reset** — automatically resets counters if you leave the frame or stop moving
-  **Annotated Frame Rendering** — skeleton overlay, dotted vertical reference lines, and angle values

---

##  Project Structure

```
Squat_Tracker/
│
├── Demo.py                   # Home page — app entry point
├── process_frame.py          # Core engine — ProcessFrame class
├── thresholds.py             # Angle threshold configuration
├── utils.py                  # Helper functions (angles, landmarks, drawing)
├── requirements.txt          # Python dependencies
├── .gitignore
│
├── pages/
│   ├── Live_Stream.py        # Live webcam page (WebRTC)
│   └── Upload_Video.py       # Uploaded video analysis page
│
└── squat/                    # Conda environment (not committed to git)
```

---

##  How It Works

### 1. Pose Detection
Each video frame is processed by **MediaPipe Pose**, which returns 33 normalized body landmarks. The app extracts the joints needed for squat analysis:

| Joint | MediaPipe Index |
|---|---|
| Nose | 0 |
| Shoulder (L/R) | 11 / 12 |
| Hip (L/R) | 23 / 24 |
| Knee (L/R) | 25 / 26 |
| Ankle (L/R) | 27 / 28 |
| Foot (L/R) | 31 / 32 |

### 2. Camera Alignment Check
The app computes the angle between both shoulders and the nose. If this **offset angle** exceeds a threshold, it means the user is facing the camera front-on rather than side-on, and a warning is displayed:
> *"Stand sideways — camera needs a side view"*

This is necessary because all squat geometry is computed from a **side-profile view**.

### 3. Body Side Selection
The app automatically picks the body side (left or right) that gives a larger vertical span from shoulder to foot — this is the visible side facing the camera.

### 4. Angle Computation
Three vertical-reference angles are computed:

| Angle | Joints Used | Purpose |
|---|---|---|
| **Hip Vertical Angle** | Shoulder → Hip → Vertical | Back posture check |
| **Knee Vertical Angle** | Hip → Knee → Vertical | Primary squat phase detector |
| **Ankle Vertical Angle** | Knee → Ankle → Vertical | Knee-over-toe / shin alignment |

### 5. Squat State Machine
The **knee vertical angle** is mapped to one of three states:

| State | Meaning | Knee Angle Range |
|---|---|---|
| `S1` | Standing / resting | 0° – 32° |
| `S2` | Transition (going down or up) | 35° – 65° |
| `S3` | Bottom / deep squat | 70° – 95° |

### 6. Sequence Tracking & Counting
A valid squat requires the sequence: **S2 → S3 → S2**, followed by a return to **S1**.

- ** Correct squat**: Full sequence completed + no posture errors
- ** Incorrect squat**: Incomplete sequence OR severe posture flag raised

### 7. Posture Feedback

| Condition | Message Shown |
|---|---|
| Hip angle too large (leaning back) | *"⚠ Lean your upper body forward a bit"* |
| Hip angle too small (bending forward) | *"⚠ You are leaning too far forward"* |
| Ankle angle too large (knee over toe) | *"⚠ Your knee is going past your toes"* |
| Knee angle too deep | *"⚠ You are squatting too deep"* |
| Moderate knee depth | *"💡 Try to go a little deeper"* |

Feedback messages persist on screen for a configurable number of frames (`CNT_FRAME_THRESH`) to prevent flickering.

---

##  Getting Started

### Prerequisites

- [Anaconda](https://www.anaconda.com/) or Miniconda
- Python 3.11

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/squat-tracker.git
cd squat-tracker
```

### 2. Create and Activate Conda Environment

```powershell
# Create environment inside project folder
conda create -p ./squat python=3.11 -y

# Activate it (Windows PowerShell)
conda activate .\squat
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

>  **Important**: `mediapipe==0.10.14` is pinned intentionally.
> Newer versions (0.10.15+) removed the `mediapipe.solutions` API that this project uses.

### 4. Run the App

```powershell
streamlit run Demo.py
```

The app will open automatically at `http://localhost:8501`.

---

##  Usage Guide

### Live Webcam Mode
1. Navigate to **Live Stream** in the sidebar
2. Click **START** to allow camera access in your browser
3. Position yourself **sideways** to the camera (left or right side facing the lens)
4. Make sure your **full body** (head to feet) is visible
5. Start squatting — the app will track your form in real time

### Upload Video Mode
1. Navigate to **Upload Video** in the sidebar
2. Click **Browse files** and select a `.mp4`, `.mov`, or `.avi` file
3. Wait for processing to complete (a progress bar shows status)
4. Preview the annotated video in the browser
5. Click **Download Annotated Video** to save the result

---

##  Configuration

All angle thresholds are defined in `thresholds.py` inside `get_thresholds_beginner()`:

```python
{
    'HIP_KNEE_VERT': {
        'NORMAL': (0, 32),    # S1 — standing
        'TRANS':  (35, 65),   # S2 — transition
        'PASS':   (70, 95)    # S3 — deep squat
    },
    'HIP_THRESH':    [10, 50],  # [min, max] back lean angle
    'ANKLE_THRESH':  45,        # knee-over-toe limit
    'KNEE_THRESH':   [50, 70, 95],
    'OFFSET_THRESH': 35.0,      # side-view validation
    'INACTIVE_THRESH': 15.0,    # seconds before counter reset
    'CNT_FRAME_THRESH': 50      # frames to keep feedback visible
}
```

---

##  Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | latest | Web UI framework |
| `streamlit-webrtc` | latest | Browser webcam access |
| `mediapipe` | 0.10.14 | Pose landmark detection |
| `opencv-python-headless` | latest | Frame reading, drawing, video I/O |
| `numpy` | latest | Numerical operations |

---

##  Key Design Decisions

- **Flip-then-draw** in live mode: The frame is horizontally flipped first, and all annotations are drawn after. This ensures text and overlays are always readable (not mirrored).
- **Vertical reference angles** instead of classic 3-point joint angles: This gives more stable and intuitive phase detection that isn't affected by the person's position in the frame.
- **MediaPipe `solutions` API**: Requires `mediapipe==0.10.14` — the newer Tasks API lacks a drop-in replacement for `mp.solutions.pose`.

---

##  DEPLOYED ON STREAMLIT
LINK-https://squatestimation.streamlit.app

