
# AI Squat Tracker

A real-time AI-powered squat analysis system built with Streamlit, MediaPipe, and OpenCV. The app detects your body posture using a webcam or an uploaded video, tracks your squat movement phases, counts correct and incorrect squats, and gives plain-English coaching feedback in real time.

---

## Demo

| Live Webcam Mode | Upload Video Mode |
|:---:|:---:|
| Stream directly from your browser | Upload `.mp4`, `.mov`, or `.avi` |
| Real-time skeleton overlay | Frame-by-frame analysis |
| Instant posture feedback | Downloadable annotated output video |

---

## Features

- Live Webcam Analysis — browser-based, powered by WebRTC  
- Uploaded Video Analysis — process pre-recorded squat videos and download annotated output  
- Pose Detection — uses MediaPipe Pose to track 33 body landmarks  
- Angle Computation — calculates hip, knee, and ankle angles relative to a vertical reference  
- Squat Phase Detection — maps movement into Standing (S1), Transition (S2), Bottom (S3)  
- Correct / Incorrect Squat Counting — based on full range of motion and posture quality  
- Real-time Coaching Feedback — messages like "Your knee is going past your toes"  
- Inactivity Reset — resets counters if the user leaves the frame  
- Annotated Frame Rendering — skeleton overlay, reference lines, and angle values  

---

## Project Structure

```

Squat_Tracker/
│
├── Demo.py
├── process_frame.py
├── thresholds.py
├── utils.py
├── requirements.txt
├── .gitignore
│
├── pages/
│   ├── Live_Stream.py
│   └── Upload_Video.py
│
└── squat/   # Conda environment (not committed)

````

---

## How It Works

### 1. Pose Detection
Each frame is processed by MediaPipe Pose to extract 33 body landmarks.

### 2. Camera Alignment Check
Ensures the user is in side view using shoulder–nose offset angle.

### 3. Body Side Selection
Chooses left or right body side based on visibility.

### 4. Angle Computation
Calculates vertical reference angles:

- Hip angle
- Knee angle
- Ankle angle

### 5. Squat State Machine

| State | Meaning | Range |
|------|--------|------|
| S1 | Standing | 0°–32° |
| S2 | Transition | 35°–65° |
| S3 | Bottom | 70°–95° |

### 6. Rep Counting
Valid squat sequence:
S2 → S3 → S2 → S1

- Correct: full sequence + good posture  
- Incorrect: incomplete or bad posture  

### 7. Feedback
Displays posture warnings such as:

- Leaning too forward
- Knee over toe
- Not deep enough

---

## Getting Started

### Prerequisites

- Anaconda or Miniconda  
- Python 3.11 (local) or 3.12 (recommended for deployment)

---

### Clone Repository

```bash
git clone https://github.com/your-username/squat-tracker.git
cd squat-tracker
````

---

### Create Environment

```bash
conda create -p ./squat python=3.11 -y
conda activate ./squat
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

Important Notes:

* mediapipe==0.10.21 is required because newer versions removed the `solutions` API
* numpy<2 is required for compatibility
* OpenCV is pinned to avoid deployment issues

---

### Run App

```bash
streamlit run Demo.py
```

---

## Usage Guide

### Live Webcam Mode

1. Open Live Stream page
2. Click Start
3. Stand sideways to the camera
4. Ensure full body is visible
5. Perform squats

---

### Upload Video Mode

1. Open Upload Video page
2. Upload a video file
3. Wait for processing
4. View results and download output

---

## Known Limitations

### Live Webcam on Streamlit Cloud

* WebRTC connection may fail on some networks
* Requires STUN/TURN configuration
* Works best on local machine or stable WiFi

If webcam fails, use Upload Video Mode.

---

### Camera Position Dependency

* Must stand sideways
* Full body must be visible
* Camera at hip height
* Good lighting required

---

### Frame Noise in Live Mode

* Frame drops may occur
* Minor inaccuracies possible
* Smoothing reduces noise but not fully

---

## Configuration

Defined in `thresholds.py`:

```python
{
    'HIP_KNEE_VERT': {
        'NORMAL': (0, 32),
        'TRANS':  (35, 65),
        'PASS':   (70, 95)
    },
    'HIP_THRESH': [10, 50],
    'ANKLE_THRESH': 45,
    'KNEE_THRESH': [50, 70, 95],
    'OFFSET_THRESH': 35.0,
    'INACTIVE_THRESH': 15.0,
    'CNT_FRAME_THRESH': 50
}
```

---

## Dependencies

| Package                | Version   |
| ---------------------- | --------- |
| streamlit              | latest    |
| streamlit-webrtc       | latest    |
| mediapipe              | 0.10.21   |
| opencv-python-headless | 4.10.0.84 |
| numpy                  | <2        |

---

## Deployment Notes

To deploy on Streamlit Cloud:

1. Set Python version to 3.12
2. Use pinned requirements.txt
3. Reboot app after updates

---

### Common Issues

| Issue                   | Cause             | Fix             |
| ----------------------- | ----------------- | --------------- |
| MediaPipe install error | Python mismatch   | Use Python 3.12 |
| mp.solutions error      | Wrong version     | Use 0.10.21     |
| cv2 import error        | OpenCV mismatch   | Pin version     |
| Webcam not working      | WebRTC limitation | Use upload mode |

---

## Live Demo

[https://squatestimation.streamlit.app](https://squatestimation.streamlit.app)

```


