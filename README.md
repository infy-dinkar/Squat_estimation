# AI Fitness Trainer

A real-time, AI-powered multi-exercise analysis system built with Streamlit, MediaPipe, and OpenCV. The app tracks your body posture via webcam or uploaded video to provide comprehensive coaching, count completed repetitions, and offer targeted improvement suggestions across three major exercises: **Squats**, **Push-ups**, and **Lunges**.

---

## Supported Exercises

| Exercise | Camera Angle | What We Analyze |
|----------|--------------|-----------------|
| **Squat** | Side profile | Knee tracking, hip depth, ankle angle, forward lean. Enforces parallel stance. |
| **Push-up** | Side profile | Elbow angle, body straightness, knee alignment. Enforces horizontal orientation. |
| **Lunge** | Side profile | Front knee depth, back leg stability, torso posture, knee-over-toe check. Enforces split stance. |

---

## Key Features

- **Multi-Exercise Mechanics:** Distinct state machines (s1 → s2 → s3 → s2 → s1) and configuration thresholds for Squats, Push-ups, and Lunges.
- **Exercise Authenticity Guard:** Prevents false counting by dynamically measuring your body's overall orientation (horizontal vs. vertical) and foot stance width (parallel vs. split) to ensure you are actually performing the selected exercise.
- **Human-Friendly Coaching:** Real-time on-screen banners like *"Straighten your back leg to keep it stable"* alongside positive reinforcement hints like *"Perfect depth! Now drive back up"*.
- **Flexible Tolerance:** Employs a smart 10% tolerance threshold allowing flexibility for realistic movement variations while keeping the core coaching feedback strict and actionable.
- **Post-Workout Suggestions:** Instead of generic "Bad reps", the app provides a single `Total Reps` counter and aggregates all unique posture corrections triggered during your session into a helpful **Suggested Improvements** digest.
- **Modern Architecture:** Utilizes Streamlit's official `st.navigation` framework for a clean sidebar UI splitting Home, Live Stream, and Upload Video interfaces.

---

## Project Structure

```
Squat_Tracker/
│
├── Demo.py              # Navigation entrypoint (Run this!)
├── process_frame.py     # Core computer vision & state-machine logic
├── thresholds.py        # Exercise-specific angle configurations & tolerances
├── utils.py             # Math helpers for angle calculation & joint extraction
├── verify.py            # Diagnostic script to check model pipeline health
├── requirements.txt
│
├── pages/
│   ├── Home.py          # App landing and introduction
│   ├── Live_Stream.py   # Webcam WebRTC interface
│   └── Upload_Video.py  # MP4 video upload and processing interface
│
└── squat/               # Conda Python environment (not committed)
```

---

## Getting Started

### Prerequisites

- Anaconda or Miniconda
- Python 3.11 (local) or 3.12 (recommended for deployment)

---

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/fitness-trainer.git
   cd fitness-trainer
   ```

2. **Create the isolated Environment**
   ```bash
   conda create -p ./squat python=3.11 -y
   conda activate ./squat
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   > **Important:** Keep `mediapipe==0.10.21`, `numpy<2`, and the specified OpenCV version untouched. Upgrading MedaiPipe will break the `solutions` API.

---

### Running the App

Always execute the application from the project root using the activated Conda environment:

```bash
conda activate ./squat
streamlit run Demo.py
```

---

## Configuration

Exercise configurations (angles, flexibility tolerance, and inactivity limits) are managed in `thresholds.py`.

```python
# Example Snippet
def get_pushup_thresholds():
    return {
        'ELBOW_THRESH': {'TOP': 150, 'TRANS_MIN': 90, 'BOTTOM': 90},
        'BODY_ANGLE_MIN': 170,
        'LEG_ANGLE_MIN': 170,
        'TOLERANCE': 0.10,          # 10% flexibility for state locks
        'INACTIVE_THRESH': 15.0     # Reset loop if leaving frame for 15s
    }
```

---

## Dependencies

| Package                | Version   | Purpose |
| ---------------------- | --------- | ------- |
| **streamlit**          | latest    | UI framework, `st.navigation` support |
| **streamlit-webrtc**   | latest    | Live browser webcam tracking |
| **mediapipe**          | 0.10.21   | 33-point body map Pose estimation |
| **opencv-python-headless** | 4.10.0.84 | Drawing skeleton overlays & video processing |

---

## Known Limitations

- **Camera Positioning:** You must ensure the camera has a clear view of your **side profile**. The Authenticity Check will prevent counting if you face the camera squarely.
- **WebRTC on Deployment:** Live webcam functionality relies on strict network requirements. If it repeatedly fails to connect on mobile or cloud Wi-Fi setups, use the **Upload Video** module instead.
