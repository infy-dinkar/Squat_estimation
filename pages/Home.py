import streamlit as st

# Note: st.set_page_config() is handled in Demo.py (the entrypoint)

st.title("AI Fitness Trainer")
st.markdown("""
Welcome to the **AI Fitness Trainer** — powered by MediaPipe and OpenCV.

This app analyses your workout posture in real-time and counts your reps,
telling you exactly what to fix as you exercise.

---

### Supported Exercises

| Exercise | Camera View | What We Analyse |
|----------|-------------|-----------------|
| **Squat**   | Side view | Knee, hip & ankle angles — depth, knee tracking, forward lean |
| **Push-up** | Side view | Elbow angle, body & leg straightness |
| **Lunge**   | Side view | Front knee depth, back leg stability, torso posture, knee-over-toe |

---

### How It Works

1. **Pose Detection** — MediaPipe tracks 33 body landmarks every frame.
2. **Camera Alignment** — The app checks you are in a proper side-profile view.
3. **Angle Computation** — Key joint angles are calculated in real-time.
4. **State Machine** — Movement is tracked through states S1 → S2 → S3 → S2 → S1.
5. **Rep Counting** — A rep is counted only when the full cycle is completed correctly.
6. **Live Coaching** — Mistake feedback is shown on-screen *before* the rep ends.

---

### Getting Started

**Select a mode from the sidebar on the left:**

- **Live Stream** — Use your webcam for instant real-time analysis.
- **Upload Video** — Upload a pre-recorded workout video for a full breakdown.

> **Tip:** Stand fully sideways to the camera so your entire body is visible from head to toe.
""")
