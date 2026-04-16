import streamlit as st

st.set_page_config(page_title="AI Squat Tracker", layout="wide")

st.title(" AI Squat Analysis System")
st.markdown("""
Welcome to the AI Squat Tracker!
This application uses MediaPipe and OpenCV to track your squat posture in real-time.

### Features
- **Live Webcam Mode**: Stream directly from your browser to analyze squats instantly.
- **Video Upload Mode**: Upload prerecorded videos to get analytics.

### How it works
1. **Pose Detection**: We track your body landmarks in real-time.
2. **Camera Alignment**: Make sure you are standing in a side-profile view (camera aligned to your side).
3. **Squat Analytics**: We analyze your knee, hip, and ankle angles to output live transition state tracking and posture feedback.

⬅ **Select a mode from the sidebar to begin.**
""")
