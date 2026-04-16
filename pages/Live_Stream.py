import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from process_frame import ProcessFrame
from thresholds import get_thresholds_beginner
from utils import get_mediapipe_pose

st.set_page_config(page_title="Live Analysis - Squat Tracker", layout="wide")
st.title("🔴 Live Squat Analysis")
st.caption("Stand sideways with the camera facing your right or left side for best results.")

# Single shared instances per session
if 'processor_live' not in st.session_state:
    thresholds = get_thresholds_beginner()
    st.session_state['processor_live'] = ProcessFrame(thresholds, flip_frame=True)

if 'pose_live' not in st.session_state:
    st.session_state['pose_live'] = get_mediapipe_pose()

processor = st.session_state['processor_live']
pose = st.session_state['pose_live']


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="rgb24")
    processed = processor.process(img, pose)
    return av.VideoFrame.from_ndarray(processed, format="rgb24")


rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="squat-tracker-live",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_configuration,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.info(" **Tips for best results:**  \n"
        "- Position your camera at hip height on your side  \n"
        "- Make sure your full body (head to feet) is visible  \n"
        "- Ensure good lighting")
