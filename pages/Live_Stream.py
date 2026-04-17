import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from process_frame import ProcessFrame
from thresholds import get_thresholds_beginner, get_pushup_thresholds, get_lunge_thresholds
from utils import get_mediapipe_pose


st.title("Live Exercise Analysis")

st.caption("Stand sideways with the camera facing your right or left side for best results.")

# ── Exercise selector ─────────────────────────────────────────────────────────
EXERCISE_OPTIONS = {
    " Squat":   ("squat",   get_thresholds_beginner),
    " Push-up": ("pushup",  get_pushup_thresholds),
    " Lunge":   ("lunge",   get_lunge_thresholds),
}

selected_label = st.selectbox(
    "Select Exercise",
    options=list(EXERCISE_OPTIONS.keys()),
    index=0,
    help="Choose which exercise you want to perform. The AI coach will adapt automatically.",
)

exercise_mode, thresh_fn = EXERCISE_OPTIONS[selected_label]

# ── Session state: recreate processor only when exercise changes ──────────────
# Using exercise_mode as part of the key ensures a clean reset on switch.
processor_key = f"processor_live_{exercise_mode}"
pose_key      = "pose_live"

if processor_key not in st.session_state:
    st.session_state[processor_key] = ProcessFrame(
        thresholds=thresh_fn(),
        flip_frame=True,
        exercise_mode=exercise_mode,
    )

if pose_key not in st.session_state:
    st.session_state[pose_key] = get_mediapipe_pose()

processor = st.session_state[processor_key]
pose      = st.session_state[pose_key]


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="rgb24")
    processed = processor.process(img, pose)
    return av.VideoFrame.from_ndarray(processed, format="rgb24")


rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# The webrtc_streamer key changes with exercise_mode so it reconnects cleanly.
webrtc_streamer(
    key=f"fitness-trainer-live-{exercise_mode}",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_configuration,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# ── Tips panel ────────────────────────────────────────────────────────────────
TIPS = {
    "squat": (
        "- Position your camera at hip height on your side\n"
        "- Make sure your full body (head to feet) is visible\n"
        "- Ensure good lighting"
    ),
    "pushup": (
        "- Lay sideways to the camera so your full body is visible\n"
        "- Keep your body in one straight line\n"
        "- Ensure good lighting"
    ),
    "lunge": (
        "- Stand sideways to the camera\n"
        "- Make sure both legs are visible throughout the movement\n"
        "- Ensure good lighting"
    ),
}
st.info("**Tips for best results:**\n" + TIPS[exercise_mode])
