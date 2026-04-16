import streamlit as st
import cv2
import tempfile
from process_frame import ProcessFrame
from thresholds import get_thresholds_beginner
from utils import get_mediapipe_pose

st.set_page_config(page_title="Upload Video - Squat Tracker", layout="wide")
st.title(" Upload Video Analysis")
st.caption("Upload a squat video to get a full breakdown of your form.")

uploaded_file = st.file_uploader(
    "Choose a video file", type=['mp4', 'mov', 'avi'],
    help="Supported formats: MP4, MOV, AVI"
)

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()

    st.info(" Analysing your video, please wait...")

    cap = cv2.VideoCapture(tfile.name)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30

    thresholds = get_thresholds_beginner()
    processor  = ProcessFrame(thresholds, flip_frame=False)
    pose       = get_mediapipe_pose()

    out_file = "output_video.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
    out      = cv2.VideoWriter(out_file, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    stframe = st.empty()

    curr_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = processor.process(frame_rgb, pose)

        stframe.image(processed, channels="RGB", use_container_width=True)
        out.write(cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))

        curr_frame += 1
        if frame_count > 0:
            progress_bar.progress(min(curr_frame / frame_count, 1.0))

    cap.release()
    out.release()

    st.success(" Analysis complete!")
    with open(out_file, "rb") as f:
        st.download_button(
            label="⬇ Download Annotated Video",
            data=f,
            file_name="squat_analysis.mp4",
            mime="video/mp4"
        )
