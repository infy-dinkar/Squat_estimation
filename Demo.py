import streamlit as st

# ── Single global page config — must live in the entrypoint only ──────────────
st.set_page_config(
    page_title="AI Fitness Trainer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Explicit navigation (required by Streamlit >= 1.36) ───────────────────────
# Defining pages here prevents the "Cannot infer page title" MPA v1 error.
pg = st.navigation([
    st.Page("pages/Home.py",         title="Home",         default=True),
    st.Page("pages/Live_Stream.py",  title="Live Stream"),
    st.Page("pages/Upload_Video.py", title="Upload Video"),
])

pg.run()
