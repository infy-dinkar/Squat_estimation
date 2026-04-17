from thresholds import get_thresholds_beginner, get_pushup_thresholds, get_lunge_thresholds
from process_frame import ProcessFrame
import numpy as np

# Instantiate all three modes
pf_sq = ProcessFrame(get_thresholds_beginner(), exercise_mode="squat")
pf_pu = ProcessFrame(get_pushup_thresholds(),   exercise_mode="pushup")
pf_lu = ProcessFrame(get_lunge_thresholds(),    exercise_mode="lunge")
print("Instantiation OK")

# Verify counters
for pf, mode in [(pf_sq,"squat"),(pf_pu,"pushup"),(pf_lu,"lunge")]:
    assert 'REP_COUNT' in pf.state_tracker
    assert 'SUGGESTIONS' in pf.state_tracker
    # test dummy frame counters logic without failing
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    pf._draw_counters(frame)
    pf._draw_rep_flash(frame, 100, 100)
    pf._draw_squat_feedback(frame, 100, 100) if mode == "squat" else pf._draw_generic_feedback(frame, {}, getattr(pf, f'_{mode.upper()}_FEEDBACK', {}), 100, 100)
print("Counters functionality OK")
