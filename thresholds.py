# thresholds.py
# ─────────────────────────────────────────────────────────────────────────────
# Threshold definitions for each supported exercise.
# Squat thresholds are unchanged from the original project.
# Push-up and Lunge thresholds have been added below.
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# SQUAT THRESHOLDS  (original — do not modify)
# ══════════════════════════════════════════════════════════════════════════════

def get_thresholds_beginner():
    return {
        'HIP_KNEE_VERT': {
            'NORMAL': (0, 32),
            'TRANS': (35, 65),
            'PASS': (70, 95)
        },
        'HIP_THRESH': [10, 50],
        'ANKLE_THRESH': 45,
        'KNEE_THRESH': [50, 70, 95],
        'OFFSET_THRESH': 35.0,
        'INACTIVE_THRESH': 15.0,
        'CNT_FRAME_THRESH': 50
    }

def get_thresholds_pro():
    return {
        'HIP_KNEE_VERT': {
            'NORMAL': (0, 32),
            'TRANS': (35, 65),
            'PASS': (80, 95)
        },
        'HIP_THRESH': [15, 50],
        'ANKLE_THRESH': 30,
        'KNEE_THRESH': [50, 80, 95],
        'OFFSET_THRESH': 35.0,
        'INACTIVE_THRESH': 15.0,
        'CNT_FRAME_THRESH': 50
    }


# ══════════════════════════════════════════════════════════════════════════════
# PUSH-UP THRESHOLDS
# Uses a side-view.  Three key angles are tracked:
#   elbow_angle = Shoulder - Elbow - Wrist
#   body_angle  = Shoulder - Hip   - Ankle  (body straightness)
#   leg_angle   = Hip      - Knee  - Ankle  (leg straightness)
#
# TOLERANCE: 10% flexibility is applied to all state-detection angle checks.
#   - Min thresholds become:  value * (1 - TOLERANCE)   e.g. 170 -> 153 deg
#   - Max thresholds become:  value * (1 + TOLERANCE)   e.g.  90 ->  99 deg
#   Feedback/correction messages keep the original strict values.
# ══════════════════════════════════════════════════════════════════════════════

def get_pushup_thresholds():
    return {
        # Elbow angle thresholds (Shoulder-Elbow-Wrist)
        'ELBOW_THRESH': {
            'TOP': 150,      # >= 150 deg -> arms fully extended (s1)
            'TRANS_MIN': 90, # >  90 deg  -> transition (s2)
            'BOTTOM': 90,    # <= 90 deg  -> lowered (s3)
        },
        # Body must stay straight throughout (Shoulder-Hip-Ankle)
        'BODY_ANGLE_MIN': 170,
        # Legs must stay straight throughout (Hip-Knee-Ankle)
        'LEG_ANGLE_MIN': 170,
        # 10% tolerance applied to STATE DETECTION only (not feedback thresholds).
        # Increase this value to make detection more lenient, decrease to tighten it.
        'TOLERANCE': 0.10,
        # Shared pipeline settings
        'OFFSET_THRESH': 35.0,
        'INACTIVE_THRESH': 15.0,
        'CNT_FRAME_THRESH': 50,
    }


# ══════════════════════════════════════════════════════════════════════════════
# LUNGE THRESHOLDS
# Uses a side-view.  Front and back legs are detected by ankle x-position.
#   front_knee_angle = front Hip - front Knee - front Ankle
#   back_knee_angle  = back  Hip - back  Knee - back  Ankle
#   torso_angle      = Shoulder  - Hip - Ankle (visible-side)
#
# TOLERANCE: 10% flexibility applied to state-detection checks only.
# ══════════════════════════════════════════════════════════════════════════════

def get_lunge_thresholds():
    return {
        # Front knee angle ranges
        'FRONT_KNEE': {
            'TOP': 160,        # >= 160 deg -> standing (s1)
            'TRANS_MIN': 110,  # >  110 deg -> transition (s2)
            'TRANS_MAX': 160,  # <  160 deg -> transition (s2)
            'BOTTOM_MIN': 80,  # >=  80 deg -> bottom (s3)
            'BOTTOM_MAX': 110, # <= 110 deg -> bottom (s3)
            'TOO_LOW': 80,     # <   80 deg -> too deep -> feedback (no tolerance applied)
        },
        # Back knee angle (stability)
        'BACK_KNEE': {
            'TOP': 160,        # >= 160 deg required for s1
            'BOTTOM_MIN': 140, # >= 140 deg required for s3
            'BOTTOM_MAX': 175, # <= 175 deg required for s3
        },
        # Torso (Shoulder-Hip-Ankle visible side)
        'TORSO_MIN_S1': 170,   # required for s1 and s3
        'TORSO_MIN_S2': 160,   # required for s2
        'TORSO_INVALID': 150,  # below this -> always invalid
        # 10% tolerance applied to STATE DETECTION only (not feedback thresholds).
        'TOLERANCE': 0.10,
        # Shared pipeline settings
        'OFFSET_THRESH': 35.0,
        'INACTIVE_THRESH': 15.0,
        'CNT_FRAME_THRESH': 50,
    }
