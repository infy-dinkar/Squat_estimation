import time
import cv2
import numpy as np
from utils import find_angle, get_landmark_features, draw_text, draw_dotted_line


class ProcessFrame:
    """
    Unified frame processor for all supported exercises.

    exercise_mode : "squat" | "pushup" | "lunge"
    flip_frame    : True for live-webcam (mirrors image before processing)
    thresholds    : dict returned by the matching get_*_thresholds() function
    """

    def __init__(self, thresholds, flip_frame=False, exercise_mode="squat"):
        self.flip_frame    = flip_frame
        self.thresholds    = thresholds
        self.exercise_mode = exercise_mode.lower()

        # ── Shared state tracker ──────────────────────────────────────────────
        self.state_tracker = {
            'state_seq':                [],
            'start_inactive_time':      time.perf_counter(),
            'start_inactive_time_front':time.perf_counter(),
            'INACTIVE_TIME':            0.0,
            'INACTIVE_TIME_FRONT':      0.0,
            # Generic feedback display flags (up to 6 correction-message slots)
            'DISPLAY_TEXT':    np.full((6,), False),
            'COUNT_FRAMES':    np.zeros((6,), dtype=np.int64),
            'INCORRECT_POSTURE': False,
            'prev_state':      None,
            'curr_state':      None,
            # Universal rep counters
            'REP_COUNT':       0,
            'SUGGESTIONS':     [],

            # Positive coaching overlay
            'POSITIVE_MSG':        '',     # current positive hint text
            'POSITIVE_FRAMES':     0,      # frames left to show it
            # Rep-completion flash
            'REP_FLASH_MSG':       '',
            'REP_FLASH_FRAMES':    0,
            # Squat-only
            'LOWER_HIPS':      False,
        }

        # ── How long (in frames) each overlay stays visible ───────────────────
        self._POSITIVE_DISPLAY_FRAMES = 45   # ~1.5 s at 30 fps
        self._REP_FLASH_FRAMES        = 60   # ~2 s at 30 fps

        # ══════════════════════════════════════════════════════════════════════
        # CORRECTION MESSAGES  — short, friendly, actionable
        # ══════════════════════════════════════════════════════════════════════

        # Squat (4 slots)
        self._SQUAT_FEEDBACK = {
            0: "Lean forward slightly — shift your chest over your knees",
            1: "You're leaning too far — bring your chest up a bit",
            2: "Knee going past toes — push your hips back more",
            3: "A bit too deep — rise up just a little",
        }

        # Push-up (4 slots)
        self._PUSHUP_FEEDBACK = {
            0: "Lock your legs straight — squeeze your glutes",
            1: "Your hips are drooping — tighten your core and lift them",
            2: "Lower your chest closer to the ground",
            3: "Push all the way up and fully extend your arms",
        }

        # Lunge (6 slots)
        self._LUNGE_FEEDBACK = {
            0: "Chest up! Stand tall and keep your back straight",
            1: "Bend your front knee a little more to go deeper",
            2: "You are going too deep — rise up slightly",
            3: "Straighten your back leg to keep it stable",
            4: "Your front knee is passing your toes — step a little wider",
            5: "Don't lean forward — keep your torso upright",
        }

        # ══════════════════════════════════════════════════════════════════════
        # POSITIVE / COACHING MESSAGES  — shown when posture is correct
        # grouped by exercise and state
        # ══════════════════════════════════════════════════════════════════════

        self._SQUAT_POSITIVE = {
            's1': "Great starting position! Begin lowering slowly",
            's2': "Nice — keep going down, control the descent",
            's3': "Perfect depth! Now drive back up through your heels",
        }

        self._PUSHUP_POSITIVE = {
            's1': "Solid plank! Lower yourself in a controlled way",
            's2': "Good — keep lowering your chest to the ground",
            's3': "Chest down! Now push the ground away and come back up",
        }

        self._LUNGE_POSITIVE = {
            's1': "Good stance! Step forward and bend your front knee",
            's2': "Looking good — keep lowering with control",
            's3': "Great depth! Drive through your front heel to come back up",
        }

        # Rep-completion flash messages
        self._REP_GOOD_MSGS = {
            "squat":  "Great squat! Keep it up!",
            "pushup": "Excellent push-up! Stay strong!",
            "lunge":  "Perfect lunge! Great form!",
        }
        self._REP_BAD_MSGS = {
            "squat":  "Rep counted — watch your form next time",
            "pushup": "Rep counted — focus on keeping straight",
            "lunge":  "Rep counted — try to stay more upright",
        }

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════════════

    def process(self, frame, pose):
        """
        Process a single RGB frame.  Returns annotated RGB frame.
        Signature is identical to the original ProcessFrame.process().
        """
        frame_height, frame_width, _ = frame.shape

        # Step 1: Flip (live-stream mode only)
        if self.flip_frame:
            frame = cv2.flip(frame, 1)

        keypoints = pose.process(frame)

        # Step 2: No pose detected
        if getattr(keypoints, 'pose_landmarks', None) is None:
            self.state_tracker['curr_state'] = None
            self.state_tracker['prev_state'] = None
            self.state_tracker['INACTIVE_TIME'] += (
                time.perf_counter() - self.state_tracker['start_inactive_time'])
            self.state_tracker['start_inactive_time'] = time.perf_counter()

            if self.state_tracker['INACTIVE_TIME'] > self.thresholds['INACTIVE_THRESH']:
                self._reset_counters()

            self._draw_counters(frame)
            draw_text(
                frame,
                "Step into frame so I can see you!",
                (max(10, int(frame_width // 2) - 195), int(frame_height // 2)),
                text_color=(255, 220, 80), text_color_bg=(40, 40, 40), font_scale=0.7
            )
            return frame

        # Reset inactivity timer
        self.state_tracker['start_inactive_time'] = time.perf_counter()
        self.state_tracker['INACTIVE_TIME'] = 0.0

        nose, left_joints, right_joints = get_landmark_features(
            keypoints.pose_landmarks, frame.shape)

        # Step 3: Camera-alignment check (shared by all exercises)
        offset_angle = find_angle(left_joints[0], right_joints[0], nose)

        if offset_angle > self.thresholds['OFFSET_THRESH']:
            self.state_tracker['INACTIVE_TIME_FRONT'] += (
                time.perf_counter() - self.state_tracker['start_inactive_time_front'])
            self.state_tracker['start_inactive_time_front'] = time.perf_counter()

            if self.state_tracker['INACTIVE_TIME_FRONT'] > self.thresholds['INACTIVE_THRESH']:
                self._reset_counters()

            self.state_tracker['curr_state'] = None
            self.state_tracker['prev_state'] = None
            self._draw_counters(frame)
            draw_text(
                frame,
                "Turn sideways - I need to see your full side profile",
                (max(10, int(frame_width // 2) - 250), 150),
                text_color=(255, 255, 255),
                text_color_bg=(180, 50, 30),
                font_scale=0.65
            )
            return frame

        self.state_tracker['start_inactive_time_front'] = time.perf_counter()
        self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0

        # Step 3.5: Exercise Authenticity Check (Orientation)
        # Prevents counting squats in push-up mode and vice versa.
        left_diff_vert  = abs(left_joints[5][1]  - left_joints[0][1])
        right_diff_vert = abs(right_joints[5][1] - right_joints[0][1])
        vis_joints = left_joints if left_diff_vert > right_diff_vert else right_joints
        
        shoulder = vis_joints[0]
        ankle    = vis_joints[5]
        
        dx = abs(ankle[0] - shoulder[0])
        dy = abs(ankle[1] - shoulder[1])
        
        # Posture checks
        is_horizontal = dx > dy * 0.8
        is_vertical   = dy > dx * 0.6
        
        # Stance checks (distinguish squats vs lunges)
        ankle_dist_x = abs(left_joints[5][0] - right_joints[5][0])
        is_split_stance = ankle_dist_x > dy * 0.25
        
        l_knee_angle = find_angle(left_joints[3], left_joints[5], left_joints[4])
        r_knee_angle = find_angle(right_joints[3], right_joints[5], right_joints[4])
        knees_bent = (l_knee_angle < 150) or (r_knee_angle < 150)
        
        wrong_exercise = False
        msg = ""

        if self.exercise_mode == "pushup" and not is_horizontal:
            wrong_exercise = True
            msg = "Please get into a horizontal position for Push-ups"
            
        elif self.exercise_mode == "squat":
            if not is_vertical:
                wrong_exercise = True
                msg = "Please stand upright for Squats"
            elif is_split_stance:
                wrong_exercise = True
                msg = "Keep feet together — don't split your stance"
                
        elif self.exercise_mode == "lunge":
            if not is_vertical:
                wrong_exercise = True
                msg = "Please stand upright for Lunges"
            elif knees_bent and not is_split_stance:
                wrong_exercise = True
                msg = "Take a wide step forward or backward for Lunges"

        if wrong_exercise:
            self.state_tracker['curr_state'] = None
            self.state_tracker['prev_state'] = None
            self._draw_counters(frame)
            draw_text(
                frame,
                msg,
                (max(10, int(frame_width // 2) - 220), 200),
                text_color=(255, 255, 255),
                text_color_bg=(180, 50, 30),
                font_scale=0.65
            )
            return frame

        # Step 4: Route to exercise-specific processor
        if self.exercise_mode == "pushup":
            frame = self._process_pushup(frame, left_joints, right_joints,
                                         frame_width, frame_height)
        elif self.exercise_mode == "lunge":
            frame = self._process_lunge(frame, left_joints, right_joints,
                                        frame_width, frame_height)
        else:
            frame = self._process_squat(frame, left_joints, right_joints,
                                        frame_width, frame_height)

        return frame

    # ══════════════════════════════════════════════════════════════════════════
    # SQUAT LOGIC  (original code — behaviour unchanged)
    # ══════════════════════════════════════════════════════════════════════════

    def _process_squat(self, frame, left_joints, right_joints, frame_width, frame_height):
        """Squat analysis — behaviour is identical to the first release."""

        # Choose body side (bigger vertical span = the visible side)
        left_diff  = abs(left_joints[-1][1]  - left_joints[0][1])
        right_diff = abs(right_joints[-1][1] - right_joints[0][1])
        joints = left_joints if left_diff > right_diff else right_joints

        shldr, _, _, hip, knee, ankle, _ = joints

        hip_ref   = (hip[0],   hip[1]   - 100)
        knee_ref  = (knee[0],  knee[1]  - 100)
        ankle_ref = (ankle[0], ankle[1] - 100)

        hip_vert_angle   = find_angle(shldr, hip_ref,  hip)
        knee_vert_angle  = find_angle(hip,   knee_ref, knee)
        ankle_vert_angle = find_angle(knee,  ankle_ref, ankle)

        # State detection
        curr_state = self._get_squat_state(knee_vert_angle)
        self.state_tracker['curr_state'] = curr_state

        # Skeleton + angle annotations
        self._draw_skeleton(frame, [shldr, hip, knee, ankle])
        cv2.line(frame, shldr, hip,   (0, 255, 0), 3)
        cv2.line(frame, hip,   knee,  (0, 255, 0), 3)
        cv2.line(frame, knee,  ankle, (0, 255, 0), 3)
        draw_dotted_line(frame, hip,   hip_ref,   (200, 200, 200), gap=8)
        draw_dotted_line(frame, knee,  knee_ref,  (200, 200, 200), gap=8)
        draw_dotted_line(frame, ankle, ankle_ref, (200, 200, 200), gap=8)
        cv2.putText(frame, f"{int(hip_vert_angle)}\xb0",
                    (hip[0] + 12, hip[1]),   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 50), 2)
        cv2.putText(frame, f"{int(knee_vert_angle)}\xb0",
                    (knee[0] + 12, knee[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 50), 2)
        cv2.putText(frame, f"{int(ankle_vert_angle)}\xb0",
                    (ankle[0] + 12, ankle[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 50), 2)

        # Update state sequence
        if curr_state is not None:
            self._update_state_sequence(curr_state)

        # ── Correction feedback flags ─────────────────────────────────────────
        posture_ok_this_frame = True

        if curr_state in ['s2', 's3']:
            if hip_vert_angle > self.thresholds['HIP_THRESH'][1]:
                self.state_tracker['DISPLAY_TEXT'][0] = True
            elif hip_vert_angle < self.thresholds['HIP_THRESH'][0]:
                self.state_tracker['DISPLAY_TEXT'][1] = True

            if ankle_vert_angle > self.thresholds['ANKLE_THRESH']:
                self.state_tracker['DISPLAY_TEXT'][2] = True
                self.state_tracker['INCORRECT_POSTURE'] = True
                posture_ok_this_frame = False

            if knee_vert_angle > self.thresholds['KNEE_THRESH'][2]:
                self.state_tracker['DISPLAY_TEXT'][3] = True
                self.state_tracker['INCORRECT_POSTURE'] = True
                posture_ok_this_frame = False

            if (self.thresholds['KNEE_THRESH'][0]
                    < knee_vert_angle < self.thresholds['KNEE_THRESH'][1]):
                self.state_tracker['LOWER_HIPS'] = True
            else:
                self.state_tracker['LOWER_HIPS'] = False
        else:
            self.state_tracker['LOWER_HIPS'] = False

        # ── Positive coaching hint ────────────────────────────────────────────
        if curr_state is not None and posture_ok_this_frame:
            hint = self._SQUAT_POSITIVE.get(curr_state, '')
            if hint:
                self.state_tracker['POSITIVE_MSG']    = hint
                self.state_tracker['POSITIVE_FRAMES'] = self._POSITIVE_DISPLAY_FRAMES

        # ── Rep counting + flash message ──────────────────────────────────────
        if (curr_state == 's1' and self.state_tracker['prev_state'] == 's2'):
            if len(self.state_tracker['state_seq']) > 0:
                self.state_tracker['REP_COUNT'] += 1
                if not self.state_tracker['INCORRECT_POSTURE']:
                    self._flash_rep(good=True)
                else:
                    self._flash_rep(good=False)

            self.state_tracker['state_seq'] = []
            self.state_tracker['INCORRECT_POSTURE'] = False

        if curr_state is not None:
            self.state_tracker['prev_state'] = curr_state

        # Advance feedback frame counters (4 slots for squat)
        self._advance_feedback_counters(4)

        # Draw all overlays
        self._draw_counters(frame)
        self._draw_state_label(frame, curr_state, frame_width)
        self._draw_squat_feedback(frame, frame_width, frame_height)
        self._draw_positive_overlay(frame, frame_width, frame_height)
        self._draw_rep_flash(frame, frame_width, frame_height)
        return frame

    # ══════════════════════════════════════════════════════════════════════════
    # PUSH-UP LOGIC
    # ══════════════════════════════════════════════════════════════════════════

    def _process_pushup(self, frame, left_joints, right_joints, frame_width, frame_height):
        """
        Push-up analysis using side view.

        Key angles:
          elbow_angle = Shoulder - Elbow - Wrist  (arm movement)
          body_angle  = Shoulder - Hip   - Ankle  (body straightness)
          leg_angle   = Hip      - Knee  - Ankle  (legs straight)

        States:
          s1: arms extended (elbow >= 150, body >= 170, legs >= 170)
          s2: transition    (90 < elbow < 150, body >= 170, legs >= 170)
          s3: bottom        (elbow <= 90, body >= 170, legs >= 170)
        """
        th = self.thresholds

        # Choose visible side (bigger vertical span)
        left_diff  = abs(left_joints[-1][1]  - left_joints[0][1])
        right_diff = abs(right_joints[-1][1] - right_joints[0][1])
        joints = left_joints if left_diff > right_diff else right_joints

        shldr, elbow, wrist, hip, knee, ankle, _ = joints

        # ── Compute angles ────────────────────────────────────────────────────
        elbow_angle = find_angle(shldr, wrist, elbow)   # Shoulder-Elbow-Wrist
        body_angle  = find_angle(shldr, ankle, hip)     # Shoulder-Hip-Ankle
        leg_angle   = find_angle(hip,   ankle, knee)    # Hip-Knee-Ankle

        # ── Draw skeleton ─────────────────────────────────────────────────────
        self._draw_skeleton(frame, [shldr, elbow, wrist, hip, knee, ankle])
        cv2.line(frame, shldr, elbow, (0, 255, 0), 3)
        cv2.line(frame, elbow, wrist, (0, 255, 0), 3)
        cv2.line(frame, shldr, hip,   (0, 255, 0), 3)
        cv2.line(frame, hip,   knee,  (0, 255, 0), 3)
        cv2.line(frame, knee,  ankle, (0, 255, 0), 3)

        cv2.putText(frame, f"{int(elbow_angle)}\xb0",
                    (elbow[0] + 10, elbow[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 220, 50), 2)
        cv2.putText(frame, f"{int(body_angle)}\xb0",
                    (hip[0] + 10,   hip[1]),   cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 220, 50), 2)
        cv2.putText(frame, f"{int(leg_angle)}\xb0",
                    (knee[0] + 10,  knee[1]),  cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 220, 50), 2)

        # ── State detection  (tolerance applied here — not to feedback flags below) ──
        # _lo() relaxes a minimum: e.g. 170 -> 153 deg at 10% tolerance
        # _hi() relaxes a maximum: e.g.  90 ->  99 deg at 10% tolerance
        body_ok = body_angle >= self._lo(th['BODY_ANGLE_MIN'])
        legs_ok = leg_angle  >= self._lo(th['LEG_ANGLE_MIN'])

        if body_ok and legs_ok:
            if elbow_angle >= self._lo(th['ELBOW_THRESH']['TOP']):
                curr_state = 's1'
            elif elbow_angle <= self._hi(th['ELBOW_THRESH']['BOTTOM']):
                curr_state = 's3'
            else:
                curr_state = 's2'
        else:
            curr_state = None   # Invalid posture

        self.state_tracker['curr_state'] = curr_state

        if curr_state is not None:
            self._update_state_sequence(curr_state)

        # ── Correction feedback flags ─────────────────────────────────────────
        posture_ok_this_frame = True

        if leg_angle < th['LEG_ANGLE_MIN']:
            self.state_tracker['DISPLAY_TEXT'][0] = True
            self.state_tracker['INCORRECT_POSTURE'] = True
            posture_ok_this_frame = False

        if body_angle < th['BODY_ANGLE_MIN']:
            self.state_tracker['DISPLAY_TEXT'][1] = True
            self.state_tracker['INCORRECT_POSTURE'] = True
            posture_ok_this_frame = False

        # "Go lower" — active while going down in transition
        if (curr_state == 's2'
                and self.state_tracker['prev_state'] in ('s1', 's2')
                and elbow_angle > th['ELBOW_THRESH']['TOP'] * 0.75):
            self.state_tracker['DISPLAY_TEXT'][2] = True

        # "Fully extend" — returning to top but not fully extended
        if (self.state_tracker['prev_state'] == 's2'
                and curr_state == 's1'
                and elbow_angle < th['ELBOW_THRESH']['TOP']):
            self.state_tracker['DISPLAY_TEXT'][3] = True

        # ── Positive coaching hint ────────────────────────────────────────────
        if curr_state is not None and posture_ok_this_frame:
            hint = self._PUSHUP_POSITIVE.get(curr_state, '')
            if hint:
                self.state_tracker['POSITIVE_MSG']    = hint
                self.state_tracker['POSITIVE_FRAMES'] = self._POSITIVE_DISPLAY_FRAMES

        # ── Rep counting ──────────────────────────────────────────────────────
        if (curr_state == 's1' and self.state_tracker['prev_state'] == 's2'):
            if len(self.state_tracker['state_seq']) > 0:
                self.state_tracker['REP_COUNT'] += 1
                if not self.state_tracker['INCORRECT_POSTURE']:
                    self._flash_rep(good=True)
                else:
                    self._flash_rep(good=False)

            self.state_tracker['state_seq'] = []
            self.state_tracker['INCORRECT_POSTURE'] = False

        if curr_state is not None:
            self.state_tracker['prev_state'] = curr_state

        self._advance_feedback_counters(4)

        self._draw_counters(frame)
        self._draw_state_label(frame, curr_state, frame_width)
        self._draw_generic_feedback(frame, self._PUSHUP_FEEDBACK, 4, frame_width, frame_height)
        self._draw_positive_overlay(frame, frame_width, frame_height)
        self._draw_rep_flash(frame, frame_width, frame_height)
        return frame

    # ══════════════════════════════════════════════════════════════════════════
    # LUNGE LOGIC
    # ══════════════════════════════════════════════════════════════════════════

    def _process_lunge(self, frame, left_joints, right_joints, frame_width, frame_height):
        """
        Lunge analysis using side view.

        Front/back leg detection:
          Ankle whose x-position is further in the forward direction is the front leg.

        Key angles:
          front_knee_angle = front Hip - front Knee - front Ankle
          back_knee_angle  = back  Hip - back  Knee - back  Ankle
          torso_angle      = Shoulder  - Hip - Ankle (visible side)
        """
        th = self.thresholds

        # ── Unpack joints ─────────────────────────────────────────────────────
        l_shldr, _, _, l_hip, l_knee, l_ankle, _ = left_joints
        r_shldr, _, _, r_hip, r_knee, r_ankle, _ = right_joints

        # ── Detect front vs back leg by ankle x-position ──────────────────────
        if r_ankle[0] > l_ankle[0]:
            front_shldr, front_hip, front_knee, front_ankle = r_shldr, r_hip, r_knee, r_ankle
            back_shldr,  back_hip,  back_knee,  back_ankle  = l_shldr, l_hip, l_knee, l_ankle
        else:
            front_shldr, front_hip, front_knee, front_ankle = l_shldr, l_hip, l_knee, l_ankle
            back_shldr,  back_hip,  back_knee,  back_ankle  = r_shldr, r_hip, r_knee, r_ankle

        # ── Torso angle (visible/closer side) ─────────────────────────────────
        left_diff  = abs(left_joints[-1][1]  - left_joints[0][1])
        right_diff = abs(right_joints[-1][1] - right_joints[0][1])
        vis_joints = left_joints if left_diff > right_diff else right_joints
        vis_shldr, _, _, vis_hip, _, vis_ankle, _ = vis_joints
        torso_angle = find_angle(vis_shldr, vis_ankle, vis_hip)

        # ── Knee angles ───────────────────────────────────────────────────────
        front_knee_angle = find_angle(front_hip, front_ankle, front_knee)
        back_knee_angle  = find_angle(back_hip,  back_ankle,  back_knee)

        # ── Knee-over-toe check ───────────────────────────────────────────────
        knee_over_toe = (front_knee[0] - front_ankle[0]) > 30

        # ── Draw skeleton ─────────────────────────────────────────────────────
        joints_to_draw = [front_hip, front_knee, front_ankle,
                          back_hip,  back_knee,  back_ankle, vis_shldr]
        self._draw_skeleton(frame, joints_to_draw)
        cv2.line(frame, vis_shldr,  front_hip,   (0, 200, 255), 3)
        cv2.line(frame, front_hip,  front_knee,  (0, 255, 0),   3)
        cv2.line(frame, front_knee, front_ankle, (0, 255, 0),   3)
        cv2.line(frame, back_hip,   back_knee,   (0, 200, 200), 3)
        cv2.line(frame, back_knee,  back_ankle,  (0, 200, 200), 3)

        cv2.putText(frame, f"FK:{int(front_knee_angle)}\xb0",
                    (front_knee[0] + 10, front_knee[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 50), 2)
        cv2.putText(frame, f"BK:{int(back_knee_angle)}\xb0",
                    (back_knee[0] + 10, back_knee[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 2)
        cv2.putText(frame, f"T:{int(torso_angle)}\xb0",
                    (vis_hip[0] + 10, vis_hip[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 80), 2)

        # ── State detection ───────────────────────────────────────────────────
        fk = th['FRONT_KNEE']
        bk = th['BACK_KNEE']

        torso_invalid = torso_angle < self._lo(th['TORSO_INVALID'])

        if torso_invalid or knee_over_toe:
            curr_state = None
        elif (front_knee_angle >= self._lo(fk['TOP'])
              and back_knee_angle >= self._lo(bk['TOP'])
              and torso_angle >= self._lo(th['TORSO_MIN_S1'])):
            curr_state = 's1'
        elif (self._lo(fk['TRANS_MIN']) < front_knee_angle < self._hi(fk['TRANS_MAX'])
              and torso_angle >= self._lo(th['TORSO_MIN_S2'])):
            curr_state = 's2'
        elif (self._lo(fk['BOTTOM_MIN']) <= front_knee_angle <= self._hi(fk['BOTTOM_MAX'])
              and self._lo(bk['BOTTOM_MIN']) <= back_knee_angle <= self._hi(bk['BOTTOM_MAX'])
              and torso_angle >= self._lo(th['TORSO_MIN_S1'])):
            curr_state = 's3'
        else:
            curr_state = None

        self.state_tracker['curr_state'] = curr_state

        if curr_state is not None:
            self._update_state_sequence(curr_state)

        # ── Correction feedback flags ─────────────────────────────────────────
        posture_ok_this_frame = True

        if torso_angle < th['TORSO_MIN_S1']:
            self.state_tracker['DISPLAY_TEXT'][0] = True
            self.state_tracker['INCORRECT_POSTURE'] = True
            posture_ok_this_frame = False

        if (curr_state == 's2'
                and self.state_tracker['prev_state'] in ('s1', 's2')
                and front_knee_angle > fk['TRANS_MAX'] * 0.85):
            self.state_tracker['DISPLAY_TEXT'][1] = True

        if front_knee_angle < fk['TOO_LOW']:
            self.state_tracker['DISPLAY_TEXT'][2] = True
            self.state_tracker['INCORRECT_POSTURE'] = True
            posture_ok_this_frame = False

        if back_knee_angle < bk['BOTTOM_MIN']:
            self.state_tracker['DISPLAY_TEXT'][3] = True
            self.state_tracker['INCORRECT_POSTURE'] = True
            posture_ok_this_frame = False

        if knee_over_toe:
            self.state_tracker['DISPLAY_TEXT'][4] = True
            self.state_tracker['INCORRECT_POSTURE'] = True
            posture_ok_this_frame = False

        if (torso_angle < th['TORSO_MIN_S2']
                and curr_state in (None, 's2', 's3')):
            self.state_tracker['DISPLAY_TEXT'][5] = True

        # ── Positive coaching hint ────────────────────────────────────────────
        if curr_state is not None and posture_ok_this_frame:
            hint = self._LUNGE_POSITIVE.get(curr_state, '')
            if hint:
                self.state_tracker['POSITIVE_MSG']    = hint
                self.state_tracker['POSITIVE_FRAMES'] = self._POSITIVE_DISPLAY_FRAMES

        # ── Rep counting ──────────────────────────────────────────────────────
        if (curr_state == 's1' and self.state_tracker['prev_state'] == 's2'):
            if len(self.state_tracker['state_seq']) > 0:
                self.state_tracker['REP_COUNT'] += 1
                if not self.state_tracker['INCORRECT_POSTURE']:
                    self._flash_rep(good=True)
                else:
                    self._flash_rep(good=False)

            self.state_tracker['state_seq'] = []
            self.state_tracker['INCORRECT_POSTURE'] = False

        if curr_state is not None:
            self.state_tracker['prev_state'] = curr_state

        self._advance_feedback_counters(6)

        self._draw_counters(frame)
        self._draw_state_label(frame, curr_state, frame_width)
        self._draw_generic_feedback(frame, self._LUNGE_FEEDBACK, 6, frame_width, frame_height)
        self._draw_positive_overlay(frame, frame_width, frame_height)
        self._draw_rep_flash(frame, frame_width, frame_height)
        return frame

    # ══════════════════════════════════════════════════════════════════════════
    # SHARED HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _reset_counters(self):
        """Reset rep counters and state sequence (called on prolonged inactivity)."""
        self.state_tracker['REP_COUNT']       = 0
        self.state_tracker['SUGGESTIONS']     = []
        self.state_tracker['state_seq']       = []

    def _lo(self, val):
        """
        Relax a MINIMUM angle threshold downward by the TOLERANCE factor.
        e.g. body must be >= 170 deg  ->  with 10% tolerance becomes >= 153 deg.
        Only push-up and lunge threshold dicts carry a TOLERANCE key;
        squat thresholds have no TOLERANCE so this returns val unchanged.
        """
        tol = self.thresholds.get('TOLERANCE', 0.0)
        return val * (1.0 - tol)

    def _hi(self, val):
        """
        Relax a MAXIMUM angle threshold upward by the TOLERANCE factor.
        e.g. elbow must be <= 90 deg  ->  with 10% tolerance becomes <= 99 deg.
        """
        tol = self.thresholds.get('TOLERANCE', 0.0)
        return val * (1.0 + tol)

    def _get_squat_state(self, knee_angle):
        """Map vertical knee angle to squat state (s1/s2/s3)."""
        hnv = self.thresholds['HIP_KNEE_VERT']
        if hnv['NORMAL'][0] <= knee_angle <= hnv['NORMAL'][1]:
            return 's1'
        elif hnv['TRANS'][0] <= knee_angle <= hnv['TRANS'][1]:
            return 's2'
        elif hnv['PASS'][0] <= knee_angle <= hnv['PASS'][1]:
            return 's3'
        return None

    def _update_state_sequence(self, state):
        """
        Shared state sequence updater. Target sequence is always [s2, s3, s2].
        """
        seq = self.state_tracker['state_seq']
        if state == 's2':
            if len(seq) == 0:
                seq.append(state)
            elif 's3' in seq and seq.count('s2') == 1:
                seq.append(state)
        elif state == 's3':
            if state not in seq and 's2' in seq:
                seq.append(state)

    def _advance_feedback_counters(self, num_slots):
        """Tick down and auto-clear feedback display flags after CNT_FRAME_THRESH frames."""
        for i in range(num_slots):
            if self.state_tracker['DISPLAY_TEXT'][i]:
                self.state_tracker['COUNT_FRAMES'][i] += 1
                if self.state_tracker['COUNT_FRAMES'][i] >= self.thresholds['CNT_FRAME_THRESH']:
                    self.state_tracker['DISPLAY_TEXT'][i] = False
                    self.state_tracker['COUNT_FRAMES'][i] = 0

    def _flash_rep(self, good: bool):
        """Set the rep-completion flash message and timer."""
        if good:
            self.state_tracker['REP_FLASH_MSG']    = self._REP_GOOD_MSGS.get(self.exercise_mode, "Great rep!")
        else:
            self.state_tracker['REP_FLASH_MSG']    = "Rep completed!"
        self.state_tracker['REP_FLASH_FRAMES'] = self._REP_FLASH_FRAMES

    def _draw_skeleton(self, frame, points):
        """Draw joint circles at each keypoint."""
        for pt in points:
            cv2.circle(frame, pt, 7, (0, 255, 0), -1)

    # ── Counter panel ─────────────────────────────────────────────────────────

    def _draw_counters(self, frame):
        """Draw total rep counter in top-left corner."""
        label = f"{self.exercise_mode.title()} Reps"
        draw_text(frame, f"{label} : {self.state_tracker['REP_COUNT']}",
                  (20, 50), text_color=(255, 255, 255), text_color_bg=(20, 20, 20), font_scale=0.8)

    # ── State phase label (top-right) ─────────────────────────────────────────

    def _draw_state_label(self, frame, curr_state, frame_width):
        """
        Show a human-readable phase label in the top-right corner so the
        user can always see which phase the system thinks they are in.
        """
        PHASE_LABELS = {
            's1': 'Standing',
            's2': 'Transitioning',
            's3': 'Bottom Position',
        }
        label = PHASE_LABELS.get(curr_state, 'Adjusting...')
        color = {
            's1': (80, 200, 80),
            's2': (255, 200, 50),
            's3': (80, 180, 255),
        }.get(curr_state, (160, 160, 160))

        text = f"Phase: {label}"
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        x = max(10, frame_width - tw - 30)
        draw_text(frame, text, (x, 50),
                  text_color=color, text_color_bg=(20, 20, 20), font_scale=0.65)

    # ── Positive coaching overlay (top-centre) ────────────────────────────────

    def _draw_positive_overlay(self, frame, frame_width, frame_height):
        """
        Show the current positive coaching hint centred near the top of the frame.
        Automatically ticks down its display counter each frame.
        """
        if self.state_tracker['POSITIVE_FRAMES'] <= 0:
            return

        msg = self.state_tracker['POSITIVE_MSG']
        (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        x = max(10, (frame_width - tw) // 2)
        draw_text(frame, msg, (x, 145),
                  text_color=(255, 255, 255),
                  text_color_bg=(30, 100, 40),
                  font_scale=0.65)

        self.state_tracker['POSITIVE_FRAMES'] -= 1

    # ── Rep-completion flash (centre of frame) ────────────────────────────────

    def _draw_rep_flash(self, frame, frame_width, frame_height):
        """
        Flash a brief congratulations / correction note after each rep is counted.
        """
        if self.state_tracker['REP_FLASH_FRAMES'] <= 0:
            return

        msg = self.state_tracker['REP_FLASH_MSG']
        good = ('Great' in msg or 'Excellent' in msg or 'Perfect' in msg)
        bg   = (30, 120, 30) if good else (120, 120, 120)

        (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        x = max(10, (frame_width - tw) // 2)
        y = frame_height // 2 - 30
        draw_text(frame, msg, (x, y),
                  text_color=(255, 255, 255), text_color_bg=bg, font_scale=0.75)

        self.state_tracker['REP_FLASH_FRAMES'] -= 1

    # ── Squat-specific feedback drawer (preserves original LOWER_HIPS hint) ──

    def _draw_squat_feedback(self, frame, frame_width, frame_height):
        """Draw active squat correction messages at the bottom of the frame."""
        active_msgs = [self._SQUAT_FEEDBACK[i]
                       for i in range(4) if self.state_tracker['DISPLAY_TEXT'][i]]
        if self.state_tracker['LOWER_HIPS']:
            active_msgs.append("Almost there — try to go just a little deeper")

        y_start = frame_height - 30 - len(active_msgs) * 38
        for idx, msg in enumerate(active_msgs):
            if msg not in self.state_tracker['SUGGESTIONS']:
                self.state_tracker['SUGGESTIONS'].append(msg)
            
            draw_text(frame, msg,
                      (20, y_start + idx * 38),
                      text_color=(255, 255, 255),
                      text_color_bg=(30, 30, 180),
                      font_scale=0.65)

    def _draw_generic_feedback(self, frame, feedback_dict, num_slots,
                               frame_width, frame_height):
        """Draw active correction messages for push-up and lunge modes."""
        active_msgs = [feedback_dict[i]
                       for i in range(num_slots) if self.state_tracker['DISPLAY_TEXT'][i]]

        y_start = frame_height - 30 - len(active_msgs) * 38
        for idx, msg in enumerate(active_msgs):
            if msg not in self.state_tracker['SUGGESTIONS']:
                self.state_tracker['SUGGESTIONS'].append(msg)
                
            draw_text(frame, msg,
                      (20, y_start + idx * 38),
                      text_color=(255, 255, 255),
                      text_color_bg=(30, 30, 180),
                      font_scale=0.65)
