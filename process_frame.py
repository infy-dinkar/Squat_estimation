import time
import cv2
import numpy as np
from utils import find_angle, get_landmark_features, draw_text, draw_dotted_line


class ProcessFrame:
    def __init__(self, thresholds, flip_frame=False):
        self.flip_frame = flip_frame
        self.thresholds = thresholds

        self.state_tracker = {
            'state_seq': [],
            'start_inactive_time': time.perf_counter(),
            'start_inactive_time_front': time.perf_counter(),
            'INACTIVE_TIME': 0.0,
            'INACTIVE_TIME_FRONT': 0.0,
            'DISPLAY_TEXT': np.full((4,), False),
            'COUNT_FRAMES': np.zeros((4,), dtype=np.int64),
            'LOWER_HIPS': False,
            'INCORRECT_POSTURE': False,
            'prev_state': None,
            'curr_state': None,
            'SQUAT_COUNT': 0,
            'IMPROPER_SQUAT': 0
        }

        # Human-readable feedback messages
        self.FEEDBACK_MESSAGES = {
            0: "⚠ Lean your upper body forward a bit",
            1: "⚠ You are leaning too far forward",
            2: "⚠ Your knee is going past your toes",
            3: "⚠ You are squatting too deep"
        }

    def _get_state(self, knee_angle):
        hip_knee_vert = self.thresholds['HIP_KNEE_VERT']
        if hip_knee_vert['NORMAL'][0] <= knee_angle <= hip_knee_vert['NORMAL'][1]:
            return 's1'
        elif hip_knee_vert['TRANS'][0] <= knee_angle <= hip_knee_vert['TRANS'][1]:
            return 's2'
        elif hip_knee_vert['PASS'][0] <= knee_angle <= hip_knee_vert['PASS'][1]:
            return 's3'
        return None

    def _update_state_sequence(self, state):
        seq = self.state_tracker['state_seq']
        if state == 's2':
            if len(seq) == 0:
                seq.append(state)
            elif 's3' in seq and seq.count('s2') == 1:
                seq.append(state)
        elif state == 's3':
            if state not in seq and 's2' in seq:
                seq.append(state)

    def _draw_annotations(self, frame, shldr, hip, knee, ankle,
                          hip_vert_angle, knee_vert_angle, ankle_vert_angle,
                          hip_ref, knee_ref, ankle_ref):
        """Draw skeleton, reference lines and angle values on the frame."""
        # Skeleton joints
        for pt in [shldr, hip, knee, ankle]:
            cv2.circle(frame, pt, 7, (0, 255, 0), -1)

        # Body lines
        cv2.line(frame, shldr, hip, (0, 255, 0), 3)
        cv2.line(frame, hip, knee, (0, 255, 0), 3)
        cv2.line(frame, knee, ankle, (0, 255, 0), 3)

        # Vertical reference lines
        draw_dotted_line(frame, hip, hip_ref, (200, 200, 200), gap=8)
        draw_dotted_line(frame, knee, knee_ref, (200, 200, 200), gap=8)
        draw_dotted_line(frame, ankle, ankle_ref, (200, 200, 200), gap=8)

        # Angle values
        cv2.putText(frame, f"{int(hip_vert_angle)}\xb0",
                    (hip[0] + 12, hip[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 50), 2)
        cv2.putText(frame, f"{int(knee_vert_angle)}\xb0",
                    (knee[0] + 12, knee[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 50), 2)
        cv2.putText(frame, f"{int(ankle_vert_angle)}\xb0",
                    (ankle[0] + 12, ankle[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 50), 2)

    def _draw_counters(self, frame):
        draw_text(frame, f"Good Squats : {self.state_tracker['SQUAT_COUNT']}",
                  (20, 50), text_color=(50, 255, 50), text_color_bg=(20, 20, 20), font_scale=0.8)
        draw_text(frame, f"Bad Squats : {self.state_tracker['IMPROPER_SQUAT']}",
                  (20, 95), text_color=(255, 80, 80), text_color_bg=(20, 20, 20), font_scale=0.8)

    def _draw_feedback(self, frame, frame_width, frame_height):
        """Draw active feedback messages in a readable panel at the bottom."""
        active_msgs = [self.FEEDBACK_MESSAGES[i]
                       for i in range(4) if self.state_tracker['DISPLAY_TEXT'][i]]

        if self.state_tracker['LOWER_HIPS']:
            active_msgs.append("💡 Try to go a little deeper")

        y_start = frame_height - 30 - len(active_msgs) * 38
        for idx, msg in enumerate(active_msgs):
            draw_text(frame, msg,
                      (20, y_start + idx * 38),
                      text_color=(255, 255, 255),
                      text_color_bg=(30, 30, 180),
                      font_scale=0.65)

    def process(self, frame, pose):
        frame_height, frame_width, _ = frame.shape

        # ── Step 1: Flip first (live mode only) so all drawing is correct-way-up ──
        if self.flip_frame:
            frame = cv2.flip(frame, 1)

        keypoints = pose.process(frame)

        # ── Step 2: No pose detected ──
        if getattr(keypoints, 'pose_landmarks', None) is None:
            self.state_tracker['curr_state'] = None
            self.state_tracker['prev_state'] = None
            self.state_tracker['INACTIVE_TIME'] += (
                time.perf_counter() - self.state_tracker['start_inactive_time'])
            self.state_tracker['start_inactive_time'] = time.perf_counter()

            if self.state_tracker['INACTIVE_TIME'] > self.thresholds['INACTIVE_THRESH']:
                self.state_tracker['SQUAT_COUNT'] = 0
                self.state_tracker['IMPROPER_SQUAT'] = 0
                self.state_tracker['state_seq'] = []

            self._draw_counters(frame)
            draw_text(frame, "No person detected in frame",
                      (int(frame_width // 2) - 160, int(frame_height // 2)),
                      text_color=(255, 200, 50), text_color_bg=(40, 40, 40), font_scale=0.7)
            return frame

        # Reset side-inactivity timer
        self.state_tracker['start_inactive_time'] = time.perf_counter()
        self.state_tracker['INACTIVE_TIME'] = 0.0

        nose, left_joints, right_joints = get_landmark_features(
            keypoints.pose_landmarks, frame.shape)

        # ── Step 3: Camera alignment check ──
        offset_angle = find_angle(left_joints[0], right_joints[0], nose)

        if offset_angle > self.thresholds['OFFSET_THRESH']:
            self.state_tracker['INACTIVE_TIME_FRONT'] += (
                time.perf_counter() - self.state_tracker['start_inactive_time_front'])
            self.state_tracker['start_inactive_time_front'] = time.perf_counter()

            if self.state_tracker['INACTIVE_TIME_FRONT'] > self.thresholds['INACTIVE_THRESH']:
                self.state_tracker['SQUAT_COUNT'] = 0
                self.state_tracker['IMPROPER_SQUAT'] = 0
                self.state_tracker['state_seq'] = []

            self.state_tracker['curr_state'] = None
            self.state_tracker['prev_state'] = None
            self._draw_counters(frame)
            draw_text(
                frame,
                "Stand sideways — camera needs a side view",
                (max(10, int(frame_width // 2) - 220), 55),
                text_color=(255, 255, 255),
                text_color_bg=(180, 30, 30),
                font_scale=0.7
            )
            return frame

        self.state_tracker['start_inactive_time_front'] = time.perf_counter()
        self.state_tracker['INACTIVE_TIME_FRONT'] = 0.0

        # ── Step 4: Choose body side (bigger vertical span = visible side) ──
        left_diff = abs(left_joints[-1][1] - left_joints[0][1])
        right_diff = abs(right_joints[-1][1] - right_joints[0][1])
        joints = left_joints if left_diff > right_diff else right_joints

        shldr, _, _, hip, knee, ankle, _ = joints

        hip_ref   = (hip[0],   hip[1]   - 100)
        knee_ref  = (knee[0],  knee[1]  - 100)
        ankle_ref = (ankle[0], ankle[1] - 100)

        hip_vert_angle   = find_angle(shldr, hip_ref,  hip)
        knee_vert_angle  = find_angle(hip,   knee_ref, knee)
        ankle_vert_angle = find_angle(knee,  ankle_ref, ankle)

        # ── Step 5: State detection ──
        self.state_tracker['curr_state'] = self._get_state(knee_vert_angle)

        # ── Step 6: Draw skeleton & angle annotations ──
        self._draw_annotations(frame, shldr, hip, knee, ankle,
                               hip_vert_angle, knee_vert_angle, ankle_vert_angle,
                               hip_ref, knee_ref, ankle_ref)

        # ── Step 7: Update state sequence ──
        if self.state_tracker['curr_state'] is not None:
            self._update_state_sequence(self.state_tracker['curr_state'])

        # ── Step 8: Feedback logic ──
        if self.state_tracker['curr_state'] in ['s2', 's3']:
            if hip_vert_angle > self.thresholds['HIP_THRESH'][1]:
                self.state_tracker['DISPLAY_TEXT'][0] = True
            elif hip_vert_angle < self.thresholds['HIP_THRESH'][0]:
                self.state_tracker['DISPLAY_TEXT'][1] = True

            if ankle_vert_angle > self.thresholds['ANKLE_THRESH']:
                self.state_tracker['DISPLAY_TEXT'][2] = True
                self.state_tracker['INCORRECT_POSTURE'] = True

            if knee_vert_angle > self.thresholds['KNEE_THRESH'][2]:
                self.state_tracker['DISPLAY_TEXT'][3] = True
                self.state_tracker['INCORRECT_POSTURE'] = True

            if (self.thresholds['KNEE_THRESH'][0]
                    < knee_vert_angle < self.thresholds['KNEE_THRESH'][1]):
                self.state_tracker['LOWER_HIPS'] = True
            else:
                self.state_tracker['LOWER_HIPS'] = False
        else:
            self.state_tracker['LOWER_HIPS'] = False

        # ── Step 9: Counting ──
        if (self.state_tracker['curr_state'] == 's1'
                and self.state_tracker['prev_state'] == 's2'):
            if (self.state_tracker['state_seq'] == ['s2', 's3', 's2']
                    and not self.state_tracker['INCORRECT_POSTURE']):
                self.state_tracker['SQUAT_COUNT'] += 1
            elif len(self.state_tracker['state_seq']) > 0:
                self.state_tracker['IMPROPER_SQUAT'] += 1

            self.state_tracker['state_seq'] = []
            self.state_tracker['INCORRECT_POSTURE'] = False

        if self.state_tracker['curr_state'] is not None:
            self.state_tracker['prev_state'] = self.state_tracker['curr_state']

        # ── Step 10: Advance feedback frame counters ──
        for i in range(4):
            if self.state_tracker['DISPLAY_TEXT'][i]:
                self.state_tracker['COUNT_FRAMES'][i] += 1
                if self.state_tracker['COUNT_FRAMES'][i] >= self.thresholds['CNT_FRAME_THRESH']:
                    self.state_tracker['DISPLAY_TEXT'][i] = False
                    self.state_tracker['COUNT_FRAMES'][i] = 0

        # ── Step 11: Draw counters and feedback on top ──
        self._draw_counters(frame)
        self._draw_feedback(frame, frame_width, frame_height)

        return frame
