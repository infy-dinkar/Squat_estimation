import cv2
import numpy as np
import mediapipe as mp

def find_angle(p1, p2, ref_pt):
    """
    Computes angle between p1-ref_pt and p2-ref_pt.
    """
    v1 = np.array(p1) - np.array(ref_pt)
    v2 = np.array(p2) - np.array(ref_pt)
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def get_landmark_array(pose_landmark, image_shape):
    """
    Convert normalized MediaPipe point to pixel coordinates.
    """
    h, w, _ = image_shape
    return int(pose_landmark.x * w), int(pose_landmark.y * h)

def get_landmark_features(kp, image_shape):
    """
    Extract useful joints from MediaPipe prediction points.
    Returns:
    - nose coordinates
    - left side joint tuple (shoulder, elbow, wrist, hip, knee, ankle, foot)
    - right side joint tuple (shoulder, elbow, wrist, hip, knee, ankle, foot)
    """
    nose = get_landmark_array(kp.landmark[0], image_shape)
    
    left_joints = (
        get_landmark_array(kp.landmark[11], image_shape),
        get_landmark_array(kp.landmark[13], image_shape),
        get_landmark_array(kp.landmark[15], image_shape),
        get_landmark_array(kp.landmark[23], image_shape),
        get_landmark_array(kp.landmark[25], image_shape),
        get_landmark_array(kp.landmark[27], image_shape),
        get_landmark_array(kp.landmark[31], image_shape)
    )
    
    right_joints = (
        get_landmark_array(kp.landmark[12], image_shape),
        get_landmark_array(kp.landmark[14], image_shape),
        get_landmark_array(kp.landmark[16], image_shape),
        get_landmark_array(kp.landmark[24], image_shape),
        get_landmark_array(kp.landmark[26], image_shape),
        get_landmark_array(kp.landmark[28], image_shape),
        get_landmark_array(kp.landmark[32], image_shape)
    )
    return nose, left_joints, right_joints

def get_mediapipe_pose():
    """
    Create and return a configured MediaPipe Pose object.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return pose

def draw_text(image, text, pos, text_color=(255, 255, 255), text_color_bg=(0, 0, 0), font_scale=1):
    """
    Draw rounded-rectangle backed text on frame.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 2
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(image, (x, y - text_h - 10), (x + text_w + 10, y + 5), text_color_bg, -1)
    cv2.putText(image, text, (x + 5, y - 5), font, font_scale, text_color, font_thickness)

def draw_dotted_line(image, pt1, pt2, color, thickness=1, gap=5):
    """
    Draw a dotted vertical reference line (or any line).
    """
    dist = ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**0.5
    if dist == 0:
        return
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + 0.5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + 0.5)
        pts.append((x, y))
    
    for i in range(0, len(pts), 2):
        if i + 1 < len(pts):
            cv2.line(image, pts[i], pts[i+1], color, thickness)
