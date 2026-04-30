"""
Feature Extraction for Hybrid GCN V2 / V5 / V6
Extracts node features and hybrid features from pose keypoints

V6 additions:
- Signed direction features (15) + has_stick binary = 49 hybrid features
- 7-dim node features with has_stick binary
- node_mask for masked pooling
- True zero-stick fallback (no NaN → origin hack)

V2 compatibility preserved for left/right models.
"""

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# V6 direction features: pass-through normalized signed values (not Gaussian similarity)
DIRECTION_FEATURES = {
    'stick_tip_signed_x': 0.5,
    'grip_signed_x': 0.5,
    'wrist_spread': 0.5,
    'stick_reach': 0.5,
    'tip_height_vs_grip': 0.5,
    'stick_forearm_dot': 1.0,
    'tip_vs_nose_signed': 0.5,
    'tip_vs_shoulder_signed': 0.5,
    'left_elbow_angle_signed': 1.0,
    'right_elbow_angle_signed': 1.0,
    'stick_angle_signed': 1.0,
    'right_wrist_height_signed': 0.5,
    'left_wrist_height_signed': 0.5,
    'left_wrist_x_signed': 0.5,
    'right_wrist_x_signed': 0.5,
    'has_stick': 1.0,  # binary hybrid feature
}

# V5 direction features: 13 signed (no has_stick, no left/right_wrist_x_signed)
DIRECTION_FEATURES_V5 = {
    'stick_tip_signed_x': 0.5,
    'grip_signed_x': 0.5,
    'wrist_spread': 0.5,
    'stick_reach': 0.5,
    'tip_height_vs_grip': 0.5,
    'stick_forearm_dot': 1.0,
    'tip_vs_nose_signed': 0.5,
    'tip_vs_shoulder_signed': 0.5,
    'left_elbow_angle_signed': 1.0,
    'right_elbow_angle_signed': 1.0,
    'stick_angle_signed': 1.0,
    'right_wrist_height_signed': 0.5,
    'left_wrist_height_signed': 0.5,
}

# Exact 46-feature order expected by v5 model (33 Gaussian + 13 signed)
V5_ALL_FEATURES = (
    'left_elbow_angle', 'right_elbow_angle', 'left_shoulder_angle', 'right_shoulder_angle',
    'left_knee_angle', 'right_knee_angle',
    'left_wrist_height', 'right_wrist_height', 'left_elbow_height', 'right_elbow_height',
    'stick_tip_height', 'stick_grip_height',
    'left_wrist_x', 'right_wrist_x', 'stick_tip_x', 'stick_grip_x',
    'stick_angle', 'stick_dx', 'stick_dy',
    'tip_vs_nose', 'tip_vs_shoulder', 'tip_vs_hip',
    'r_hand_vs_nose', 'r_hand_vs_shoulder', 'r_hand_vs_hip',
    'tip_side', 'grip_side', 'foot_stagger',
    'hands_distance', 'stick_length',
    'stick_grip_to_r_wrist', 'stick_right_of_center', 'r_wrist_vs_l_wrist_x',
    'stick_tip_signed_x', 'grip_signed_x', 'wrist_spread', 'stick_reach',
    'tip_height_vs_grip', 'stick_forearm_dot', 'tip_vs_nose_signed',
    'tip_vs_shoulder_signed', 'left_elbow_angle_signed', 'right_elbow_angle_signed',
    'stick_angle_signed', 'right_wrist_height_signed', 'left_wrist_height_signed',
)


def calculate_angle(p1, p2, p3):
    """Calculate 3D angle at p2 formed by p1-p2-p3 using three-dimensional coordinates"""
    # Handle 2D input for backward compatibility
    if len(p1) == 2:
        p1 = [p1[0], p1[1], 0.0]
    if len(p2) == 2:
        p2 = [p2[0], p2[1], 0.0]
    if len(p3) == 2:
        p3 = [p3[0], p3[1], 0.0]
    
    # 3D vector construction
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]])
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)


def calculate_distance(p1, p2):
    """Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def extract_raw_features(image, stick_detector):
    """
    Extract raw features from a single image.
    
    Args:
        image: numpy array (BGR format) or path to image
        stick_detector: YOLO model for stick detection
    
    Returns:
        dict with:
            - pose_keypoints: [33, 4] array (x, y, z, visibility)
            - stick_keypoints: [2, 4] array (grip and tip)
            - global_features: dict of computed geometric features
    """
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image
        
    if img is None:
        return None
    
    h, w = img.shape[:2]
    
    # Extract body pose with MediaPipe
    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(static_image_mode=True, model_complexity=2)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(img_rgb)
    pose_detector.close()
    
    if not results.pose_landmarks:
        return None
    
    # Get keypoints (normalized 3D world coordinates)
    kpts = []
    for lm in results.pose_landmarks.landmark:
        kpts.append([lm.x, lm.y, lm.z, lm.visibility])
    kpts = np.array(kpts)
    
    # Extract stick with YOLO
    stick_results = stick_detector(img, verbose=False)[0]
    if stick_results.keypoints is not None and len(stick_results.keypoints.data) > 0:
        stick_kpts = stick_results.keypoints.data[0].cpu().numpy()
        # Normalize and add z=0 for stick (YOLO doesn't provide depth)
        stick_grip = [stick_kpts[0, 0] / w, stick_kpts[0, 1] / h, 0.0, stick_kpts[0, 2]]
        stick_tip = [stick_kpts[1, 0] / w, stick_kpts[1, 1] / h, 0.0, stick_kpts[1, 2]]
    else:
        # FIX #2: NaN sentinel when stick not detected
        stick_grip = [np.nan, np.nan, 0.0, 0.0]
        stick_tip = [np.nan, np.nan, 0.0, 0.0]
    
    stick_keypoints = np.array([stick_grip, stick_tip])
    
    # Compute global geometric features
    features = compute_global_features_from_kpts(kpts, stick_keypoints)
    
    return {
        'pose_keypoints': kpts,
        'stick_keypoints': stick_keypoints,
        'global_features': features,
        'has_stick_detected': bool(stick_results.keypoints is not None and len(stick_results.keypoints.data) > 0)
    }


def compute_global_features_from_kpts(kpts, stick_keypoints, world_landmarks=None, has_stick_detected=None, version='v2'):
    """
    Compute global geometric features from pose and stick keypoints.
    
    Args:
        kpts: [33, 4] pose keypoints (normalized image coordinates)
        stick_keypoints: [2, 4] stick keypoints (grip, tip)
        world_landmarks: Optional MediaPipe world landmarks for 3D angles (v5/v6)
        has_stick_detected: Whether YOLO actually detected the stick (for has_stick feature)
        version: 'v2', 'v5', or 'v6' — controls fallback behavior and angle source
        
    Returns:
        dict of features
    """
    stick_grip = stick_keypoints[0]
    stick_tip = stick_keypoints[1]
    
    # V5/V6: sanitize NaN to origin/zero fallback so all features are computed normally.
    # The models learned these constant offsets during training.
    if version in ('v5', 'v6'):
        if np.isnan(stick_grip).any():
            stick_grip = np.array([0.0, 0.0, 0.0, 0.0])
        if np.isnan(stick_tip).any():
            stick_tip = np.array([0.0, 0.0, 0.0, 0.0])
        stick_available = True
    else:
        # V2: NaN sentinel = unavailable, zero out stick features
        stick_available = not (np.isnan(stick_grip[0]) or np.isnan(stick_tip[0]))
    
    features = {}
    
    # Point getter for angles: use world landmarks for v5/v6 if available
    if version in ('v5', 'v6') and world_landmarks is not None:
        def get_point(idx):
            lm = world_landmarks[idx]
            return [lm.x, lm.y, lm.z]
    else:
        def get_point(idx):
            return kpts[idx]
    
    # Joint angles
    features['left_elbow_angle'] = calculate_angle(get_point(11), get_point(13), get_point(15))
    features['right_elbow_angle'] = calculate_angle(get_point(12), get_point(14), get_point(16))
    features['left_shoulder_angle'] = calculate_angle(get_point(13), get_point(11), get_point(23))
    features['right_shoulder_angle'] = calculate_angle(get_point(14), get_point(12), get_point(24))
    features['left_knee_angle'] = calculate_angle(get_point(23), get_point(25), get_point(27))
    features['right_knee_angle'] = calculate_angle(get_point(24), get_point(26), get_point(28))
    
    # Heights relative to hip center
    hip_center_y = (kpts[23][1] + kpts[24][1]) / 2
    features['left_wrist_height'] = hip_center_y - kpts[15][1]
    features['right_wrist_height'] = hip_center_y - kpts[16][1]
    features['left_elbow_height'] = hip_center_y - kpts[13][1]
    features['right_elbow_height'] = hip_center_y - kpts[14][1]
    
    # Horizontal positions relative to hip center
    hip_center_x = (kpts[23][0] + kpts[24][0]) / 2
    features['left_wrist_x'] = kpts[15][0] - hip_center_x
    features['right_wrist_x'] = kpts[16][0] - hip_center_x
    
    # Stick-dependent features — computed normally for v5/v6 (model learned offsets),
    # zeroed out for v2 when stick is unavailable.
    if stick_available:
        features['stick_tip_height'] = hip_center_y - stick_tip[1]
        features['stick_grip_height'] = hip_center_y - stick_grip[1]
        features['stick_tip_x'] = stick_tip[0] - hip_center_x
        features['stick_grip_x'] = stick_grip[0] - hip_center_x
        
        # Stick orientation
        stick_vector = np.array([stick_tip[0] - stick_grip[0], stick_tip[1] - stick_grip[1]])
        features['stick_angle'] = np.degrees(np.arctan2(stick_vector[1], stick_vector[0]))
        
        stick_len = np.linalg.norm(stick_vector) + 1e-6
        features['stick_dx'] = stick_vector[0] / stick_len
        features['stick_dy'] = stick_vector[1] / stick_len
    else:
        features['stick_tip_height'] = 0.0
        features['stick_grip_height'] = 0.0
        features['stick_tip_x'] = 0.0
        features['stick_grip_x'] = 0.0
        features['stick_angle'] = 0.0
        features['stick_dx'] = 0.0
        features['stick_dy'] = 0.0
    
    # Expert features (relative to body landmarks)
    root_x = (kpts[23][0] + kpts[24][0]) / 2
    root_y = (kpts[23][1] + kpts[24][1]) / 2
    shoulder_y = (kpts[11][1] + kpts[12][1]) / 2
    nose_y = kpts[0][1]
    
    if stick_available:
        features['tip_vs_nose'] = stick_tip[1] - nose_y
        features['tip_vs_shoulder'] = stick_tip[1] - shoulder_y
        features['tip_vs_hip'] = stick_tip[1] - root_y
    else:
        features['tip_vs_nose'] = 0.0
        features['tip_vs_shoulder'] = 0.0
        features['tip_vs_hip'] = 0.0
    
    features['r_hand_vs_nose'] = kpts[16][1] - nose_y
    features['r_hand_vs_shoulder'] = kpts[16][1] - shoulder_y
    features['r_hand_vs_hip'] = kpts[16][1] - root_y
    
    if stick_available:
        features['tip_side'] = stick_tip[0] - root_x
        features['grip_side'] = stick_grip[0] - root_x
    else:
        features['tip_side'] = 0.0
        features['grip_side'] = 0.0
    
    features['foot_stagger'] = kpts[27][1] - kpts[28][1]
    
    # Distances
    features['hands_distance'] = calculate_distance(kpts[15], kpts[16])
    if stick_available:
        features['stick_length'] = calculate_distance(stick_grip, stick_tip)
    else:
        features['stick_length'] = 0.0

    # === V6 additional base features ===
    features['stick_grip_to_r_wrist'] = calculate_distance(stick_grip, kpts[16]) if stick_available else 0.0
    features['stick_right_of_center'] = (stick_tip[0] - root_x) if stick_available else 0.0
    features['r_wrist_vs_l_wrist_x'] = kpts[16][0] - kpts[15][0]

    # === SIGNED DIRECTION FEATURES ===
    shoulder_width = np.linalg.norm(kpts[11, :2] - kpts[12, :2]) + 1e-8

    features['stick_tip_signed_x'] = (stick_tip[0] - hip_center_x) / shoulder_width if stick_available else 0.0
    features['grip_signed_x'] = (stick_grip[0] - hip_center_x) / shoulder_width if stick_available else 0.0
    features['wrist_spread'] = (kpts[15][0] - kpts[16][0]) / shoulder_width
    features['stick_reach'] = abs(stick_tip[0] - stick_grip[0]) / shoulder_width if stick_available else 0.0
    features['tip_height_vs_grip'] = (stick_tip[1] - stick_grip[1]) / shoulder_width if stick_available else 0.0

    # stick_forearm_dot
    grip_px = np.array([stick_grip[0], stick_grip[1]]) if stick_available else np.array([0.0, 0.0])
    dist_to_rwrist = np.linalg.norm(grip_px - kpts[16, :2]) if stick_available else float('inf')
    dist_to_lwrist = np.linalg.norm(grip_px - kpts[15, :2]) if stick_available else float('inf')
    if stick_available:
        if dist_to_rwrist < dist_to_lwrist:
            forearm_vec = kpts[16, :2] - kpts[14, :2]
        else:
            forearm_vec = kpts[15, :2] - kpts[13, :2]
        stick_vec_2d = np.array([stick_tip[0] - stick_grip[0], stick_tip[1] - stick_grip[1]])
        forearm_len = np.linalg.norm(forearm_vec) + 1e-8
        stick_len_2d = np.linalg.norm(stick_vec_2d) + 1e-8
        if forearm_len > 0.01 and stick_len_2d > 0.01:
            dot = np.dot(forearm_vec / forearm_len, stick_vec_2d / stick_len_2d)
            features['stick_forearm_dot'] = dot
        else:
            features['stick_forearm_dot'] = 0.5
    else:
        features['stick_forearm_dot'] = 0.5

    features['tip_vs_nose_signed'] = (stick_tip[1] - nose_y) / shoulder_width if stick_available else 0.0
    features['tip_vs_shoulder_signed'] = (stick_tip[1] - shoulder_y) / shoulder_width if stick_available else 0.0
    features['left_elbow_angle_signed'] = features['left_elbow_angle'] / 180.0
    features['right_elbow_angle_signed'] = features['right_elbow_angle'] / 180.0
    features['stick_angle_signed'] = features['stick_angle'] / 180.0
    features['right_wrist_height_signed'] = features['right_wrist_height'] / shoulder_width
    features['left_wrist_height_signed'] = features['left_wrist_height'] / shoulder_width
    features['left_wrist_x_signed'] = features['left_wrist_x'] / shoulder_width
    features['right_wrist_x_signed'] = features['right_wrist_x'] / shoulder_width

    # has_stick binary hybrid feature (v6 only; v5 compute_hybrid_features_v5 ignores it)
    if has_stick_detected is not None:
        features['has_stick'] = 1.0 if has_stick_detected else 0.0
    else:
        features['has_stick'] = 1.0 if stick_available else 0.0

    return features


def gaussian_similarity(value, mean, std):
    """Compute similarity score using Gaussian distribution"""
    if std < 1e-6:
        std = 1e-6
    return np.exp(-0.5 * ((value - mean) / std) ** 2)


def compute_hybrid_features(raw_features, templates, viewpoint, class_name):
    """
    Convert raw geometric features to similarity scores.
    V2-compatible: iterates template keys so output length matches the model.
    
    Args:
        raw_features: dict of geometric feature values
        templates: dict loaded from feature_templates.json
        viewpoint: 'front', 'left', or 'right'
        class_name: target class name
    
    Returns:
        numpy array of similarity scores (matches template key count)
    """
    key = f"{viewpoint}_{class_name}"
    
    if key not in templates:
        return np.zeros(len(raw_features), dtype=np.float32)
    
    template = templates[key]
    hybrid_features = []
    
    # FIX: iterate template keys (not raw_features) so vector length always
    # matches the model's expected hybrid_in_channels, regardless of extra
    # keys added for v6.
    for feat_name in template:
        if feat_name in raw_features:
            feat_value = raw_features[feat_name]
            mean = template[feat_name]['mean']
            std = template[feat_name]['std']
            similarity = gaussian_similarity(feat_value, mean, std)
            hybrid_features.append(similarity)
        else:
            hybrid_features.append(0.0)
    
    return np.array(hybrid_features, dtype=np.float32)


def compute_hybrid_features_v6(raw_features, templates, viewpoint, class_name):
    """
    V6 hybrid feature vector: 49-dim = 33 Gaussian + 15 signed + 1 has_stick.
    Signed direction features are pass-through (normalized), not Gaussian.
    Matches training pipeline exactly.
    
    Args:
        raw_features: dict of geometric feature values (must include all 49 keys)
        templates: dict loaded from feature_templates.json
        viewpoint: 'front', 'left', or 'right'
        class_name: target class name
    
    Returns:
        numpy array of 49 hybrid features
    """
    key = f"{viewpoint}_{class_name}"
    
    if key not in templates:
        return np.zeros(49, dtype=np.float32)
    
    template = templates[key]
    hybrid_features = []
    
    for feat_name, feat_value in raw_features.items():
        if feat_name in DIRECTION_FEATURES:
            normalized = feat_value / DIRECTION_FEATURES[feat_name]
            normalized = np.clip(normalized, -3.0, 3.0)
            hybrid_features.append(normalized)
        elif feat_name in template:
            mean = template[feat_name]['mean']
            std = template[feat_name]['std']
            similarity = gaussian_similarity(feat_value, mean, std)
            hybrid_features.append(similarity)
        else:
            hybrid_features.append(0.0)
    
    return np.array(hybrid_features, dtype=np.float32)


def compute_hybrid_features_v5(raw_features, templates, viewpoint, class_name):
    """
    V5 hybrid feature vector: 46-dim = 33 Gaussian + 13 signed direction.
    Iterates V5_ALL_FEATURES in fixed order so model input is deterministic.
    Signed features: pass-through normalized (not Gaussian).
    Gaussian features: similarity against template mean/std.
    Missing template entry: 0.0 fallback.
    """
    key = f"{viewpoint}_{class_name}"
    template = templates.get(key, {})
    hybrid_features = []

    for feat_name in V5_ALL_FEATURES:
        feat_value = raw_features.get(feat_name, 0.0)

        if feat_name in DIRECTION_FEATURES_V5:
            normalized = feat_value / DIRECTION_FEATURES_V5[feat_name]
            normalized = np.clip(normalized, -3.0, 3.0)
            hybrid_features.append(normalized)
        elif feat_name in template and isinstance(template[feat_name], dict):
            mean = template[feat_name]['mean']
            std = template[feat_name]['std']
            similarity = gaussian_similarity(feat_value, mean, std)
            hybrid_features.append(similarity)
        else:
            hybrid_features.append(0.0)

    return np.array(hybrid_features, dtype=np.float32)


def extract_node_features(pose_keypoints, stick_keypoints):
    """
    Extract per-node features with 3D coordinates.
    Each node gets: [x, y, z, visibility, distance_to_hip_3d, angle_from_hip]
    
    Args:
        pose_keypoints: [33, 4] array from MediaPipe
        stick_keypoints: [2, 4] array from YOLO
    
    Returns:
        [35, 6] array of node features
    """
    # FIX #2: Replace NaN stick keypoints with zeros so GCN receives
    # valid numeric input.  Zero-valued nodes at the hip center produce
    # zero dist/angle, contributing no discriminative signal.
    clean_stick = stick_keypoints.copy()
    if np.isnan(clean_stick).any():
        clean_stick = np.zeros_like(clean_stick)
    
    # Combine all nodes (33 pose + 2 stick)
    all_keypoints = np.vstack([pose_keypoints, clean_stick])
    
    # Compute hip center for reference (3D)
    hip_center = (pose_keypoints[23, :3] + pose_keypoints[24, :3]) / 2  # [x, y, z]
    
    node_features = []
    for i, kpt in enumerate(all_keypoints):
        x, y, z, vis = kpt
        
        # 3D Distance to hip center
        dist_to_hip = np.sqrt((x - hip_center[0])**2 + (y - hip_center[1])**2 + (z - hip_center[2])**2)
        
        # Angle from hip center (2D projection)
        angle_from_hip = np.degrees(np.arctan2(y - hip_center[1], x - hip_center[0]))
        
        # Node feature: [x, y, z, vis, dist_to_hip_3d, angle_from_hip]
        node_features.append([x, y, z, vis, dist_to_hip, angle_from_hip])
    
    return np.array(node_features, dtype=np.float32)


def extract_node_features_v6(pose_keypoints, stick_keypoints, has_stick_detected=True):
    """
    V6 node features: 7-dim [x, y, z, vis, dist_to_hip_3d, angle_from_hip, has_stick].
    
    - Body nodes (0-32): has_stick = 1.0
    - Stick nodes (33-34): has_stick = 1.0 if detected, 0.0 if zero-stick fallback
    - True zero for invisible/missing nodes (vis < 1e-6): all 7 dims = 0.0
    
    Args:
        pose_keypoints: [33, 4] array from MediaPipe
        stick_keypoints: [2, 4] array (true zeros when not detected)
        has_stick_detected: bool, whether YOLO actually detected the stick
    
    Returns:
        [35, 7] array of node features
    """
    all_keypoints = np.vstack([pose_keypoints, stick_keypoints])
    hip_center = (pose_keypoints[23, :3] + pose_keypoints[24, :3]) / 2
    
    node_features = []
    for i, kpt in enumerate(all_keypoints):
        x, y, z, vis = kpt
        is_stick_node = i >= 33
        
        if vis < 1e-6:
            node_features.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            dist_to_hip = np.sqrt((x - hip_center[0])**2 + (y - hip_center[1])**2 + (z - hip_center[2])**2)
            angle_from_hip = np.degrees(np.arctan2(y - hip_center[1], x - hip_center[0]))
            
            if is_stick_node and not has_stick_detected:
                has_stick = 0.0
            else:
                has_stick = 1.0
            
            node_features.append([x, y, z, vis, dist_to_hip, angle_from_hip, has_stick])
    
    return np.array(node_features, dtype=np.float32)


def create_node_mask(has_stick_detected):
    """
    Create node-level mask for v6 masked pooling.
    
    Args:
        has_stick_detected: bool, whether YOLO detected the stick
    
    Returns:
        [35] array: 1.0 for body nodes (0-32), 1.0 for stick nodes (33-34) if detected, else 0.0
    """
    mask = np.ones(35, dtype=np.float32)
    if not has_stick_detected:
        mask[33:] = 0.0  # zero out stick nodes
    return mask
