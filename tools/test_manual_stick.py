"""
Test manual stick patterns on a single image.
Displays the image with:
- Skeleton (green)
- Manually-defined stick based on pose (yellow)
- Detected pose class
"""

import cv2
import mediapipe as mp
import numpy as np
import sys
import os

# Add parent directory to path to import manual patterns
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from manual_stick_patterns import get_stick_pattern

# Import keras model for pose prediction
from tensorflow import keras
import joblib


def calculate_angle(point1, point2, point3):
    """Calculate angle at point2 formed by point1-point2-point3"""
    vector1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
    vector2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
    
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    
    cos_angle = dot_product / (magnitude1 * magnitude2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def extract_features(world_landmarks):
    """Extract 3D coordinates from world landmarks for pose classification (same as pose_analyzer.py)"""
    # Convert world landmarks to numpy array
    landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks])
    
    # Normalize by hip center (same as in pose_analyzer.py)
    hip_center = (landmarks_np[23] + landmarks_np[24]) / 2.0
    coords = (landmarks_np - hip_center).flatten()
    
    return coords


def draw_manual_stick(image, landmarks, pose_class, image_width, image_height):
    """Draw stick based on manual pattern definition"""
    
    # Get manual stick pattern for this pose
    pattern = get_stick_pattern(pose_class)
    
    if pattern is None:
        cv2.putText(image, "No manual pattern defined for this pose", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return image
    
    # Get hand landmarks
    hand = pattern['hand']
    if hand == 'right':
        wrist_idx = mp.solutions.pose.PoseLandmark.RIGHT_WRIST
        elbow_idx = mp.solutions.pose.PoseLandmark.RIGHT_ELBOW
    else:
        wrist_idx = mp.solutions.pose.PoseLandmark.LEFT_WRIST
        elbow_idx = mp.solutions.pose.PoseLandmark.LEFT_ELBOW
    
    wrist = landmarks[wrist_idx.value]
    elbow = landmarks[elbow_idx.value]
    
    wrist_point = np.array([wrist.x * image_width, wrist.y * image_height])
    elbow_point = np.array([elbow.x * image_width, elbow.y * image_height])
    
    # Calculate forearm direction (from elbow to wrist)
    arm_vector = wrist_point - elbow_point
    arm_angle_rad = np.arctan2(arm_vector[1], arm_vector[0])
    
    # Add the manual offset angle
    stick_offset_rad = np.radians(pattern['stick_arm_angle_degrees'])
    stick_angle_rad = arm_angle_rad + stick_offset_rad
    
    # Calculate stick length based on body proportion
    left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_knee = landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value]
    
    shoulder_mid = np.array([
        (left_shoulder.x + right_shoulder.x) / 2 * image_width,
        (left_shoulder.y + right_shoulder.y) / 2 * image_height
    ])
    knee_point = np.array([left_knee.x * image_width, left_knee.y * image_height])
    body_height = np.linalg.norm(shoulder_mid - knee_point)
    
    stick_length = body_height * pattern['stick_length_ratio']
    
    # Calculate stick endpoint
    stick_end = wrist_point + stick_length * np.array([
        np.cos(stick_angle_rad),
        np.sin(stick_angle_rad)
    ])
    
    # Draw the stick
    cv2.line(image, 
             tuple(wrist_point.astype(int)), 
             tuple(stick_end.astype(int)), 
             (0, 255, 255), 3)  # Yellow
    
    # Draw wrist point
    cv2.circle(image, tuple(wrist_point.astype(int)), 5, (255, 0, 0), -1)  # Blue
    
    # Add text info
    cv2.putText(image, f"Pose: {pose_class}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, f"Hand: {hand} | Angle offset: {pattern['stick_arm_angle_degrees']}", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return image


def test_image(image_path):
    """Test manual stick pattern on a single image"""
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    image_height, image_width = image.shape[:2]
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)
    
    # Process image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if not results.pose_landmarks or not results.pose_world_landmarks:
        print("No pose detected in image")
        return
    
    # Draw skeleton
    mp.solutions.drawing_utils.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
        mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)
    )
    
    # Predict pose class using the trained model
    try:
        model = keras.models.load_model('models/arnis_coordinates_classifier.keras')
        label_encoder = joblib.load('models/label_encoder.joblib')
        
        # Extract 3D world landmarks features (99 features: 33 landmarks × 3 coords)
        features = extract_features(results.pose_world_landmarks.landmark)
        features_array = np.expand_dims(features, axis=0)
        
        predictions = model.predict(features_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
        
        print(f"Detected pose: {predicted_class}")
        
    except Exception as e:
        print(f"Could not load model for prediction: {e}")
        print("Using filename to guess pose class...")
        # Try to extract pose from filename
        import re
        filename = os.path.basename(image_path)
        predicted_class = filename.split('_')[0]  # Rough guess
    
    # Draw manual stick
    image = draw_manual_stick(image, results.pose_landmarks.landmark, 
                             predicted_class, image_width, image_height)
    
    # Save and display
    output_path = image_path.replace('.jpg', '_manual_stick_test.jpg').replace('.png', '_manual_stick_test.png')
    cv2.imwrite(output_path, image)
    print(f"Saved result to: {output_path}")
    
    # Display
    cv2.imshow('Manual Stick Pattern Test', image)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/test_manual_stick.py <path_to_image>")
        print("Example: python tools/test_manual_stick.py test_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    test_image(image_path)
