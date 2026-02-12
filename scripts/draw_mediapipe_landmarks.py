
import cv2
import mediapipe as mp
import sys
import numpy as np

def draw_landmarks_with_ids(image_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Process image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if not results.pose_landmarks:
        print("No landmarks detected.")
        return

    h, w, _ = image.shape
    
    # Draw landmarks
    print(f"Drawing landmarks on {image_path}...")
    
    # Draw connections (skeleton)
    mp.solutions.drawing_utils.draw_landmarks(
        image, 
        results.pose_landmarks, 
        mp_pose.POSE_CONNECTIONS
    )

    # Draw IDs
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        if landmark.visibility < 0.5:
            continue
            
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        
        # Draw larger circle for visibility
        cv2.circle(image, (cx, cy), 5, (0, 255, 255), -1)
        
        # Put Text ID
        cv2.putText(image, str(idx), (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        print(f"Landmark {idx}: ({cx}, {cy}) z={landmark.z:.2f}")

    output_path = "debug_landmarks.jpg"
    cv2.imwrite(output_path, image)
    print(f"Saved output to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/draw_mediapipe_landmarks.py <image_path>")
    else:
        draw_landmarks_with_ids(sys.argv[1])
