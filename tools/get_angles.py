import cv2
import mediapipe as mp
import numpy as np
import os
import json
from pathlib import Path
from itertools import chain

class GetAngles:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.7
        )
        
        self.joint_configs = {
            'left_elbow': [
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.LEFT_ELBOW,
                self.mp_pose.PoseLandmark.LEFT_WRIST
            ],
            'left_shoulder': [
                self.mp_pose.PoseLandmark.LEFT_HIP,
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.LEFT_ELBOW
            ],
            'right_elbow': [
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                self.mp_pose.PoseLandmark.RIGHT_WRIST
            ],
            'right_shoulder': [
                self.mp_pose.PoseLandmark.RIGHT_HIP,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_ELBOW
            ],
            # Add new joint configurations
            'right_hip': [
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_HIP,
                self.mp_pose.PoseLandmark.RIGHT_KNEE
            ],
            'right_knee': [
                self.mp_pose.PoseLandmark.RIGHT_HIP,
                self.mp_pose.PoseLandmark.RIGHT_KNEE,
                self.mp_pose.PoseLandmark.RIGHT_ANKLE
            ],
            'right_ankle': [
                self.mp_pose.PoseLandmark.RIGHT_KNEE,
                self.mp_pose.PoseLandmark.RIGHT_ANKLE,
                self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
            ],
            # Add left side equivalents for completeness
            'left_hip': [
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.LEFT_HIP,
                self.mp_pose.PoseLandmark.LEFT_KNEE
            ],
            'left_knee': [
                self.mp_pose.PoseLandmark.LEFT_HIP,
                self.mp_pose.PoseLandmark.LEFT_KNEE,
                self.mp_pose.PoseLandmark.LEFT_ANKLE
            ],
            'left_ankle': [
                self.mp_pose.PoseLandmark.LEFT_KNEE,
                self.mp_pose.PoseLandmark.LEFT_ANKLE,
                self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX
            ]
        }

        self.facing_directions = ['front', 'left'] #'right'
        
        # Update joint_configs_by_facing to include all joints for each facing direction
        self.joint_configs_by_facing = {
            'front': {
                'left_elbow': [
                    self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                    self.mp_pose.PoseLandmark.LEFT_ELBOW,
                    self.mp_pose.PoseLandmark.LEFT_WRIST
                ],
                'left_shoulder': [
                    self.mp_pose.PoseLandmark.LEFT_HIP,
                    self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                    self.mp_pose.PoseLandmark.LEFT_ELBOW
                ],
                'right_elbow': [
                    self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                    self.mp_pose.PoseLandmark.RIGHT_WRIST
                ],
                'right_shoulder': [
                    self.mp_pose.PoseLandmark.RIGHT_HIP,
                    self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    self.mp_pose.PoseLandmark.RIGHT_ELBOW
                ],
                # Add new joints
                'right_hip': [
                    self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    self.mp_pose.PoseLandmark.RIGHT_HIP,
                    self.mp_pose.PoseLandmark.RIGHT_KNEE
                ],
                'right_knee': [
                    self.mp_pose.PoseLandmark.RIGHT_HIP,
                    self.mp_pose.PoseLandmark.RIGHT_KNEE,
                    self.mp_pose.PoseLandmark.RIGHT_ANKLE
                ],
                'right_ankle': [
                    self.mp_pose.PoseLandmark.RIGHT_KNEE,
                    self.mp_pose.PoseLandmark.RIGHT_ANKLE,
                    self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
                ],
                'left_hip': [
                    self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                    self.mp_pose.PoseLandmark.LEFT_HIP,
                    self.mp_pose.PoseLandmark.LEFT_KNEE
                ],
                'left_knee': [
                    self.mp_pose.PoseLandmark.LEFT_HIP,
                    self.mp_pose.PoseLandmark.LEFT_KNEE,
                    self.mp_pose.PoseLandmark.LEFT_ANKLE
                ],
                'left_ankle': [
                    self.mp_pose.PoseLandmark.LEFT_KNEE,
                    self.mp_pose.PoseLandmark.LEFT_ANKLE,
                    self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX
                ]
            }
        }
        
        # Copy the same joint configurations for left and right facing
        self.joint_configs_by_facing['left'] = self.joint_configs_by_facing['front'].copy()
        self.joint_configs_by_facing['right'] = self.joint_configs_by_facing['front'].copy()

    def calculate_angle(self, a, b, c):
        a = np.array([a.x, a.y, a.z])
        b = np.array([b.x, b.y, b.z])
        c = np.array([c.x, c.y, c.z])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)

    def analyze_image(self, image_path, output_path, facing_direction='front', tolerance=10):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return None
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        if not results.pose_world_landmarks:
            print(f"No pose detected in {image_path}")
            return None
            
        joint_configs = self.joint_configs_by_facing[facing_direction]
        angles = {}
        annotated_image = image.copy()
        h, w, _ = image.shape
        
        for joint_name, landmarks in joint_configs.items():
            points = [results.pose_world_landmarks.landmark[lm] for lm in landmarks]
            angle = self.calculate_angle(*points)
            angles[joint_name] = [
                max(0, angle - tolerance),  
                min(180, angle + tolerance)  
            ]
            
            mid_point = results.pose_landmarks.landmark[landmarks[1]]
            text_x = int(mid_point.x * w)
            text_y = int(mid_point.y * h)
            
            cv2.putText(
                annotated_image,
                f"{joint_name}: {angle:.1f}°",
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        self.mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2)
        )
        
        cv2.imwrite(output_path, annotated_image)
        
        return angles

def main():
    base_dir = Path("pose_reference_images")
    raw_dir = base_dir / "raw"
    analyzed_dir = base_dir / "analyzed"
    pose_defs = {}
    
    base_dir.mkdir(exist_ok=True)
    analyzed_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)
    
    analyzer = GetAngles()
    
    image_files = chain(
        raw_dir.glob("*.jpg"),
        raw_dir.glob("*.png"),
        raw_dir.glob("*.JPG"),
        raw_dir.glob("*.PNG")
    )
    
    for image_file in image_files:
        if "analyzed" in str(image_file):
            continue
            
        pose_name = image_file.stem
        facing = 'front'  
        
        for direction in analyzer.facing_directions:
            if f"_{direction}" in pose_name.lower():  
                facing = direction
                break
                
        print(f"Processing {pose_name} ({facing} facing)...")
        
        output_path = analyzed_dir / f"{pose_name}_analyzed.jpg"
        angles = analyzer.analyze_image(
            str(image_file), 
            str(output_path),
            facing_direction=facing
        )
        
        if angles:
            pose_defs[f"{pose_name}_correct"] = angles
    
    if pose_defs:
        definitions_path = Path("pose_definitions.py")
        with open(definitions_path, "w") as f:
            f.write("# Auto-generated pose definitions\n\n")
            f.write("POSE_LIBRARY = {\n")
            
            for pose_name, angles in pose_defs.items():
                f.write(f"    '{pose_name}': {{\n")
                for joint, (min_angle, max_angle) in angles.items():
                    f.write(f"        '{joint}': [{min_angle:.1f}, {max_angle:.1f}],\n")
                f.write("    },\n")
            
            f.write("}\n")
        
        print(f"\nPose definitions saved to {definitions_path}")
        print(f"Analyzed images saved in {analyzed_dir}")

if __name__ == "__main__":
    main()