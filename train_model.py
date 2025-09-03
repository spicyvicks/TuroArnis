import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import gaussiannb
from sklearn.metrics import accuracy_score, classification_report
import joblib

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def calculate_angle(a, b, c):
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c) 
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle 

print("[info] starting feature extraction from multi-class image dataset...")

dataset_folder = 'dataset_multiclass'
csv_output_file = 'arnis_multiclass_poses.csv'
header = [
    'class', 'left_elbow', 'left_shoulder', 'left_hip', 'left_knee', 'left_ankle',
    'right_elbow', 'right_shoulder', 'right_hip', 'right_knee', 'right_ankle'
]

pose_classes = [d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))]
print(f"[info] found pose classes: {pose_classes}")

with open(csv_output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header) 

    for class_name in pose_classes:
        class_folder_path = os.path.join(dataset_folder, class_name)
        
        for filename in os.listdir(class_folder_path):
            image_path = os.path.join(class_folder_path, filename)
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"warning: could not read image {image_path}. skipping.")
                continue

            image_rgb = cv2.cvtColor(image, cv2.color_bgr2rgb)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                try:
                    landmarks = results.pose_landmarks.landmark
                    h, w, _ = image.shape

                    l_shoulder = [landmarks[mp_pose.poselandmark.left_shoulder.value].x * w, landmarks[mp_pose.poselandmark.left_shoulder.value].y * h]
                    l_elbow = [landmarks[mp_pose.poselandmark.left_elbow.value].x * w, landmarks[mp_pose.poselandmark.left_elbow.value].y * h]
                    l_wrist = [landmarks[mp_pose.poselandmark.left_wrist.value].x * w, landmarks[mp_pose.poselandmark.left_wrist.value].y * h]
                    l_hip = [landmarks[mp_pose.poselandmark.left_hip.value].x * w, landmarks[mp_pose.poselandmark.left_hip.value].y * h]
                    l_knee = [landmarks[mp_pose.poselandmark.left_knee.value].x * w, landmarks[mp_pose.poselandmark.left_knee.value].y * h]
                    l_ankle = [landmarks[mp_pose.poselandmark.left_ankle.value].x * w, landmarks[mp_pose.poselandmark.left_ankle.value].y * h]
                    l_foot_index = [landmarks[mp_pose.poselandmark.left_foot_index.value].x * w, landmarks[mp_pose.poselandmark.left_foot_index.value].y * h]
                    r_shoulder = [landmarks[mp_pose.poselandmark.right_shoulder.value].x * w, landmarks[mp_pose.poselandmark.right_shoulder.value].y * h]
                    r_elbow = [landmarks[mp_pose.poselandmark.right_elbow.value].x * w, landmarks[mp_pose.poselandmark.right_elbow.value].y * h]
                    r_wrist = [landmarks[mp_pose.poselandmark.right_wrist.value].x * w, landmarks[mp_pose.poselandmark.right_wrist.value].y * h]
                    r_hip = [landmarks[mp_pose.poselandmark.right_hip.value].x * w, landmarks[mp_pose.poselandmark.right_hip.value].y * h]
                    r_knee = [landmarks[mp_pose.poselandmark.right_knee.value].x * w, landmarks[mp_pose.poselandmark.right_knee.value].y * h]
                    r_ankle = [landmarks[mp_pose.poselandmark.right_ankle.value].x * w, landmarks[mp_pose.poselandmark.right_ankle.value].y * h]
                    r_foot_index = [landmarks[mp_pose.poselandmark.right_foot_index.value].x * w, landmarks[mp_pose.poselandmark.right_foot_index.value].y * h]
                    
                    angles = {
                        'left_elbow': calculate_angle(l_shoulder, l_elbow, l_wrist),
                        'left_shoulder': calculate_angle(l_hip, l_shoulder, l_elbow),
                        'left_hip': calculate_angle(l_shoulder, l_hip, l_knee),
                        'left_knee': calculate_angle(l_hip, l_knee, l_ankle),
                        'left_ankle': calculate_angle(l_knee, l_ankle, l_foot_index),
                        'right_elbow': calculate_angle(r_shoulder, r_elbow, r_wrist),
                        'right_shoulder': calculate_angle(r_hip, r_shoulder, r_elbow),
                        'right_hip': calculate_angle(r_shoulder, r_hip, r_knee),
                        'right_knee': calculate_angle(r_hip, r_knee, r_ankle),
                        'right_ankle': calculate_angle(r_knee, r_ankle, r_foot_index)
                    }

                    row = [class_name] + list(angles.values())
                    writer.writerow(row)

                except Exception as e:
                    print(f"error processing {image_path}: {e}")

print(f"[info] feature extraction complete. data saved to {csv_output_file}")

print("\n[info] starting multi-class model training...")

data = pd.read_csv(csv_output_file)

x = data.drop('class', axis=1) 
y = data['class'] 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = gaussiannb()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nmodel accuracy: {accuracy * 100:.2f}%")
print("\nclassification report:")
print(classification_report(y_test, y_pred))

model_filename = 'arnis_multiclass_classifier.joblib'
joblib.dump(model, model_filename)
class_names_filename = 'arnis_class_names.joblib'
joblib.dump(model.classes_, class_names_filename)


print(f"\n[info] training complete. model saved to {model_filename} and class names to {class_names_filename}")

pose.close()