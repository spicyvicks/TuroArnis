import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def calculate_angle_3d(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    dot_product = np.dot(ba, bc)
    magnitude = np.linalg.norm(ba) * np.linalg.norm(bc)
    cosine_angle = np.clip(dot_product / (magnitude + 1e-6), -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def extract_features_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None: return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_world_landmarks:
        try:
            landmarks = results.pose_world_landmarks.landmark
            coords = {lm.name.lower(): [lm.x, lm.y, lm.z] for lm in mp_pose.PoseLandmark}
            
            angles = {
                'left_elbow': calculate_angle_3d(coords['left_shoulder'], coords['left_elbow'], coords['left_wrist']),
                'left_shoulder': calculate_angle_3d(coords['left_hip'], coords['left_shoulder'], coords['left_elbow']),
                'left_hip': calculate_angle_3d(coords['left_shoulder'], coords['left_hip'], coords['left_knee']),
                'left_knee': calculate_angle_3d(coords['left_hip'], coords['left_knee'], coords['left_ankle']),
                'right_elbow': calculate_angle_3d(coords['right_shoulder'], coords['right_elbow'], coords['right_wrist']),
                'right_shoulder': calculate_angle_3d(coords['right_hip'], coords['right_shoulder'], coords['right_elbow']),
                'right_hip': calculate_angle_3d(coords['right_shoulder'], coords['right_hip'], coords['right_knee']),
                'right_knee': calculate_angle_3d(coords['right_hip'], coords['right_knee'], coords['right_ankle']),
            }
            return angles
        except Exception:
            return None
    return None

if __name__ == "__main__":
    dataset_folder = 'dataset_multiclass_2'
    csv_output_file = 'pose_detection.csv'
    
    print("[info] starting feature extraction...")
    
    pose_classes = [d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))]
    
    header = ['class'] + [
        'left_elbow', 'left_shoulder', 'left_hip', 'left_knee',
        'right_elbow', 'right_shoulder', 'right_hip', 'right_knee'
    ]

    with open(csv_output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for class_name in pose_classes:
            print(f"processing class: {class_name}")
            class_folder_path = os.path.join(dataset_folder, class_name)
            for filename in os.listdir(class_folder_path):
                image_path = os.path.join(class_folder_path, filename)
                angles = extract_features_from_image(image_path)
                
                if angles:
                    row = [class_name] + [angles.get(joint, 0) for joint in header[1:]]
                    writer.writerow(row)

    print(f"[info] feature extraction complete. data saved to {csv_output_file}")

    print("\n[info] starting random forest model training...")
    
    data = pd.read_csv(csv_output_file)
    X = data.drop('class', axis=1)
    y = data['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    print("[info] training the model...")
    model.fit(X_train, y_train)

    print("[info] evaluating the model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nmodel accuracy: {accuracy * 100:.2f}%")
    print("\nclassification report:")
    print(classification_report(y_test, y_pred))

    model_filename = 'models/arnis_random_forest_classifier_2.joblib'
    class_names_filename = 'models/arnis_class_names_2.joblib'
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, model_filename)
    joblib.dump(model.classes_, class_names_filename)

    print(f"\n[info] training complete. model saved to {model_filename}")

    pose.close()