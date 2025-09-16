import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

print("[INFO] Starting feature extraction from multi-class image dataset...")

dataset_folder = 'dataset_multiclass'
csv_output_file = 'arnis_multiclass_poses_3d.csv' 

header = ['class']
for landmark in mp_pose.PoseLandmark:
    header += [f'{landmark.name.lower()}_x', f'{landmark.name.lower()}_y', f'{landmark.name.lower()}_z']

pose_classes = [d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))]
print(f"[INFO] Found pose classes: {pose_classes}")

with open(csv_output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header) 

    for class_name in pose_classes:
        class_folder_path = os.path.join(dataset_folder, class_name)
        
        for filename in os.listdir(class_folder_path):
            image_path = os.path.join(class_folder_path, filename)
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"WARNING: Could not read image {image_path}. Skipping.")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_world_landmarks:
                try:
                    landmarks = results.pose_world_landmarks.landmark
                    
                    row = [class_name]
                    for landmark in landmarks:
                        row.extend([landmark.x, landmark.y, landmark.z])
                    
                    writer.writerow(row)

                except Exception as e:
                    print(f"ERROR: Error processing {image_path}: {e}")
            else:
                print(f"WARNING: No pose detected in {image_path}. Skipping.")


print(f"[INFO] Feature extraction complete. 3D data saved to {csv_output_file}")

print("\n[INFO] Starting model training using MLP CLASSIFIER...")

data = pd.read_csv(csv_output_file)
data.dropna(inplace=True)

X = data.drop('class', axis=1) 
y = data['class']

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) 

model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42, early_stopping=True, verbose=True)

print("\n[INFO] Training the neural network (this may take a moment)...")

model.fit(X_train_scaled, y_train)

print("\n[INFO] Evaluating the model...")

y_pred = model.predict(X_test_scaled)


accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_, zero_division=0))


model_filename = 'arnis_mlp_classifier.joblib'
joblib.dump(model, model_filename)
scaler_filename = 'mlp_scaler.joblib' 
joblib.dump(scaler, scaler_filename)
class_names_filename = 'arnis_class_names.joblib'
joblib.dump(encoder.classes_, class_names_filename)

print(f"\n[INFO] Training complete. Model saved to {model_filename}, scaler to {scaler_filename}, and class names to {class_names_filename}")


pose.close()