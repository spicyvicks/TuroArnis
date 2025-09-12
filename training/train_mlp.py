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
    # Use the landmark's name (e.g., "NOSE", "LEFT_EYE") for the header
    header += [f'{landmark.name.lower()}_x', f'{landmark.name.lower()}_y', f'{landmark.name.lower()}_z']

pose_classes = [d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))]
print(f"[INFO] Found pose classes: {pose_classes}")

with open(csv_output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header) # Write the new 3D header

    # Loop through each class (e.g., 'correct_form', 'incorrect_form')
    for class_name in pose_classes:
        class_folder_path = os.path.join(dataset_folder, class_name)
        
        # Loop through each image in the class folder
        for filename in os.listdir(class_folder_path):
            image_path = os.path.join(class_folder_path, filename)
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"WARNING: Could not read image {image_path}. Skipping.")
                continue

            # Process the image to find pose landmarks
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            # --- Changed: Use 3D world landmarks for feature extraction ---
            if results.pose_world_landmarks:
                try:
                    # Get the 3D landmarks
                    landmarks = results.pose_world_landmarks.landmark
                    
                    # Flatten the landmark coordinates into a single list
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

# Load the newly created dataset
data = pd.read_csv(csv_output_file)
data.dropna(inplace=True)

# Separate features (X) and labels (y)
X = data.drop('class', axis=1) 
y = data['class']

# Encode string labels into numbers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# --- Feature Scaling: Crucial for Neural Networks ---
# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Use the same scaler fitted on the training data

# Initialize the MLP Classifier
# hidden_layer_sizes defines the architecture of the neural network.
# max_iter is the number of epochs (passes over the data).
model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42, early_stopping=True, verbose=True)

print("\n[INFO] Training the neural network (this may take a moment)...")
# Train the model on the SCALED data
model.fit(X_train_scaled, y_train)

print("\n[INFO] Evaluating the model...")
# Make predictions on the SCALED test data
y_pred = model.predict(X_test_scaled)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_, zero_division=0))

# Save the trained model, the encoder, AND THE SCALER
model_filename = 'arnis_mlp_classifier.joblib'
joblib.dump(model, model_filename)
scaler_filename = 'mlp_scaler.joblib' # The scaler MUST be saved too
joblib.dump(scaler, scaler_filename)
class_names_filename = 'arnis_class_names.joblib'
joblib.dump(encoder.classes_, class_names_filename)

print(f"\n[INFO] Training complete. Model saved to {model_filename}, scaler to {scaler_filename}, and class names to {class_names_filename}")

# Clean up the pose model
pose.close()