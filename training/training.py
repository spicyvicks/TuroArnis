import os
import csv
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def calculate_angle_3d(a, b, c):
    #3 3D points using cross products for direction
    
    #angle to numpy arrays
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    #vector from point
    ba = a - b  
    bc = c - b  
    
    cross_product = np.cross(ba, bc)
    dot_product = np.dot(ba, bc)
    
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)
    
    #no zero product
    if magnitude_ba * magnitude_bc < 1e-10:
        return 0.0
        
    #signed angle using arctan2(norm of cross product, dot product) = angle, range [-π, π]
    angle = np.arctan2(np.linalg.norm(cross_product), dot_product)
    angle_deg = np.degrees(angle)
    
    #y component of cross pfor sign
    if cross_product[1] < 0:
        angle_deg = -angle_deg
        
    return angle_deg

def extract_features_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_world_landmarks:
        print(f"[WARNING] No pose detected in image: {image_path}")
        return None

    try:
        landmarks = results.pose_world_landmarks.landmark
        
        #dictionary landmark coordinates
        lm_data = {}
        for idx, landmark in enumerate(landmarks):
            #PoseLandmark enum names and store
            landmark_name = mp_pose.PoseLandmark(idx).name.lower()
            lm_data[landmark_name] = [landmark.x, landmark.y, landmark.z]

        angles = {
            'left_elbow': calculate_angle_3d(lm_data['left_shoulder'], lm_data['left_elbow'], lm_data['left_wrist']),
            'left_shoulder': calculate_angle_3d(lm_data['left_hip'], lm_data['left_shoulder'], lm_data['left_elbow']),
            'left_hip': calculate_angle_3d(lm_data['left_shoulder'], lm_data['left_hip'], lm_data['left_knee']),
            'left_knee': calculate_angle_3d(lm_data['left_hip'], lm_data['left_knee'], lm_data['left_ankle']),
            'right_elbow': calculate_angle_3d(lm_data['right_shoulder'], lm_data['right_elbow'], lm_data['right_wrist']),
            'right_shoulder': calculate_angle_3d(lm_data['right_hip'], lm_data['right_shoulder'], lm_data['right_elbow']),
            'right_hip': calculate_angle_3d(lm_data['right_shoulder'], lm_data['right_hip'], lm_data['right_knee']),
            'right_knee': calculate_angle_3d(lm_data['right_hip'], lm_data['right_knee'], lm_data['right_ankle']),
        }
        return angles
    except Exception as e:
        print(f"[ERROR] Error processing landmarks in {image_path}: {str(e)}")
        return None

#training
if __name__ == "__main__":
    #root dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    dataset_folder = os.path.join(root_dir, 'dataset_multiclass_2')
    csv_output_file = os.path.join(root_dir, 'arnis_poses_for_tf_classification.csv')

    print(f"[DEBUG] Current working directory: {os.getcwd()}")
    print(f"[DEBUG] Dataset folder absolute path: {os.path.abspath(dataset_folder)}")
    print(f"[DEBUG] CSV output absolute path: {os.path.abspath(csv_output_file)}")

    #save
    model_folder = os.path.join(root_dir, 'models_tf')
    if not os.path.exists(dataset_folder):
        raise ValueError(f"Dataset folder '{dataset_folder}' not found")
    
    classes = sorted([d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))])
    if not classes:
        raise ValueError(f"No class folders found in '{dataset_folder}'")
        
    print(f"[info] found {len(classes)} classes: {classes}")
    for class_name in classes:
        class_path = os.path.join(dataset_folder, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"[info] class '{class_name}' has {len(images)} images")

    #feature extraction
    print("[info] starting feature extraction...")
    pose_classes = sorted([d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))])
    
    header = ['class'] + [
        'left_elbow', 'left_shoulder', 'left_hip', 'left_knee',
        'right_elbow', 'right_shoulder', 'right_hip', 'right_knee'
    ]

    with open(csv_output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for class_name in pose_classes:
            print(f"[DEBUG] processing class: {class_name}")
            class_folder_path = os.path.join(dataset_folder, class_name)
            image_count = 0
            processed_count = 0
            
            for filename in os.listdir(class_folder_path):
                image_path = os.path.join(class_folder_path, filename)
                print(f"[DEBUG] processing image: {filename}")
                angles = extract_features_from_image(image_path)
                
                if angles:
                    row = [class_name] + [angles.get(joint, 0) for joint in header[1:]]
                    writer.writerow(row)
                    processed_count += 1
                else:
                    print(f"[WARNING] Failed to extract features from {filename}")
                image_count += 1
            
            print(f"[DEBUG] Class {class_name}: processed {processed_count}/{image_count} images")

    print(f"[info] feature extraction complete. data saved to {csv_output_file}")
    pose.close()

    #tensorflow
    print("\n[info] preparing data for tensorflow...")
    data = pd.read_csv(csv_output_file)

    if data.empty:
        raise ValueError("No data was loaded from the CSV file")

    print(f"[info] loaded {len(data)} samples")

    #x NaN value
    if data.isnull().any().any():
        print("[warning] found NaN values in data, removing affected rows...")
        data = data.dropna()
        print(f"[info] {len(data)} samples remaining after cleaning")

    #csv quick check
    print("\n[info] CSV file preview:")
    with open(csv_output_file, 'r') as f:
        for i, line in enumerate(f):
            if i < 5:
                print(line.strip())
            else:
                break

    X = data.drop('class', axis=1).values  
    y_labels = data['class'].values  

    #check data if exists
    if len(X) == 0 or len(y_labels) == 0:
        raise ValueError("No valid samples found in the dataset")

    #print classes
    unique_classes = np.unique(y_labels)
    print(f"[info] found {len(unique_classes)} unique classes: {unique_classes}")

    #print labels
    label_encoder = LabelEncoder()
    y_integers = label_encoder.fit_transform(y_labels)

    #check encoded labels
    if len(np.unique(y_integers)) < 2:
        raise ValueError("Need at least 2 classes for classification")

    num_classes = len(np.unique(y_integers))
    y_categorical = tf.keras.utils.to_categorical(y_integers, num_classes=num_classes)
    
    num_features = X.shape[1]
    num_classes = len(pose_classes)

    #training & testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical)

    #build
    print("\n[info] building the neural network model...")
    model = Sequential([
        Dense(64, activation='relu', input_shape=(num_features,)),
        tf.keras.layers.Dropout(0.2),
        Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    print("[info] compiling the model...")
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    model.summary()

    #training
    print("\n[info] training the model...")
    history = model.fit(
        X_train, y_train,
        epochs=50, 
        batch_size=8,
        validation_data=(X_test, y_test)
    )

    #evaluation
    print("\n[info] evaluating the model on the test set...")
    y_pred_proba = model.predict(X_test)
    y_pred_indices = np.argmax(y_pred_proba, axis=1)
    y_test_indices = np.argmax(y_test, axis=1)

    print("\nclassification report:")
    print(classification_report(y_test_indices, y_pred_indices, target_names=pose_classes))

    print("\n[info] generating confusion matrix visualization...")
    cm = confusion_matrix(y_test_indices, y_pred_indices)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, 
                annot=True,  
                fmt='d',
                cmap='Blues',
                xticklabels=pose_classes,
                yticklabels=pose_classes)
    
    plt.title('Confusion Matrix - Arnis')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    confusion_matrix_path = os.path.join(model_folder, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    print(f"[info] confusion matrix visualization saved to {confusion_matrix_path}")
    plt.close()
    
    
    print("\n[info] saving model and label encoder...")
    model_folder = 'models_tf'
    os.makedirs(model_folder, exist_ok=True)
    model.save(os.path.join(model_folder, 'arnis_tf_classifier.h5'))
    joblib.dump(label_encoder, os.path.join(model_folder, 'label_encoder.joblib'))
    
    print("[info] training complete and artifacts saved.")

def train_hybrid_model(X, y, test_size=0.2):
    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=test_size, 
        random_state=42, 
        stratify=y_encoded
    )
    
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    # Save models
    joblib.dump(rf_model, 'models/arnis_random_forest_classifier.joblib')
    joblib.dump(encoder, 'models/label_encoder.joblib')
    
    return rf_model, encoder