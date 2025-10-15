import os
import csv
import sys
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from duo.duo_instance import ArnisClassifiers

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def calculate_angle_3d(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    dot_product = np.dot(ba, bc)
    magnitude = np.linalg.norm(ba) * np.linalg.norm(bc)
    if magnitude < 1e-10: return 0.0
    cosine_angle = np.clip(dot_product / magnitude, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def extract_features_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None: return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if not results.pose_world_landmarks: return None
    try:
        landmarks = results.pose_world_landmarks.landmark
        lm_data = {mp_pose.PoseLandmark(idx).name.lower(): [lm.x, lm.y, lm.z] for idx, lm in enumerate(landmarks)}
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
        print(f"[error] processing landmarks in {image_path}: {e}")
        return None

def plot_training_history(history, save_path):
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Model Accuracy Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.ylim(0, 1.05) 

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Model Loss Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n[info] Training history plot saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    dataset_folder = os.path.join(project_root, 'dataset_multiclass_2')
    csv_output_file = os.path.join(project_root, 'arnis_poses_features.csv')
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    RUN_EXTRACTION = True

    if RUN_EXTRACTION:
        print("\n[info] starting feature extraction...")
        feature_columns = ['left_elbow', 'left_shoulder', 'left_hip', 'left_knee',
                           'right_elbow', 'right_shoulder', 'right_hip', 'right_knee']
        header = ['class'] + feature_columns
        pose_classes = sorted([d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))])
        with open(csv_output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for class_name in pose_classes:
                class_folder_path = os.path.join(dataset_folder, class_name)
                for filename in os.listdir(class_folder_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(class_folder_path, filename)
                        angles = extract_features_from_image(image_path)
                        if angles:
                            writer.writerow([class_name] + [angles.get(joint, 0) for joint in feature_columns])
        print(f"[info] feature extraction complete.")
    else:
        print(f"[info] skipping feature extraction.")
    pose.close()

    print("\n[info] preparing data...")
    data = pd.read_csv(csv_output_file).dropna()
    X = data.drop('class', axis=1).values
    y_labels = data['class'].values
    
    label_encoder = LabelEncoder()
    y_integers = label_encoder.fit_transform(y_labels)
    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)
    print(f"[info] found classes: {class_names}")

    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y_integers, test_size=0.2, random_state=42, stratify=y_integers
    )
    print(f"[info] data split: {len(X_pool)} for training pool, {len(X_test)} for final testing.")

    print("\n" + "="*50)
    print("      K-FOLD CROSS-VALIDATION (5 FOLDS)")
    print("="*50)
    
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    tf_val_scores = []
    rf_val_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_pool, y_pool)):
        print(f"--- FOLD {fold+1}/{n_splits} ---")
        X_train_fold, X_val_fold = X_pool[train_idx], X_pool[val_idx]
        y_train_fold, y_val_fold = y_pool[train_idx], y_pool[val_idx]

        y_train_tf = tf.keras.utils.to_categorical(y_train_fold, num_classes=num_classes)
        y_val_tf = tf.keras.utils.to_categorical(y_val_fold, num_classes=num_classes)
        
        temp_tf_model = Sequential([
            Dense(64, activation='relu', input_shape=(X.shape[1],)), Dropout(0.2),
            Dense(32, activation='relu'), Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        temp_tf_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        temp_tf_model.fit(X_train_fold, y_train_tf, epochs=50, batch_size=8, verbose=0)
        
        _, acc_tf = temp_tf_model.evaluate(X_val_fold, y_val_tf, verbose=0)
        tf_val_scores.append(acc_tf)
        print(f"  - TF validation accuracy: {acc_tf:.4f}")

        temp_rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        temp_rf_model.fit(X_train_fold, y_train_fold)
        
        acc_rf = temp_rf_model.score(X_val_fold, y_val_fold)
        rf_val_scores.append(acc_rf)
        print(f"  - RF validation accuracy: {acc_rf:.4f}")

    print("\n--- average Cross-Validation scores ---")
    print(f"average TF accuracy: {np.mean(tf_val_scores):.4f} (+/- {np.std(tf_val_scores):.4f})")
    print(f"average RF accuracy: {np.mean(rf_val_scores):.4f} (+/- {np.std(rf_val_scores):.4f})")

    print("\n[info] training final models on the full training pool...")
    
    y_pool_tf = tf.keras.utils.to_categorical(y_pool, num_classes=num_classes)
    y_test_tf = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

    final_tf_model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],)), Dropout(0.2),
        Dense(32, activation='relu'), Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    final_tf_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("[info] starting final TensorFlow model training...")
    history = final_tf_model.fit(
        X_pool, 
        y_pool_tf, 
        epochs=50, 
        batch_size=8, 
        validation_data=(X_test, y_test_tf), 
        verbose=1
    )
    print("[info] final tensorflow model trained.")

    # --- Generate and Save the Training History Plot ---
    history_plot_path = os.path.join(models_dir, 'training_history.png')
    plot_training_history(history, history_plot_path)


    # --- Final Random Forest Model (No changes needed here) ---
    final_rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    final_rf_model.fit(X_pool, y_pool)
    print("[info] final random forest model trained.")

    print("\n" + "="*50)
    print("      FINAL EVALUATION ON HOLD-OUT TEST SET")
    print("="*50)
    
    tf_pred_proba = final_tf_model.predict(X_test, verbose=0)
    rf_pred_proba = final_rf_model.predict_proba(X_test)
    combined_proba = (tf_pred_proba + rf_pred_proba) / 2.0
    
    y_pred_tf = np.argmax(tf_pred_proba, axis=1)
    y_pred_rf = np.argmax(rf_pred_proba, axis=1)
    y_pred_hybrid = np.argmax(combined_proba, axis=1)

    print("\n--- Final Accuracy Scores ---")
    print(f"Final TensorFlow Model Accuracy: {accuracy_score(y_test, y_pred_tf):.4f}")
    print(f"Final Random Forest Model Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"FINAL HYBRID MODEL ACCURACY:     {accuracy_score(y_test, y_pred_hybrid):.4f}\n")

    print("\n--- Final Classification Report (Hybrid Model) ---")
    print(classification_report(y_test, y_pred_hybrid, target_names=class_names))

    print("\n[info] saving final models and unified wrapper...")
    tf_model_path_rel = 'models/arnis_tf_classifier.h5'
    rf_model_path_rel = 'models/arnis_rf_classifier.joblib'
    encoder_path_rel = 'models/label_encoder.joblib'

    final_tf_model.save(os.path.join(project_root, tf_model_path_rel))
    joblib.dump(final_rf_model, os.path.join(project_root, rf_model_path_rel))
    joblib.dump(label_encoder, os.path.join(project_root, encoder_path_rel))
    
    arnis_classifier_wrapper = ArnisClassifiers(
        tf_model_path=tf_model_path_rel,
        rf_model_path=rf_model_path_rel,
        encoder_path=encoder_path_rel
    )
    
    wrapper_path = os.path.join(models_dir, 'arnis_classifiers.joblib')
    joblib.dump(arnis_classifier_wrapper, wrapper_path)

    print("\n[success] training and evaluation complete.")
    print(f"final unified wrapper saved to: {wrapper_path}")