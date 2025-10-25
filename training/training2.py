import os
import csv
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def compute_angle(p1, p2, p3):
    """Compute angle between three points (p1-p2-p3) in 3D space."""
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2]])
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1], p3[2]-p2[2]])
    
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # Clip to handle floating point errors
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.arccos(cosine)
    return np.degrees(angle)

def compute_vector_angle(v1, v2):
    """Compute angle between two vectors."""
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))

def extract_pose_features(coords):
    """Extract meaningful angles and ratios from pose coordinates.
    
    Args:
        coords: Flattened array of shape (99,) containing x,y,z coordinates
               for 33 landmarks in order [x1,y1,z1,x2,y2,z2,...]
    
    Returns:
        Array of computed angles and features
    """
    # Reshape to (33, 3) for easier indexing
    points = coords.reshape(-1, 3)
    
    # Key point indices in MediaPipe pose
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    
    features = []
    
    # 1. Arm angles (crucial for strikes and blocks)
    left_arm_angle = compute_angle(
        points[LEFT_SHOULDER], points[LEFT_ELBOW], points[LEFT_WRIST])
    right_arm_angle = compute_angle(
        points[RIGHT_SHOULDER], points[RIGHT_ELBOW], points[RIGHT_WRIST])
    features.extend([left_arm_angle, right_arm_angle])
    
    # 2. Shoulder line angle relative to horizontal
    shoulder_vector = points[RIGHT_SHOULDER] - points[LEFT_SHOULDER]
    horizontal = np.array([1, 0, 0])
    shoulder_angle = compute_vector_angle(shoulder_vector, horizontal)
    features.append(shoulder_angle)
    
    # 3. Torso orientation (hip-shoulder alignment)
    hip_vector = points[RIGHT_HIP] - points[LEFT_HIP]
    torso_twist = compute_vector_angle(shoulder_vector, hip_vector)
    features.append(torso_twist)
    
    # 4. Knee angles (stance)
    left_knee_angle = compute_angle(
        points[LEFT_HIP], points[LEFT_KNEE], points[LEFT_ANKLE])
    right_knee_angle = compute_angle(
        points[RIGHT_HIP], points[RIGHT_KNEE], points[RIGHT_ANKLE])
    features.extend([left_knee_angle, right_knee_angle])
    
    # 5. Upper arm angles relative to torso
    torso_vertical = points[LEFT_SHOULDER] - points[LEFT_HIP]
    left_upper_arm = points[LEFT_ELBOW] - points[LEFT_SHOULDER]
    right_upper_arm = points[RIGHT_ELBOW] - points[RIGHT_SHOULDER]
    
    left_arm_torso_angle = compute_vector_angle(torso_vertical, left_upper_arm)
    right_arm_torso_angle = compute_vector_angle(torso_vertical, right_upper_arm)
    features.extend([left_arm_torso_angle, right_arm_torso_angle])
    
    # 6. Stance width (normalized by shoulder width)
    stance_width = np.linalg.norm(points[LEFT_ANKLE] - points[RIGHT_ANKLE])
    shoulder_width = np.linalg.norm(points[LEFT_SHOULDER] - points[RIGHT_SHOULDER])
    stance_ratio = stance_width / (shoulder_width + 1e-6)  # Avoid division by zero
    features.append(stance_ratio)
    
    # 7. Forward lean angle
    hip_center = (points[LEFT_HIP] + points[RIGHT_HIP]) / 2
    shoulder_center = (points[LEFT_SHOULDER] + points[RIGHT_SHOULDER]) / 2
    spine_vector = shoulder_center - hip_center
    vertical = np.array([0, 1, 0])
    lean_angle = compute_vector_angle(spine_vector, vertical)
    features.append(lean_angle)
    
    # 8. Asymmetry measures (useful for detecting incorrect form)
    arm_length_ratio = (np.linalg.norm(points[LEFT_WRIST] - points[LEFT_SHOULDER]) /
                       (np.linalg.norm(points[RIGHT_WRIST] - points[RIGHT_SHOULDER]) + 1e-6))
    features.append(arm_length_ratio)
    
    return np.array(features)

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
    print(f"\n[INFO] Training history plot saved to: {save_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues')
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('Actual Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(save_path)
    print(f"[INFO] Confusion matrix plot saved to: {save_path}")
    plt.close()

def main():
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report
    
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.append(project_root)
    
    csv_input_file = os.path.join(project_root, 'arnis_poses_coordinates.csv')
    models_dir = os.path.join(project_root, 'models')
    model_save_path = os.path.join(models_dir, 'arnis_angles_classifier.keras')
    encoder_save_path = os.path.join(models_dir, 'label_encoder_angles.joblib')
    os.makedirs(models_dir, exist_ok=True)
    
    print("\n[STAGE 1] Loading and transforming pose data...")
    
    # Load raw coordinates
    data = pd.read_csv(csv_input_file)
    X_raw = data.drop('class', axis=1).values
    y_labels = data['class'].values
    
    # Transform coordinates into angles and features
    print("  - Computing pose angles and features...")
    X_features = np.array([extract_pose_features(coords) for coords in tqdm(X_raw)])
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)
    
    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)
    num_features = X_features.shape[1]
    print(f"  - Training on {num_features} pose features for {num_classes} classes")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  - Data split: {len(X_train)} for training, {len(X_test)} for testing")
    
    print("\n[STAGE 2] Creating and training model...")
    
    # Create model optimized for angle-based features
    model = tf.keras.models.Sequential([
        # Input and normalization
        tf.keras.layers.Input((num_features,)),
        tf.keras.layers.BatchNormalization(),
        
        # Dense layers with residual connections
        tf.keras.layers.Dense(64, activation=None),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(64, activation=None),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(32, activation=None),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(0.2),
        
        # Output layer
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile with Adam optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        weight_decay=1e-5
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Training callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=50,
            restore_best_weights=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=20,
            min_lr=1e-6
        )
    ]
    
    # Train model
    print("\n  - Starting model training...")
    history = model.fit(
        X_train,
        y_train,
        epochs=500,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n[SUCCESS] Model training complete.")
    
    # Evaluation
    print("\n" + "="*50)
    print("      FINAL EVALUATION AND SAVING")
    print("="*50)
    
    val_loss, val_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Model Accuracy on Test Set: {val_acc:.4f}")
    
    # Predictions and classification report
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\n--- Final Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    
    # Save plots
    history_plot_path = os.path.join(models_dir, 'training_history_angles.png')
    plot_training_history(history, history_plot_path)
    
    cm_plot_path = os.path.join(models_dir, 'confusion_matrix_angles.png')
    plot_confusion_matrix(y_test, y_pred, class_names, cm_plot_path)
    
    # Save model and encoder
    print("\n[INFO] Saving model and label encoder...")
    model.save(model_save_path)
    import joblib
    joblib.dump(label_encoder, encoder_save_path)
    
    print(f"\n[SUCCESS] Process complete!")
    print(f"  - Model saved to: {model_save_path}")
    print(f"  - Label encoder saved to: {encoder_save_path}")
    
    # Print feature importance analysis
    feature_names = [
        "Left Arm Angle", "Right Arm Angle",
        "Shoulder Line Angle", "Torso Twist",
        "Left Knee Angle", "Right Knee Angle",
        "Left Arm-Torso Angle", "Right Arm-Torso Angle",
        "Stance Width Ratio", "Forward Lean Angle",
        "Arm Length Ratio"
    ]
    
    # Create a simple dense layer to check feature importance
    importance_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(num_classes, use_bias=False)
    ])
    importance_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    importance_model.fit(X_features, y, epochs=10, verbose=0)
    
    # Get feature importance scores
    importance = np.abs(importance_model.layers[0].get_weights()[0]).mean(axis=1)
    importance = 100 * importance / importance.sum()
    
    print("\n=== Feature Importance Analysis ===")
    for name, score in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True):
        print(f"{name:20s}: {score:5.1f}%")

if __name__ == "__main__":
    main()
