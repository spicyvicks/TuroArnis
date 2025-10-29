import os
import csv
import sys
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import shutil

def split_dataset(input_folder, train_folder, val_folder, val_split=0.2, random_seed=42):
    """Split dataset into training and validation folders."""
    np.random.seed(random_seed)
    
    # Create output directories
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    
    # Get all class folders
    class_folders = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]
    
    total_train = 0
    total_val = 0
    
    print("\n[INFO] Splitting dataset into training and validation sets...")
    for class_name in tqdm(class_folders, desc="Processing classes"):
        # Create class folders in train and val
        train_class_folder = os.path.join(train_folder, class_name)
        val_class_folder = os.path.join(val_folder, class_name)
        os.makedirs(train_class_folder, exist_ok=True)
        os.makedirs(val_class_folder, exist_ok=True)
        
        # Get all images in the class
        class_path = os.path.join(input_folder, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Randomly shuffle images
        np.random.shuffle(images)
        
        # Calculate split point
        split_idx = int(len(images) * (1 - val_split))
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Copy images to respective folders
        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_class_folder, img)
            shutil.copy2(src, dst)
        total_train += len(train_images)
        
        for img in val_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(val_class_folder, img)
            shutil.copy2(src, dst)
        total_val += len(val_images)
    
    print(f"\n[INFO] Dataset split complete:")
    print(f"  - Training images: {total_train}")
    print(f"  - Validation images: {total_val}")
    return total_train, total_val

def extract_features_from_folder(folder_path, output_csv, num_processes=4):
    """Extract features from all images in a folder and save to CSV."""
    print(f"\n[INFO] Extracting features from: {folder_path}")
    
    header = ['class'] + [f'{ax}_{i}' for i in range(33) for ax in ['x', 'y', 'z']]
    pose_classes = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
    
    all_image_paths = []
    path_to_class_map = {}
    for class_name in pose_classes:
        class_folder_path = os.path.join(folder_path, class_name)
        for filename in os.listdir(class_folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(class_folder_path, filename)
                all_image_paths.append(full_path)
                path_to_class_map[full_path] = class_name
    
    print(f"  - Using {num_processes} processes for {len(all_image_paths)} images...")
    
    with Pool(processes=num_processes, initializer=init_worker) as pool:
        image_coords = pool.imap(extract_coordinates_from_image, all_image_paths)
        results = list(tqdm(image_coords, total=len(all_image_paths), desc="  - Extracting Coordinates"))
    
    success_count = 0
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, coords in enumerate(results):
            if coords:
                image_path = all_image_paths[i]
                class_name = path_to_class_map[image_path]
                writer.writerow([class_name] + coords)
                success_count += 1
    
    print(f"  - Successfully processed {success_count} images")
    return success_count

# Your existing worker and extraction functions here
worker_pose_instance = None

def init_worker():
    global worker_pose_instance
    worker_pose_instance = mp.solutions.pose.Pose(
        static_image_mode=True, min_detection_confidence=0.5
    )

def extract_coordinates_from_image(image_path):
    global worker_pose_instance
    if worker_pose_instance is None:
        init_worker()
        
    image = cv2.imread(image_path)
    if image is None: 
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = worker_pose_instance.process(image_rgb)
    
    if not results.pose_world_landmarks:
        return None
        
    try:
        coordinates = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark]).flatten().tolist()
        return coordinates
    except Exception:
        return None

# Your existing plotting functions here
def plot_training_history(history, save_path, plt):
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

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import joblib
    import seaborn as sns
    import tensorflow as tf
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    dataset_folder = os.path.join(project_root, 'dataset')
    train_folder = os.path.join(project_root, 'dataset_split', 'train')
    val_folder = os.path.join(project_root, 'dataset_split', 'val')
    
    train_csv = os.path.join(project_root, 'arnis_poses_coordinates_train.csv')
    val_csv = os.path.join(project_root, 'arnis_poses_coordinates_val.csv')
    
    models_dir = os.path.join(project_root, 'models')
    model_save_path = os.path.join(models_dir, 'arnis_coordinates_classifier.keras')
    encoder_path = os.path.join(models_dir, 'label_encoder.joblib')
    
    os.makedirs(models_dir, exist_ok=True)
    
    # Split dataset into train and validation sets
    print("\n[STAGE 1] Splitting dataset...")
    total_train, total_val = split_dataset(dataset_folder, train_folder, val_folder)
    
    # Extract features for both sets
    print("\n[STAGE 2] Extracting features...")
    train_samples = extract_features_from_folder(train_folder, train_csv)
    val_samples = extract_features_from_folder(val_folder, val_csv)
    
    print("\n[STAGE 3] Preparing data for training...")
    # Load training data
    train_data = pd.read_csv(train_csv).dropna()
    X_train = train_data.drop('class', axis=1).values
    y_train_labels = train_data['class'].values
    
    # Load validation data
    val_data = pd.read_csv(val_csv).dropna()
    X_val = val_data.drop('class', axis=1).values
    y_val_labels = val_data['class'].values
    
    # Fit label encoder on training data only
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_labels)
    y_val = label_encoder.transform(y_val_labels)
    
    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)
    num_features = X_train.shape[1]
    
    print(f"\n[INFO] Training configuration:")
    print(f"  - Number of features: {num_features}")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Validation samples: {len(X_val)}")
    
    print("\n[STAGE 4] Training model...")
    
    # Model definition (same as your existing model)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        weight_decay=1e-5
    )
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((num_features,)),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.L2(1e-4)),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(0.4),
        
        tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.L2(1e-4)),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.L2(1e-4)),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.L2(1e-4)),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(0.1),
        
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    es_callback = tf.keras.callbacks.EarlyStopping(
        patience=50,
        monitor='val_accuracy',
        restore_best_weights=True,
        mode='max'
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=20,
        min_lr=1e-6,
        mode='min'
    )
    
    print("\n  - Starting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=500,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=[es_callback, reduce_lr],
        verbose=1
    )
    
    print("\n[STAGE 5] Evaluating model...")
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal Validation Accuracy: {val_acc:.4f}")
    
    y_pred = np.argmax(model.predict(X_val), axis=1)
    print("\n=== Classification Report ===")
    print(classification_report(y_val, y_pred, target_names=class_names, zero_division=0))
    
    # Save model artifacts
    print("\n[STAGE 6] Saving model artifacts...")
    history_plot_path = os.path.join(models_dir, 'training_history.png')
    plot_training_history(history, history_plot_path, plt)
    
    model.save(model_save_path)
    joblib.dump(label_encoder, encoder_path)
    
    print(f"\n[SUCCESS] Training complete!")
    print(f"  - Model saved to: {model_save_path}")
    print(f"  - Label encoder saved to: {encoder_path}")
    print(f"  - Training history plot saved to: {history_plot_path}")