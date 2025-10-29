import os
import csv
import sys
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

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

def extract_features_from_folder(folder_path, output_csv, num_processes=4):
    """Extract features from all images in a folder and save to CSV."""
    print(f"\n[INFO] Extracting features from: {folder_path}")
    
    header = ['class'] + [f'{ax}_{i}' for i in range(33) for ax in ['x', 'y', 'z']]
    pose_classes = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
    
    all_image_paths = []
    path_to_class_map = {}
    class_counts = {}
    
    for class_name in pose_classes:
        class_folder_path = os.path.join(folder_path, class_name)
        class_counts[class_name] = 0
        for filename in os.listdir(class_folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(class_folder_path, filename)
                all_image_paths.append(full_path)
                path_to_class_map[full_path] = class_name
                class_counts[class_name] += 1
    
    print("\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"  - {class_name}: {count} images")
    
    print(f"\n  - Using {num_processes} processes for {len(all_image_paths)} images...")
    
    with Pool(processes=num_processes, initializer=init_worker) as pool:
        image_coords = pool.imap(extract_coordinates_from_image, all_image_paths)
        results = list(tqdm(image_coords, total=len(all_image_paths), desc="  - Extracting Coordinates"))
    
    success_count = 0
    class_success_counts = {}
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, coords in enumerate(results):
            if coords:
                image_path = all_image_paths[i]
                class_name = path_to_class_map[image_path]
                writer.writerow([class_name] + coords)
                success_count += 1
                class_success_counts[class_name] = class_success_counts.get(class_name, 0) + 1
    
    print("\nSuccessful extractions per class:")
    for class_name, count in class_success_counts.items():
        total = class_counts[class_name]
        print(f"  - {class_name}: {count}/{total} ({count/total*100:.1f}%)")
    
    print(f"\n  - Total successful extractions: {success_count}/{len(all_image_paths)}")
    return success_count

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
    
    # Define your training and validation dataset folders here
    train_folder = os.path.join(project_root, 'dataset_multiclass_2')  # Your main dataset
    val_folder = os.path.join(project_root, 'dataset', 'validation')   # Your validation dataset
    
    # Output paths
    train_csv = os.path.join(project_root, 'arnis_poses_coordinates_train.csv')
    val_csv = os.path.join(project_root, 'arnis_poses_coordinates_val.csv')
    
    models_dir = os.path.join(project_root, 'models')
    model_save_path = os.path.join(models_dir, 'arnis_coordinates_classifier.keras')
    encoder_path = os.path.join(models_dir, 'label_encoder.joblib')
    
    os.makedirs(models_dir, exist_ok=True)
    
    # Check if validation folder exists
    if not os.path.exists(val_folder):
        print(f"\n[ERROR] Validation folder not found: {val_folder}")
        print("Please create a 'validation' folder in your dataset directory")
        print("with the following structure:")
        print("\ndataset/")
        print("    validation/")
        print("        class1/")
        print("        class2/")
        print("        ...")
        sys.exit(1)
    
    # Extract features
    print("\n[STAGE 1] Extracting features...")
    print("\nProcessing TRAINING data:")
    train_samples = extract_features_from_folder(train_folder, train_csv)
    print("\nProcessing VALIDATION data:")
    val_samples = extract_features_from_folder(val_folder, val_csv)
    
    print("\n[STAGE 2] Preparing data for training...")
    # Load training data
    train_data = pd.read_csv(train_csv).dropna()
    X_train = train_data.drop('class', axis=1).values
    y_train_labels = train_data['class'].values
    
    # Load validation data
    val_data = pd.read_csv(val_csv).dropna()
    X_val = val_data.drop('class', axis=1).values
    y_val_labels = val_data['class'].values
    
    # Fit label encoder on all classes (both train and val)
    all_labels = np.concatenate([y_train_labels, y_val_labels])
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    
    # Transform labels
    y_train = label_encoder.transform(y_train_labels)
    y_val = label_encoder.transform(y_val_labels)
    
    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)
    num_features = X_train.shape[1]
    
    print(f"\n[INFO] Training configuration:")
    print(f"  - Number of features: {num_features}")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Validation samples: {len(X_val)}")
    
    print("\n[STAGE 3] Training model...")
    
    # Model definition
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
    
    print("\n[STAGE 4] Evaluating model...")
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal Validation Accuracy: {val_acc:.4f}")
    
    y_pred = np.argmax(model.predict(X_val), axis=1)
    print("\n=== Classification Report ===")
    print(classification_report(y_val, y_pred, target_names=class_names, zero_division=0))
    
    # Save model artifacts
    print("\n[STAGE 5] Saving model artifacts...")
    history_plot_path = os.path.join(models_dir, 'training_history.png')
    plot_training_history(history, history_plot_path, plt)
    
    model.save(model_save_path)
    joblib.dump(label_encoder, encoder_path)
    
    print(f"\n[SUCCESS] Training complete!")
    print(f"  - Model saved to: {model_save_path}")
    print(f"  - Label encoder saved to: {encoder_path}")
    print(f"  - Training history plot saved to: {history_plot_path}")