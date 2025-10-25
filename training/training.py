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

def plot_training_history(history, save_path, plt):
    plt.figure(figsize=(15, 6))
    #acc
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Model Accuracy Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.ylim(0, 1.05) 
    #loss
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
    print(f"\n[INFO] Training history plot (linear graph) saved to: {save_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path, plt, sns, confusion_matrix):
    #heatmap
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


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg') 
    import joblib
    import seaborn as sns
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.append(project_root)

    dataset_folder = os.path.join(project_root, 'dataset_multiclass_2')
    csv_output_file = os.path.join(project_root, 'arnis_poses_coordinates.csv')
    models_dir = os.path.join(project_root, 'models')
    model_save_path = os.path.join(models_dir, 'arnis_coordinates_classifier.keras')
    encoder_path = os.path.join(models_dir, 'label_encoder.joblib')

    os.makedirs(models_dir, exist_ok=True)
    
    RUN_FEATURE_EXTRACTION = True 

    if RUN_FEATURE_EXTRACTION:
        print("\n[STAGE 1] Starting Coordinate Feature Extraction...")
        
        header = ['class'] + [f'{ax}_{i}' for i in range(33) for ax in ['x', 'y', 'z']]
        
        pose_classes = sorted([d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))])
        
        all_image_paths = []
        path_to_class_map = {}
        for class_name in pose_classes:
            class_folder_path = os.path.join(dataset_folder, class_name)
            for filename in os.listdir(class_folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(class_folder_path, filename)
                    all_image_paths.append(full_path)
                    path_to_class_map[full_path] = class_name
        
        num_processes = 4
        print(f"  - Using {num_processes} processes for {len(all_image_paths)} images...")

        with Pool(processes=num_processes, initializer=init_worker) as pool:
            image_coords = pool.imap(extract_coordinates_from_image, all_image_paths)
            results = list(tqdm(image_coords, total=len(all_image_paths), desc="  - Extracting Coordinates"))

        with open(csv_output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i, coords in enumerate(results):
                if coords:
                    image_path = all_image_paths[i]
                    class_name = path_to_class_map[image_path]
                    writer.writerow([class_name] + coords)
        
        print("\n[SUCCESS] Coordinate extraction complete. CSV saved.")
        print("="*50)
    else:
        print("\n[STAGE 1] Skipping coordinate extraction. Using existing CSV.")
        print("="*50)

    print("\n[STAGE 2] Starting Model Training...")
    
    data = pd.read_csv(csv_output_file).dropna()
    X = data.drop('class', axis=1).values
    y_labels = data['class'].values
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)
    
    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)
    num_features = X.shape[1]
    print(f"  - Training on {num_features} features for {num_classes} classes.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  - Data split: {len(X_train)} for training, {len(X_test)} for testing.")

    # Define optimizer with momentum and weight decay
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,  # Initial learning rate
        beta_1=0.9,  # momentum
        beta_2=0.999,
        weight_decay=1e-5  # L2 regularization
    )
    
    # Enhanced model architecture
    model = tf.keras.models.Sequential([
        # Input and normalization
        tf.keras.layers.Input((num_features,)),
        tf.keras.layers.BatchNormalization(),
        
        # First block - larger features
        tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.L2(1e-4)),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(0.4),
        
        # Second block - medium features
        tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.L2(1e-4)),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(0.3),
        
        # Third block - focused features
        tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.L2(1e-4)),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(0.2),
        
        # Fourth block - class-specific features
        tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.L2(1e-4)),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(0.1),
        
        # Output layer
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
    
    print("\n  - Starting model training... (Progress will be shown for each epoch below)")
    history = model.fit(
        X_train, y_train,
        epochs=500,
        batch_size=16,  # Smaller batch size for better generalization
        validation_data=(X_test, y_test),
        callbacks=[es_callback, reduce_lr],
        verbose=1
    )
    print("\n[SUCCESS] Model training complete.")

    print("\n" + "="*50)
    print("      FINAL EVALUATION AND SAVING")
    print("="*50)
    val_loss, val_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Model Accuracy on Test Set: {val_acc:.4f}\n")

    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    print("\n--- Final Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    
    print("\n[INFO] Generating and saving evaluation plots...")
    history_plot_path = os.path.join(models_dir, 'training_history.png')
    plot_training_history(history, history_plot_path, plt)
    
    cm_plot_path = os.path.join(models_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_test, y_pred, class_names, cm_plot_path, plt, sns, confusion_matrix)

    print("\n[INFO] Saving final model and label encoder...")
    model.save(model_save_path)
    joblib.dump(label_encoder, encoder_path)

    print(f"\n[SUCCESS] Process complete.")
    print(f"  - Best Keras model saved to: {model_save_path}")
    print(f"  - Label encoder saved to: {encoder_path}")