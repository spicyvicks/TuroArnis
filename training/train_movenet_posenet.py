import os
import cv2
import numpy as np
import time
import json
import gc
from pathlib import Path

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

from tf_keras.models import Sequential
from tf_keras.layers import Dense, Dropout, BatchNormalization
from tf_keras.optimizers import Adam
from tf_keras.callbacks import EarlyStopping, ReduceLROnPlateau


class MoveNetModel:
    def __init__(self, model_type='thunder'):
        model_dir = f'models/movenet_{model_type}'
        
        if not Path(model_dir).exists():
            raise FileNotFoundError(
                f"Model not found at {model_dir}\n"
            )
            
        print(f"Loading {model_type} model from {model_dir}...")
        self.model = tf.saved_model.load(model_dir)
        self.movenet = self.model.signatures['serving_default']
        self.classifier = None
        self.model_type = model_type
        
    def extract_features(self, image_path):
        """Extract keypoints from image using MoveNet"""
        image = cv2.imread(str(image_path))
        if image is None:
            return None
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        size = 256 if self.model_type == 'thunder' else 192
        image_resized = tf.image.resize_with_pad(
            tf.expand_dims(image, axis=0),
            size, size
        )
        image_resized = tf.cast(image_resized, dtype=tf.int32)
        
        outputs = self.movenet(image_resized)
        keypoints = outputs['output_0'].numpy()[0, 0, :, :]
        
        hip_center = (keypoints[11, :2] + keypoints[12, :2]) / 2.0
        keypoints[:, :2] = keypoints[:, :2] - hip_center
        
        return keypoints.flatten()
    
    def train(self, X_train, y_train, X_val, y_val, num_classes):
        input_dim = X_train.shape[1]
        
        self.classifier = Sequential([
            Dense(256, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        self.classifier.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        history = self.classifier.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Store training history for plotting
        self.training_history = history.history
        
        return history
    
    def predict(self, features):
        """Predict class from features"""
        if features is None or self.classifier is None:
            return -1
        pred = self.classifier.predict(np.expand_dims(features, axis=0), verbose=0)
        return np.argmax(pred)
    
    def save_classifier(self, path):
        """Save classifier only (not the pose detection model)"""
        self.classifier.save(f"{path}_classifier.keras")
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'input_shape': self.classifier.input_shape[1:]
        }
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        # Save training history for later comparison
        if hasattr(self, 'training_history'):
            # Convert numpy float32 to Python float for JSON serialization
            history_converted = {key: [float(val) for val in values] 
                                for key, values in self.training_history.items()}
            with open(f"{path}_history.json", 'w') as f:
                json.dump(history_converted, f, indent=2)


class PoseNetModel:
    """PoseNet using MoveNet Lightning as implementation - loads from disk"""
    def __init__(self):
        """Load lightweight pose model"""
        model_dir = 'models/movenet_lightning'
        
        if not Path(model_dir).exists():
            raise FileNotFoundError(
                f"Model not found at {model_dir}\n"
            )
            
        print(f"Loading PoseNet (MoveNet Lightning) from {model_dir}...")
        self.model = tf.saved_model.load(model_dir)
        self.movenet = self.model.signatures['serving_default']
        self.classifier = None
        self.name = "PoseNet (MoveNet Lightning)"
        
    def extract_features(self, image_path):
        """Extract keypoints from image"""
        image = cv2.imread(str(image_path))
        if image is None:
            return None
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = tf.image.resize_with_pad(
            tf.expand_dims(image, axis=0),
            192, 192
        )
        image_resized = tf.cast(image_resized, dtype=tf.int32)
        
        outputs = self.movenet(image_resized)
        keypoints = outputs['output_0'].numpy()[0, 0, :, :]
        
        hip_center = (keypoints[11, :2] + keypoints[12, :2]) / 2.0
        keypoints[:, :2] = keypoints[:, :2] - hip_center
        
        return keypoints.flatten()
    
    def train(self, X_train, y_train, X_val, y_val, num_classes):
        """Train classifier"""
        input_dim = X_train.shape[1]
        
        self.classifier = Sequential([
            Dense(256, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        self.classifier.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        history = self.classifier.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        self.training_history = history.history
        
        return history
    
    def predict(self, features):
        """Predict class from features"""
        if features is None or self.classifier is None:
            return -1
        pred = self.classifier.predict(np.expand_dims(features, axis=0), verbose=0)
        return np.argmax(pred)
    
    def save_classifier(self, path):
        """Save classifier and training history"""
        self.classifier.save(f"{path}_classifier.keras")
        
        if hasattr(self, 'training_history'):
            history_converted = {key: [float(val) for val in values] 
                                for key, values in self.training_history.items()}
            with open(f"{path}_history.json", 'w') as f:
                json.dump(history_converted, f, indent=2)


def plot_training_history(history_dict, model_name, save_path):
    """Plot training history and save to file"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name} Training History', fontsize=14, fontweight='bold')
    
    # Accuracy plot
    ax1.plot(history_dict['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history_dict['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('Model Accuracy', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history_dict['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history_dict['val_loss'], label='Val Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.set_title('Model Loss', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved training history plot: {save_path}")


def load_dataset(dataset_path):
    """Load dataset and split into train/val/test"""
    dataset_path = Path(dataset_path)
    
    class_folders = [f for f in dataset_path.iterdir() 
                    if f.is_dir() and 'incorrect' not in f.name.lower()]
    
    label_names = [f.name for f in class_folders]
    images = []
    labels = []
    
    print(f"Loading dataset from {dataset_path}...")
    for idx, class_folder in enumerate(class_folders):
        image_files = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.png'))
        print(f"  {class_folder.name}: {len(image_files)} images")
        for img_path in image_files:
            images.append(str(img_path))
            labels.append(idx)
    
    print(f"Total: {len(images)} images across {len(label_names)} classes")
    
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_names


def extract_features_dataset(model, image_paths, model_name):
    """Extract features for entire dataset"""
    print(f"\nExtracting {model_name} features from {len(image_paths)} images...")
    features = []
    times = []
    
    for i, img_path in enumerate(image_paths):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(image_paths)}")
        
        start = time.time()
        feat = model.extract_features(img_path)
        elapsed = time.time() - start
        times.append(elapsed)
        
        if feat is not None:
            features.append(feat)
        else:
            print(f"  Warning: Failed to extract features from {img_path}")
            features.append(np.zeros(51))  # Dummy features
    
    return np.array(features), times


def main():
    print("="*60)
    print("STEP 1: TRAIN MOVENET AND POSENET")
    print("="*60)
    
    # Load dataset
    X_train, X_val, X_test, y_train, y_val, y_test, label_names = load_dataset('dataset_multiclass_2')
    num_classes = len(label_names)
    
    split_data = {
        'X_test': X_test,
        'y_test': y_test,
        'label_names': label_names
    }
    with open('models/test_split.json', 'w') as f:
        json.dump(split_data, f, indent=2)
    
    print(f"\nDataset split:")
    print(f"  Train: {len(X_train)} images")
    print(f"  Val: {len(X_val)} images")
    print(f"  Test: {len(X_test)} images")
    print(f"  Classes: {num_classes}")
    
    print("\n" + "="*60)
    print("TRAINING MOVENET (THUNDER)")
    print("="*60)
    
    movenet = MoveNetModel(model_type='thunder')
    
    if os.path.exists('models/movenet_classifier.keras'):
        print("\n[INFO] MoveNet classifier already exists, loading...")
        from tf_keras.models import load_model
        movenet.classifier = load_model('models/movenet_classifier.keras')
        
        X_test_mn, test_times_mn = extract_features_dataset(movenet, X_test, 'MoveNet-Test')
        train_time_mn = 0
        history_mn = None
    else:
        X_train_mn, _ = extract_features_dataset(movenet, X_train, 'MoveNet-Train')
        X_val_mn, _ = extract_features_dataset(movenet, X_val, 'MoveNet-Val')
        X_test_mn, test_times_mn = extract_features_dataset(movenet, X_test, 'MoveNet-Test')
        
        print("\nTraining MoveNet classifier...")
        train_start = time.time()
        history_mn = movenet.train(X_train_mn, np.array(y_train), X_val_mn, np.array(y_val), num_classes)
        train_time_mn = time.time() - train_start
    
    print("\nEvaluating MoveNet...")
    y_pred_mn = [movenet.predict(feat) for feat in X_test_mn]
    acc_mn = accuracy_score(y_test, y_pred_mn)
    precision_mn, recall_mn, f1_mn, _ = precision_recall_fscore_support(y_test, y_pred_mn, average='weighted', zero_division=0)
    
    print(f"MoveNet Results:")
    print(f"  Accuracy: {acc_mn:.4f}")
    print(f"  Precision: {precision_mn:.4f}")
    print(f"  Recall: {recall_mn:.4f}")
    print(f"  F1-Score: {f1_mn:.4f}")
    print(f"  Avg inference time: {np.mean(test_times_mn)*1000:.2f} ms")
    print(f"  Training time: {train_time_mn/60:.2f} minutes")
    
    if history_mn is not None:
        movenet.save_classifier('models/movenet')
    elif not os.path.exists('models/movenet_history.json') and hasattr(movenet, 'training_history'):
        movenet.save_classifier('models/movenet')
    
    if hasattr(movenet, 'training_history'):
        plot_training_history(movenet.training_history, 'MoveNet Thunder', 'plots/movenet_training_history.png')
    
    movenet_results = {
        'accuracy': float(acc_mn),
        'precision': float(precision_mn),
        'recall': float(recall_mn),
        'f1_score': float(f1_mn),
        'avg_inference_time_ms': float(np.mean(test_times_mn) * 1000),
        'fps': float(1.0 / np.mean(test_times_mn)),
        'training_time_s': float(train_time_mn),
        'y_pred': [int(p) for p in y_pred_mn],
        'inference_times': [float(t) for t in test_times_mn]
    }
    
    with open('models/movenet_results.json', 'w') as f:
        json.dump(movenet_results, f, indent=2)
    
    del movenet
    gc.collect()
    tf.keras.backend.clear_session()
    print("\n✓ MoveNet training complete and saved")
    
    print("\n" + "="*60)
    print("TRAINING POSENET (LIGHTNING)")
    print("="*60)
    
    posenet = PoseNetModel()
    
    if os.path.exists('models/posenet_classifier.keras'):
        print("\n[INFO] PoseNet classifier already exists, loading...")
        from tf_keras.models import load_model
        posenet.classifier = load_model('models/posenet_classifier.keras')
        
        X_test_pn, test_times_pn = extract_features_dataset(posenet, X_test, 'PoseNet-Test')
        train_time_pn = 0
        history_pn = None
    else:
        # Extract features
        X_train_pn, _ = extract_features_dataset(posenet, X_train, 'PoseNet-Train')
        X_val_pn, _ = extract_features_dataset(posenet, X_val, 'PoseNet-Val')
        X_test_pn, test_times_pn = extract_features_dataset(posenet, X_test, 'PoseNet-Test')
        
        # Train
        print("\nTraining PoseNet classifier...")
        train_start = time.time()
        history_pn = posenet.train(X_train_pn, np.array(y_train), X_val_pn, np.array(y_val), num_classes)
        train_time_pn = time.time() - train_start
    
    # Evaluate
    print("\nEvaluating PoseNet...")
    y_pred_pn = [posenet.predict(feat) for feat in X_test_pn]
    acc_pn = accuracy_score(y_test, y_pred_pn)
    precision_pn, recall_pn, f1_pn, _ = precision_recall_fscore_support(y_test, y_pred_pn, average='weighted', zero_division=0)
    
    print(f"PoseNet Results:")
    print(f"  Accuracy: {acc_pn:.4f}")
    print(f"  Precision: {precision_pn:.4f}")
    print(f"  Recall: {recall_pn:.4f}")
    print(f"  F1-Score: {f1_pn:.4f}")
    print(f"  Avg inference time: {np.mean(test_times_pn)*1000:.2f} ms")
    print(f"  Training time: {train_time_pn/60:.2f} minutes")
    
    # Save (only if newly trained)
    if history_pn is not None:
        posenet.save_classifier('models/posenet')
    elif not os.path.exists('models/posenet_history.json') and hasattr(posenet, 'training_history'):
        # Save history if it's missing but we have it in memory
        posenet.save_classifier('models/posenet')
    
    # Plot training history
    if hasattr(posenet, 'training_history'):
        plot_training_history(posenet.training_history, 'PoseNet Lightning', 'plots/posenet_training_history.png')
    
    # Save results
    posenet_results = {
        'accuracy': float(acc_pn),
        'precision': float(precision_pn),
        'recall': float(recall_pn),
        'f1_score': float(f1_pn),
        'avg_inference_time_ms': float(np.mean(test_times_pn) * 1000),
        'fps': float(1.0 / np.mean(test_times_pn)),
        'training_time_s': float(train_time_pn),
        'y_pred': [int(p) for p in y_pred_pn],
        'inference_times': [float(t) for t in test_times_pn]
    }
    
    with open('models/posenet_results.json', 'w') as f:
        json.dump(posenet_results, f, indent=2)
    
    # Cleanup
    del posenet
    gc.collect()
    tf.keras.backend.clear_session()
    print("\n✓ PoseNet training complete and saved")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nSaved models:")
    print("  - models/movenet_classifier.keras")
    print("  - models/movenet_results.json")
    print("  - models/posenet_classifier.keras")
    print("  - models/posenet_results.json")
    print("  - models/test_split.json")


if __name__ == "__main__":
    main()
