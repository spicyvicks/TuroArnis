import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def generate_classification_report():
    print("\n[INFO] Starting classification report generation...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    model_path = os.path.join(project_root, 'models', 'arnis_coordinates_classifier.keras')
    encoder_path = os.path.join(project_root, 'models', 'label_encoder.joblib')
    data_path = os.path.join(project_root, 'arnis_poses_coordinates.csv')
    report_save_path = os.path.join(project_root, 'models', 'classification_report.txt')
    
    print("[INFO] Loading model and encoder...")
    model = tf.keras.models.load_model(model_path)
    label_encoder = joblib.load(encoder_path)
    
    print("[INFO] Loading test data...")
    df = pd.read_csv(data_path)
    
    X = df.drop('class', axis=1).values
    y_true = df['class'].values
    
    print("[INFO] Generating predictions...")
    y_pred_proba = model.predict(X, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    
    print("\n=== Classification Report ===\n")
    report = classification_report(y_true, y_pred_labels, zero_division=0)
    print(report)
    
    with open(report_save_path, 'w') as f:
        f.write("Classification Report for Arnis Pose Detection\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)
    
    print(f"\n[INFO] Report saved to: {report_save_path}")
    
    cm_save_path = os.path.join(project_root, 'models', 'confusion_matrix_detailed.png')
    plt.figure(figsize=(15, 12))
    
    classes = sorted(df['class'].unique())
    cm = confusion_matrix(y_true, y_pred_labels)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    
    plt.title('Confusion Matrix', pad=20)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    
    plt.tight_layout()
    
    # Save confusion matrix
    plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Confusion matrix saved to: {cm_save_path}")
    
    # Calculate and display some additional metrics
    print("\n=== Additional Metrics ===")
    print(f"Total samples: {len(y_true)}")
    print(f"Number of classes: {len(classes)}")
    
    # Calculate per-class accuracy
    class_correct = {}
    class_total = {}
    
    for true, pred in zip(y_true, y_pred_labels):
        if true not in class_total:
            class_total[true] = 0
            class_correct[true] = 0
        class_total[true] += 1
        if true == pred:
            class_correct[true] += 1
    
    print("\nPer-class sample count:")
    for class_name in sorted(class_total.keys()):
        total = class_total[class_name]
        correct = class_correct[class_name]
        accuracy = (correct / total) * 100
        print(f"{class_name:30s}: {total:4d} samples, Accuracy: {accuracy:6.2f}%")

if __name__ == "__main__":
    generate_classification_report()