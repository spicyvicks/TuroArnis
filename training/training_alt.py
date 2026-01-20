"""
Random Forest and XGBoost training for pose classification
Alternative architectures to DNN
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# optional: xgboost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[WARN] XGBoost not installed. Run: pip install xgboost")

# paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

def get_next_version(models_dir):
    """get next version number"""
    existing = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d)) and d.startswith('v')]
    if not existing:
        return 1
    versions = []
    for d in existing:
        try:
            versions.append(int(d.split('_')[0][1:]))
        except:
            pass
    return max(versions) + 1 if versions else 1

def train_random_forest(csv_path, models_dir):
    """train random forest classifier"""
    print("\n" + "="*50)
    print("  RANDOM FOREST TRAINING")
    print("="*50)
    
    # load data
    data = pd.read_csv(csv_path).dropna()
    X = data.iloc[:, 1:].values
    y_labels = data.iloc[:, 0].values
    
    # encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)
    class_names = list(label_encoder.classes_)
    
    print(f"\n  Classes: {len(class_names)}")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    
    # split data
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.11, random_state=42, stratify=y_temp)
    
    print(f"  Split: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
    
    # scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # hyperparameters (good defaults for pose classification)
    model = RandomForestClassifier(
        n_estimators=200,       # number of trees
        max_depth=20,           # prevent overfitting
        min_samples_split=5,    # minimum samples to split
        min_samples_leaf=2,     # minimum samples in leaf
        max_features='sqrt',    # features per split
        class_weight='balanced', # handle imbalanced classes
        random_state=42,
        n_jobs=-1               # use all cores
    )
    
    print("\n  Training Random Forest...")
    model.fit(X_train, y_train)
    
    # evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    print(f"\n  Train Accuracy: {train_acc*100:.2f}%")
    print(f"  Val Accuracy:   {val_acc*100:.2f}%")
    print(f"  Test Accuracy:  {test_acc*100:.2f}%")
    
    # save model
    version_num = get_next_version(models_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_name = f"v{version_num:03d}_{timestamp}"
    version_dir = os.path.join(models_dir, version_name)
    os.makedirs(version_dir, exist_ok=True)
    
    model_path = os.path.join(version_dir, 'model_rf.joblib')
    encoder_path = os.path.join(version_dir, 'label_encoder.joblib')
    scaler_path = os.path.join(version_dir, 'scaler.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)
    joblib.dump(scaler, scaler_path)
    
    # save metadata
    metadata = {
        'version': version_name,
        'model_type': 'random_forest',
        'trained_at': datetime.now().isoformat(),
        'test_accuracy': float(test_acc),
        'val_accuracy': float(val_acc),
        'num_classes': len(class_names),
        'num_features': X.shape[1],
        'train_samples': len(X_train),
        'class_names': class_names,
        'hyperparameters': {
            'n_estimators': 200,
            'max_depth': 20
        }
    }
    with open(os.path.join(version_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n  Saved to: {version_name}")
    
    # update active model if better
    update_active_model(models_dir, version_name, version_dir, model_path, encoder_path, scaler_path, test_acc, 'random_forest')
    
    return test_acc, version_name


def train_xgboost(csv_path, models_dir):
    """train xgboost classifier"""
    if not HAS_XGBOOST:
        print("\n[ERROR] XGBoost not installed. Run: pip install xgboost")
        return None, None
    
    print("\n" + "="*50)
    print("  XGBOOST TRAINING")
    print("="*50)
    
    # load data
    data = pd.read_csv(csv_path).dropna()
    X = data.iloc[:, 1:].values
    y_labels = data.iloc[:, 0].values
    
    # encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)
    class_names = list(label_encoder.classes_)
    
    print(f"\n  Classes: {len(class_names)}")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    
    # split data
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.11, random_state=42, stratify=y_temp)
    
    print(f"  Split: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
    
    # scale features (XGBoost doesn't need scaling but helps consistency)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # hyperparameters (good defaults for pose classification)
    model = xgb.XGBClassifier(
        n_estimators=200,       # number of boosting rounds
        max_depth=6,            # tree depth (lower = less overfit)
        learning_rate=0.1,      # step size
        subsample=0.8,          # sample ratio per tree
        colsample_bytree=0.8,   # feature ratio per tree
        min_child_weight=3,     # minimum sum of instance weight
        gamma=0.1,              # minimum loss reduction
        reg_alpha=0.1,          # L1 regularization
        reg_lambda=1.0,         # L2 regularization
        objective='multi:softmax',
        num_class=len(class_names),
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    print("\n  Training XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    print(f"\n  Train Accuracy: {train_acc*100:.2f}%")
    print(f"  Val Accuracy:   {val_acc*100:.2f}%")
    print(f"  Test Accuracy:  {test_acc*100:.2f}%")
    
    # save model
    version_num = get_next_version(models_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_name = f"v{version_num:03d}_{timestamp}"
    version_dir = os.path.join(models_dir, version_name)
    os.makedirs(version_dir, exist_ok=True)
    
    model_path = os.path.join(version_dir, 'model_xgb.joblib')
    encoder_path = os.path.join(version_dir, 'label_encoder.joblib')
    scaler_path = os.path.join(version_dir, 'scaler.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)
    joblib.dump(scaler, scaler_path)
    
    # save metadata
    metadata = {
        'version': version_name,
        'model_type': 'xgboost',
        'trained_at': datetime.now().isoformat(),
        'test_accuracy': float(test_acc),
        'val_accuracy': float(val_acc),
        'num_classes': len(class_names),
        'num_features': X.shape[1],
        'train_samples': len(X_train),
        'class_names': class_names,
        'hyperparameters': {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1
        }
    }
    with open(os.path.join(version_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n  Saved to: {version_name}")
    
    # update active model if better
    update_active_model(models_dir, version_name, version_dir, model_path, encoder_path, scaler_path, test_acc, 'xgboost')
    
    return test_acc, version_name


def update_active_model(models_dir, version_name, version_dir, model_path, encoder_path, scaler_path, test_acc, model_type):
    """update active model only if accuracy is higher"""
    active_model_path = os.path.join(models_dir, 'active_model.json')
    
    should_set_active = True
    if os.path.exists(active_model_path):
        with open(active_model_path, 'r') as f:
            current_active = json.load(f)
        current_metadata_path = os.path.join(current_active['path'], 'metadata.json')
        if os.path.exists(current_metadata_path):
            with open(current_metadata_path, 'r') as f:
                current_metadata = json.load(f)
            current_acc = current_metadata.get('test_accuracy', 0)
            if test_acc <= current_acc:
                should_set_active = False
                print(f"\n  New accuracy ({test_acc*100:.2f}%) <= current ({current_acc*100:.2f}%)")
                print(f"  Keeping {current_active['version']} as active")
    
    if should_set_active:
        active_config = {
            'version': version_name,
            'path': version_dir,
            'model_path': model_path,
            'model_type': model_type,
            'encoder_path': encoder_path,
            'scaler_path': scaler_path,
            'test_accuracy': float(test_acc),
            'set_at': datetime.now().isoformat()
        }
        with open(active_model_path, 'w') as f:
            json.dump(active_config, f, indent=2)
        print(f"\n  [OK] Set as active model ({test_acc*100:.2f}%)")


if __name__ == "__main__":
    models_dir = os.path.join(project_root, 'models')
    csv_path = os.path.join(project_root, 'arnis_poses_angles.csv')
    
    os.makedirs(models_dir, exist_ok=True)
    
    print("\nSelect model to train:")
    print("  1. Random Forest")
    print("  2. XGBoost")
    
    choice = input("\nChoice: ").strip()
    
    if choice == '1':
        train_random_forest(csv_path, models_dir)
    elif choice == '2':
        train_xgboost(csv_path, models_dir)
    else:
        print("Invalid choice")
