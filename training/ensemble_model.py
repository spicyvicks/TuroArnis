"""
Ensemble Model for TuroArnis Pose Classification
Combines multiple model architectures to improve prediction accuracy

ENSEMBLE METHODOLOGY:
An ensemble model combines predictions from multiple diverse models to achieve
better performance than any single model. The key principles are:

1. DIVERSITY: Use different model architectures (DNN, XGBoost)
   - Each learns different patterns in the data
   - Errors from different models are less correlated
   
2. VOTING STRATEGIES:
   a) Soft Voting (Weighted Averaging):
      - Each model outputs probability distributions
      - Average the probabilities with optional weights
      - Take class with highest averaged probability
      - Best when models output well-calibrated probabilities
   
   b) Hard Voting (Majority Vote):
      - Each model outputs single class prediction
      - Count votes for each class
      - Take class with most votes
      - More robust to poorly calibrated probabilities

3. WHY IT WORKS:
   - Reduces overfitting (averaging smooths out individual model biases)
   - Captures different aspects of data (DNN: complex patterns, XGB: gradient boosting)
   - More stable predictions (less sensitive to data variations)

TYPICAL ACCURACY GAINS: 2-10% improvement over best individual model
"""

import os
import sys
import json
import shutil
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

MODELS_DIR = os.path.join(project_root, 'models')


class EnsembleClassifier:
    """
    Ensemble classifier that combines multiple model types
    
    LOGIC:
    1. Load models from versioned directories
    2. For each prediction:
       - Get predictions from all models
       - Combine using selected voting strategy
       - Return final prediction
    """
    
    def __init__(self, model_versions=None, voting='soft', weights=None, verbose=True):
        """
        Initialize ensemble classifier
        
        Args:
            model_versions: List of version names (e.g. ['v017_ang3_xgb', 'v015_dnn'])
                           If None, will auto-select best DNN and XGBoost models
            voting: 'soft' (probability averaging) or 'hard' (majority vote)
            weights: List of weights for each model (only for soft voting)
                    If None, uses equal weights
            verbose: Print loading info
        """
        self.voting = voting
        self.weights = weights
        self.verbose = verbose
        self.models = []
        self.model_info = []
        self.scaler = None
        self.label_encoder = None
        
        # Load models
        if model_versions is None:
            model_versions = self._auto_select_models()
        
        self._load_models(model_versions)
        
        # Validate weights
        if self.weights is not None:
            if len(self.weights) != len(self.models):
                raise ValueError(f"Number of weights ({len(self.weights)}) must match number of models ({len(self.models)})")
            # Normalize weights
            self.weights = np.array(self.weights) / np.sum(self.weights)
        else:
            # Equal weights
            self.weights = np.ones(len(self.models)) / len(self.models)
    
    def _auto_select_models(self):
        """
        Auto-select best model of each type based on accuracy
        
        LOGIC:
        - Find all models grouped by type (random_forest, xgboost)
        - Select highest accuracy model from each type
        - Exclude DNN models
        - Returns list of version names
        """
        if not os.path.exists(MODELS_DIR):
            raise ValueError(f"Models directory not found: {MODELS_DIR}")
        
        # Group models by type (excluding DNN)
        model_groups = {
            'random_forest': [],
            'xgboost': []
        }
        
        for item in os.listdir(MODELS_DIR):
            item_path = os.path.join(MODELS_DIR, item)
            if os.path.isdir(item_path) and item.startswith('v'):
                metadata_path = os.path.join(item_path, 'metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    model_type = metadata.get('model_type', 'dnn')
                    accuracy = metadata.get('test_accuracy', 0)
                    
                    # Skip DNN models - only use RF and XGBoost
                    if model_type != 'dnn' and model_type in model_groups:
                        model_groups[model_type].append({
                            'name': item,
                            'accuracy': accuracy,
                            'metadata': metadata
                        })
        
        # Select best from each type
        selected = []
        for model_type, models in model_groups.items():
            if models:
                # Sort by accuracy descending
                models.sort(key=lambda x: x['accuracy'], reverse=True)
                best = models[0]
                selected.append(best['name'])
                if self.verbose:
                    print(f"[AUTO-SELECT] {model_type.upper()}: {best['name']} (Accuracy: {best['accuracy']*100:.2f}%)")
        
        if not selected:
            raise ValueError("No models found in models directory")
        
        return selected
    
    def _load_models(self, model_versions):
        """
        Load models, encoders, and scalers from version directories
        
        LOGIC:
        - For each version, load the appropriate model file
        - DNN models: .keras files (TensorFlow)
        - RF/XGBoost: .joblib files (scikit-learn)
        - Also load shared label encoder and scaler
        """
        print(f"\n{'='*60}")
        print(f"  LOADING ENSEMBLE MODELS ({self.voting.upper()} VOTING)")
        print(f"{'='*60}")
        
        for version_name in model_versions:
            version_path = os.path.join(MODELS_DIR, version_name)
            metadata_path = os.path.join(version_path, 'metadata.json')
            
            if not os.path.exists(metadata_path):
                print(f"[WARN] Skipping {version_name}: metadata.json not found")
                continue
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            model_type = metadata.get('model_type', 'dnn')
            
            # Load model based on type
            if model_type == 'dnn':
                model_path = os.path.join(version_path, 'model.keras')
                if os.path.exists(model_path):
                    # Import TensorFlow only if needed
                    import tensorflow as tf
                    model = tf.keras.models.load_model(model_path)
                else:
                    print(f"[WARN] Skipping {version_name}: model.keras not found")
                    continue
            else:
                # Random Forest or XGBoost (use type-specific filenames)
                if model_type == 'random_forest':
                    model_path = os.path.join(version_path, 'model_rf.joblib')
                elif model_type == 'xgboost':
                    model_path = os.path.join(version_path, 'model_xgb.joblib')
                else:
                    # Fallback for other types
                    model_path = os.path.join(version_path, 'model.joblib')
                
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                else:
                    print(f"[WARN] Skipping {version_name}: {os.path.basename(model_path)} not found")
                    continue
            
            # Load scaler (use first model's scaler)
            if self.scaler is None:
                scaler_path = os.path.join(version_path, 'scaler.joblib')
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
            
            # Load label encoder (use first model's encoder)
            if self.label_encoder is None:
                encoder_path = os.path.join(version_path, 'label_encoder.joblib')
                if os.path.exists(encoder_path):
                    self.label_encoder = joblib.load(encoder_path)
            
            self.models.append(model)
            self.model_info.append({
                'name': version_name,
                'type': model_type,
                'accuracy': metadata.get('test_accuracy', 0)
            })
            
            if self.verbose:
                acc = metadata.get('test_accuracy', 0) * 100
                print(f"  ✓ Loaded {version_name} ({model_type.upper()}) - Accuracy: {acc:.2f}%")
        
        print(f"{'='*60}")
        print(f"  Total models loaded: {len(self.models)}")
        print(f"{'='*60}\n")
        
        if len(self.models) == 0:
            raise ValueError("No models could be loaded")
        
        if self.scaler is None or self.label_encoder is None:
            raise ValueError("Could not load scaler or label encoder")
    
    def predict_proba(self, X):
        """
        Predict class probabilities using ensemble
        
        LOGIC (Soft Voting):
        1. Scale input features
        2. Get probability predictions from each model
        3. Weight and average probabilities
        4. Return averaged probability distribution
        
        Args:
            X: Feature array (n_samples, n_features)
        
        Returns:
            Averaged probability array (n_samples, n_classes)
        """
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Collect probabilities from all models
        all_probas = []
        
        for i, model in enumerate(self.models):
            model_type = self.model_info[i]['type']
            
            if model_type == 'dnn':
                # DNN outputs probabilities directly
                proba = model.predict(X_scaled, verbose=0)
            else:
                # Random Forest and XGBoost both have predict_proba method
                proba = model.predict_proba(X_scaled)
            
            all_probas.append(proba)
        
        # Weight and average
        all_probas = np.array(all_probas)  # Shape: (n_models, n_samples, n_classes)
        
        # Apply weights
        weighted_probas = np.zeros_like(all_probas[0])
        for i in range(len(self.models)):
            weighted_probas += self.weights[i] * all_probas[i]
        
        return weighted_probas
    
    def predict(self, X):
        """
        Predict class labels using ensemble
        
        LOGIC:
        - Soft voting: Use predict_proba and take argmax
        - Hard voting: Get predictions from each model and majority vote
        
        Args:
            X: Feature array (n_samples, n_features)
        
        Returns:
            Predicted class labels
        """
        if self.voting == 'soft':
            # Get averaged probabilities and take argmax
            probas = self.predict_proba(X)
            predictions_encoded = np.argmax(probas, axis=1)
        else:
            # Hard voting: majority vote
            X_scaled = self.scaler.transform(X)
            
            # Collect predictions from all models
            all_predictions = []
            
            for i, model in enumerate(self.models):
                model_type = self.model_info[i]['type']
                
                if model_type == 'dnn':
                    proba = model.predict(X_scaled, verbose=0)
                    pred = np.argmax(proba, axis=1)
                else:
                    pred = model.predict(X_scaled)
                
                all_predictions.append(pred)
            
            all_predictions = np.array(all_predictions)  # Shape: (n_models, n_samples)
            
            # Majority vote for each sample
            predictions_encoded = []
            for sample_idx in range(all_predictions.shape[1]):
                votes = all_predictions[:, sample_idx]
                # Count votes (using bincount)
                vote_counts = np.bincount(votes, weights=self.weights)
                predictions_encoded.append(np.argmax(vote_counts))
            
            predictions_encoded = np.array(predictions_encoded)
        
        # Decode to original labels
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        return predictions
    
    def evaluate(self, X, y):
        """
        Evaluate ensemble on test data
        
        Returns:
            accuracy, predictions, classification report
        """
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        report = classification_report(y, predictions)
        
        return accuracy, predictions, report
    
    def get_model_contributions(self, X):
        """
        Analyze individual model contributions to predictions
        
        Returns dict with per-model predictions and confidences
        """
        X_scaled = self.scaler.transform(X)
        contributions = []
        
        for i, model in enumerate(self.models):
            model_type = self.model_info[i]['type']
            
            if model_type == 'dnn':
                proba = model.predict(X_scaled, verbose=0)
            else:
                proba = model.predict(X_scaled)
            
            pred_encoded = np.argmax(proba, axis=1)
            pred_labels = self.label_encoder.inverse_transform(pred_encoded)
            confidence = np.max(proba, axis=1)
            
            contributions.append({
                'model': self.model_info[i]['name'],
                'type': model_type,
                'predictions': pred_labels,
                'confidence': confidence,
                'weight': self.weights[i]
            })
        
        return contributions


def evaluate_ensemble(csv_path=None, model_versions=None, voting='soft', weights=None):
    """
    Evaluate ensemble model on test data
    
    SCRIPT LOGIC:
    1. Load dataset (features + labels)
    2. Split into train/test
    3. Create ensemble classifier
    4. Evaluate on test set
    5. Compare with individual model performances
    6. Print detailed report
    
    Args:
        csv_path: Path to CSV with features (default: arnis_poses_angles.csv)
        model_versions: List of model versions to use (default: auto-select)
        voting: 'soft' or 'hard'
        weights: Model weights (default: equal)
    """
    # Default CSV path
    if csv_path is None:
        csv_path = os.path.join(project_root, 'arnis_poses_angles.csv')
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV file not found: {csv_path}")
        return
    
    print(f"\n[INFO] Loading dataset from: {os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path)
    
    # Extract features and labels
    X = df.drop('class', axis=1).values
    y = df['class'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"[INFO] Dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"[INFO] Test set: {len(X_test)} samples\n")
    
    # Create ensemble
    ensemble = EnsembleClassifier(
        model_versions=model_versions,
        voting=voting,
        weights=weights,
        verbose=True
    )
    
    # Evaluate ensemble
    print(f"\n{'='*60}")
    print(f"  ENSEMBLE EVALUATION")
    print(f"{'='*60}")
    
    accuracy, predictions, report = ensemble.evaluate(X_test, y_test)
    
    print(f"\n📊 ENSEMBLE ACCURACY: {accuracy*100:.2f}%")
    print(f"\nVoting Strategy: {voting.upper()}")
    if weights:
        print(f"Weights: {weights}")
    print(f"\n{'-'*60}")
    print("CLASSIFICATION REPORT:")
    print(f"{'-'*60}")
    print(report)
    
    # Compare with individual models
    print(f"\n{'='*60}")
    print(f"  INDIVIDUAL MODEL COMPARISON")
    print(f"{'='*60}")
    
    for i, info in enumerate(ensemble.model_info):
        model_acc = info['accuracy'] * 100
        improvement = (accuracy - info['accuracy']) * 100
        print(f"  {info['name']} ({info['type'].upper()})")
        print(f"    - Individual Accuracy: {model_acc:.2f}%")
        print(f"    - Ensemble Improvement: {improvement:+.2f}%")
        print()
    
    print(f"{'='*60}\n")
    
    return ensemble, accuracy


def optimize_ensemble_weights(csv_path=None, model_versions=None, voting='soft'):
    """
    Optimize ensemble weights using grid search
    
    LOGIC:
    1. Split data into train/val/test
    2. Try different weight combinations on validation set
    3. Select weights that give best validation accuracy
    4. Report test accuracy with optimal weights
    
    Args:
        csv_path: Path to CSV with features
        model_versions: List of model versions (default: auto-select)
        voting: 'soft' or 'hard'
    
    Returns:
        best_weights, best_accuracy, ensemble
    """
    # Default CSV path
    if csv_path is None:
        csv_path = os.path.join(project_root, 'arnis_poses_angles.csv')
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV file not found: {csv_path}")
        return None, 0, None
    
    print(f"\n{'='*60}")
    print(f"  OPTIMIZING ENSEMBLE WEIGHTS")
    print(f"{'='*60}\n")
    
    # Load data
    df = pd.read_csv(csv_path)
    X = df.drop('class', axis=1).values
    y = df['class'].values
    
    # Split: 60% train, 20% validation, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp  # 0.25 * 0.8 = 0.2
    )
    
    print(f"[INFO] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)} samples\n")
    
    # Create base ensemble to get model list
    base_ensemble = EnsembleClassifier(
        model_versions=model_versions,
        voting=voting,
        weights=None,
        verbose=True
    )
    
    num_models = len(base_ensemble.models)
    
    if num_models < 2:
        print("[ERROR] Need at least 2 models for ensemble")
        return None, 0, None
    
    # Grid search for weights
    print(f"\n{'='*60}")
    print(f"  GRID SEARCH ({num_models} models)")
    print(f"{'='*60}\n")
    
    best_weights = None
    best_val_acc = 0
    
    # Generate weight combinations
    from itertools import product
    
    if num_models == 2:
        # For 2 models: try weights from 0.1 to 0.9 in steps of 0.1
        weight_range = [i/10 for i in range(1, 10)]
        combinations = [(w, 1-w) for w in weight_range]
    else:
        # For 3+ models: coarser grid to avoid explosion
        weight_range = [0.2, 0.3, 0.4, 0.5]
        combinations = []
        for combo in product(weight_range, repeat=num_models-1):
            last_weight = 1.0 - sum(combo)
            if 0.1 <= last_weight <= 0.6:  # Last weight should also be reasonable
                combinations.append(combo + (last_weight,))
    
    print(f"[INFO] Testing {len(combinations)} weight combinations...\n")
    
    for weights in combinations:
        # Create ensemble with these weights
        ensemble = EnsembleClassifier(
            model_versions=model_versions,
            voting=voting,
            weights=list(weights),
            verbose=False
        )
        
        # Evaluate on validation set
        val_acc, _, _ = ensemble.evaluate(X_val, y_val)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = weights
    
    print(f"{'='*60}")
    print(f"  OPTIMIZATION RESULTS")
    print(f"{'='*60}\n")
    print(f"Best weights: {[f'{w:.2f}' for w in best_weights]}")
    print(f"Validation accuracy: {best_val_acc*100:.2f}%\n")
    
    # Test with optimal weights
    final_ensemble = EnsembleClassifier(
        model_versions=model_versions,
        voting=voting,
        weights=list(best_weights),
        verbose=False
    )
    
    test_acc, _, test_report = final_ensemble.evaluate(X_test, y_test)
    
    print(f"\n📊 TEST ACCURACY (optimized): {test_acc*100:.2f}%\n")
    print(f"{'='*60}\n")
    
    return list(best_weights), test_acc, final_ensemble


def save_ensemble_model(model_versions, weights, voting, accuracy, csv_path, models_dir, name_suffix=None):
    """
    Save ensemble configuration as a versioned model
    
    LOGIC:
    1. Create version directory (v{N}_ensemble_{suffix})
    2. Save ensemble_config.json with model list, weights, voting
    3. Save metadata.json (compatible with existing system)
    4. Copy scaler and label_encoder from component models
    
    Args:
        model_versions: List of model version names in ensemble
        weights: List of weights for each model
        voting: 'soft' or 'hard'
        accuracy: Test accuracy of ensemble
        csv_path: CSV used for training/evaluation
        models_dir: Directory to save ensemble
        name_suffix: Optional suffix for version name
    
    Returns:
        version_name, version_path
    """
    # Get next version number
    from model_manager import get_next_version_number
    version_num = get_next_version_number()
    
    # Create version name
    if name_suffix:
        version_name = f"v{version_num:03d}_ensemble_{name_suffix}"
    else:
        version_name = f"v{version_num:03d}_ensemble"
    
    version_path = os.path.join(models_dir, version_name)
    os.makedirs(version_path, exist_ok=True)
    
    # Save ensemble configuration
    ensemble_config = {
        'model_versions': model_versions,
        'weights': weights,
        'voting': voting,
        'created_at': datetime.now().isoformat()
    }
    
    config_path = os.path.join(version_path, 'ensemble_config.json')
    with open(config_path, 'w') as f:
        json.dump(ensemble_config, f, indent=2)
    
    # Save metadata (compatible with model manager)
    metadata = {
        'model_type': 'ensemble',
        'test_accuracy': accuracy,
        'trained_at': datetime.now().isoformat(),
        'num_classes': None,  # Will be filled from component model
        'train_samples': None,
        'csv_used': os.path.basename(csv_path),
        'component_models': model_versions,
        'voting_strategy': voting
    }
    
    # Copy scaler and encoder from first component model
    first_model_path = os.path.join(models_dir, model_versions[0])
    
    # Copy scaler
    src_scaler = os.path.join(first_model_path, 'scaler.joblib')
    dst_scaler = os.path.join(version_path, 'scaler.joblib')
    if os.path.exists(src_scaler):
        shutil.copy(src_scaler, dst_scaler)
    
    # Copy label encoder
    src_encoder = os.path.join(first_model_path, 'label_encoder.joblib')
    dst_encoder = os.path.join(version_path, 'label_encoder.joblib')
    if os.path.exists(src_encoder):
        shutil.copy(src_encoder, dst_encoder)
        
        # Get num_classes from encoder
        encoder = joblib.load(dst_encoder)
        metadata['num_classes'] = len(encoder.classes_)
    
    # Save metadata
    metadata_path = os.path.join(version_path, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"  ENSEMBLE MODEL SAVED")
    print(f"{'='*60}")
    print(f"  Version: {version_name}")
    print(f"  Path: {version_path}")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  Models: {', '.join([m.split('_')[0] for m in model_versions])}")
    print(f"{'='*60}\n")
    
    return version_name, version_path


def create_ensemble_model():
    """
    Interactive function to create and save an ensemble model
    
    WORKFLOW:
    1. Select component models
    2. Choose to optimize weights or use equal weights
    3. Evaluate ensemble
    4. Save ensemble configuration
    """
    print("\n" + "="*60)
    print("  CREATE ENSEMBLE MODEL")
    print("="*60)
    
    # Get available non-ensemble models
    available_models = []
    if os.path.exists(MODELS_DIR):
        for item in os.listdir(MODELS_DIR):
            item_path = os.path.join(MODELS_DIR, item)
            if os.path.isdir(item_path) and item.startswith('v'):
                metadata_path = os.path.join(item_path, 'metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Exclude existing ensembles and DNN models - only RF and XGBoost allowed
                    model_type = metadata.get('model_type', 'dnn')
                    if model_type != 'ensemble' and model_type != 'dnn':
                        available_models.append({
                            'name': item,
                            'type': model_type,
                            'accuracy': metadata.get('test_accuracy', 0)
                        })
    
    if len(available_models) < 2:
        print("[ERROR] Need at least 2 trained models (RF/XGBoost) to create ensemble")
        return
    
    # Sort by accuracy
    available_models.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print("\nAvailable models:")
    for i, m in enumerate(available_models, 1):
        print(f"  {i}. {m['name']} ({m['type'].upper()}) - {m['accuracy']*100:.2f}%")
    
    print("\nOptions:")
    print("  1. Auto-select best RF + XGBoost")
    print("  2. Manually select models")
    
    choice = input("\nEnter choice (1-2): ").strip()
    
    model_versions = None
    
    if choice == '2':
        # Manual selection
        indices = input("Enter model numbers separated by commas (e.g. 1,3): ").strip()
        try:
            indices = [int(x.strip()) - 1 for x in indices.split(',')]
            model_versions = [available_models[i]['name'] for i in indices if 0 <= i < len(available_models)]
            
            if len(model_versions) < 2:
                print("[ERROR] Must select at least 2 models")
                return
        except:
            print("[ERROR] Invalid input")
            return
    
    # Voting strategy
    voting = input("\nVoting strategy (soft/hard) [default=soft]: ").strip().lower()
    if voting not in ['soft', 'hard']:
        voting = 'soft'
    
    # CSV path
    csv_choice = input("\nUse angles or coordinates features? (angles/coordinates) [default=angles]: ").strip().lower()
    if csv_choice == 'coordinates':
        csv_path = os.path.join(project_root, 'arnis_poses_coordinates.csv')
    else:
        csv_path = os.path.join(project_root, 'arnis_poses_angles.csv')
    
    # Weight optimization
    optimize = input("\nOptimize weights using grid search? (y/n) [default=y]: ").strip().lower()
    
    if optimize != 'n':
        # Optimize weights
        weights, accuracy, ensemble = optimize_ensemble_weights(csv_path, model_versions, voting)
        if weights is None:
            print("[ERROR] Weight optimization failed")
            return
        # Extract model versions from ensemble
        model_versions = [info['name'] for info in ensemble.model_info]
    else:
        # Use equal weights
        ensemble, accuracy = evaluate_ensemble(csv_path, model_versions, voting, None)
        if ensemble is None:
            return
        weights = list(ensemble.weights)
        model_versions = [info['name'] for info in ensemble.model_info]
    
    # Ask for name suffix
    name_suffix = input("\nEnter name for this ensemble (or press Enter to skip): ").strip()
    if name_suffix:
        name_suffix = name_suffix.replace(' ', '_').replace('-', '_')
        name_suffix = ''.join(c for c in name_suffix if c.isalnum() or c == '_')
    else:
        name_suffix = None
    
    # Save ensemble
    version_name, version_path = save_ensemble_model(
        model_versions, weights, voting, accuracy, 
        csv_path, MODELS_DIR, name_suffix
    )
    
    # Ask if set as active
    set_active = input("\nSet this ensemble as active model? (y/n) [default=n]: ").strip().lower()
    if set_active == 'y':
        from model_manager import set_active_model
        set_active_model(version_name)


def interactive_ensemble():
    """
    Interactive menu for ensemble model
    """
    print("\n" + "="*60)
    print("  ENSEMBLE MODEL EVALUATION")
    print("="*60)
    
    # Get available models
    available_models = []
    if os.path.exists(MODELS_DIR):
        for item in os.listdir(MODELS_DIR):
            item_path = os.path.join(MODELS_DIR, item)
            if os.path.isdir(item_path) and item.startswith('v'):
                metadata_path = os.path.join(item_path, 'metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    available_models.append({
                        'name': item,
                        'type': metadata.get('model_type', 'dnn'),
                        'accuracy': metadata.get('test_accuracy', 0)
                    })
    
    if not available_models:
        print("[ERROR] No models found. Train models first.")
        return
    
    # Sort by accuracy
    available_models.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print("\nAvailable models:")
    for i, m in enumerate(available_models, 1):
        print(f"  {i}. {m['name']} ({m['type'].upper()}) - {m['accuracy']*100:.2f}%")
    
    print("\nOptions:")
    print("  1. Auto-select best model of each type")
    print("  2. Manually select models")
    
    choice = input("\nEnter choice (1-2): ").strip()
    
    model_versions = None
    weights = None
    
    if choice == '2':
        # Manual selection
        indices = input("Enter model numbers separated by commas (e.g. 1,3,5): ").strip()
        try:
            indices = [int(x.strip()) - 1 for x in indices.split(',')]
            model_versions = [available_models[i]['name'] for i in indices if 0 <= i < len(available_models)]
            
            # Ask for weights
            use_weights = input("Use custom weights? (y/n) [default=n]: ").strip().lower()
            if use_weights == 'y':
                weights_str = input(f"Enter {len(model_versions)} weights separated by commas: ").strip()
                weights = [float(x.strip()) for x in weights_str.split(',')]
        except:
            print("[ERROR] Invalid input. Using auto-select.")
            model_versions = None
    
    # Voting strategy
    voting = input("\nVoting strategy (soft/hard) [default=soft]: ").strip().lower()
    if voting not in ['soft', 'hard']:
        voting = 'soft'
    
    # CSV path
    csv_choice = input("\nUse angles or coordinates features? (angles/coordinates) [default=angles]: ").strip().lower()
    if csv_choice == 'coordinates':
        csv_path = os.path.join(project_root, 'arnis_poses_coordinates.csv')
    else:
        csv_path = os.path.join(project_root, 'arnis_poses_angles.csv')
    
    # Run evaluation
    evaluate_ensemble(csv_path, model_versions, voting, weights)


if __name__ == "__main__":
    interactive_ensemble()
