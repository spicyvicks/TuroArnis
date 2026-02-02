#ensemble model classifier for inference only
#lightweight version for app deployment (does not include training utilities)

import os
import json
import joblib
import numpy as np

class EnsembleClassifier:
    def __init__(self, model_versions=None, voting='soft', weights=None, verbose=True, models_dir=None):
        self.voting = voting
        self.weights = weights
        self.verbose = verbose
        self.models = []
        self.model_info = []
        self.scaler = None
        self.label_encoder = None
        
        #models directory - must be provided for app deployment
        if models_dir is None:
            #try to find app/models
            current_dir = os.path.dirname(os.path.abspath(__file__))
            app_dir = os.path.dirname(current_dir)
            models_dir = os.path.join(app_dir, 'models')
        
        self.models_dir = models_dir
        
        if model_versions is None:
            raise ValueError("model_versions must be provided for app deployment")
        
        self._load_models(model_versions)
        
        #normalize weights
        if self.weights is not None:
            if len(self.weights) != len(self.models):
                raise ValueError(f"Number of weights ({len(self.weights)}) must match number of models ({len(self.models)})")
            self.weights = np.array(self.weights) / np.sum(self.weights)
        else:
            self.weights = np.ones(len(self.models)) / len(self.models)
    
    def _load_models(self, model_versions):
        if self.verbose:
            print(f"[info] loading ensemble models ({self.voting} voting)")
        
        for version_name in model_versions:
            version_path = os.path.join(self.models_dir, version_name)
            metadata_path = os.path.join(version_path, 'metadata.json')
            
            if not os.path.exists(metadata_path):
                if self.verbose:
                    print(f"[warn] skipping {version_name}: metadata.json not found")
                continue
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            model_type = metadata.get('model_type', 'dnn')
            
            #load model based on type
            if model_type == 'dnn':
                model_path = os.path.join(version_path, 'model.keras')
                if os.path.exists(model_path):
                    import tensorflow as tf
                    model = tf.keras.models.load_model(model_path)
                else:
                    if self.verbose:
                        print(f"[warn] skipping {version_name}: model.keras not found")
                    continue
            else:
                if model_type == 'random_forest':
                    model_path = os.path.join(version_path, 'model_rf.joblib')
                elif model_type == 'xgboost':
                    model_path = os.path.join(version_path, 'model_xgb.joblib')
                else:
                    model_path = os.path.join(version_path, 'model.joblib')
                
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                else:
                    if self.verbose:
                        print(f"[warn] skipping {version_name}: {os.path.basename(model_path)} not found")
                    continue
            
            #load scaler and encoder from first model
            if self.scaler is None:
                scaler_path = os.path.join(version_path, 'scaler.joblib')
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
            
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
                print(f"  loaded {version_name} ({model_type}) - accuracy: {acc:.2f}%")
        
        if self.verbose:
            print(f"[info] total models loaded: {len(self.models)}")
        
        if len(self.models) == 0:
            raise ValueError("No models could be loaded")
        
        if self.scaler is None or self.label_encoder is None:
            raise ValueError("Could not load scaler or label encoder")
    
    def predict_proba(self, X):
        #scale features
        X_scaled = self.scaler.transform(X)
        
        #collect probabilities from all models
        all_probas = []
        
        for i, model in enumerate(self.models):
            model_type = self.model_info[i]['type']
            
            if model_type == 'dnn':
                proba = model.predict(X_scaled, verbose=0)
            else:
                proba = model.predict_proba(X_scaled)
            
            all_probas.append(proba)
        
        #weight and average
        all_probas = np.array(all_probas)
        
        weighted_probas = np.zeros_like(all_probas[0])
        for i in range(len(self.models)):
            weighted_probas += self.weights[i] * all_probas[i]
        
        return weighted_probas
    
    def predict(self, X):
        if self.voting == 'soft':
            probas = self.predict_proba(X)
            predictions_encoded = np.argmax(probas, axis=1)
        else:
            #hard voting
            X_scaled = self.scaler.transform(X)
            
            all_predictions = []
            
            for i, model in enumerate(self.models):
                model_type = self.model_info[i]['type']
                
                if model_type == 'dnn':
                    proba = model.predict(X_scaled, verbose=0)
                    pred = np.argmax(proba, axis=1)
                else:
                    pred = model.predict(X_scaled)
                
                all_predictions.append(pred)
            
            all_predictions = np.array(all_predictions)
            
            predictions_encoded = []
            for sample_idx in range(all_predictions.shape[1]):
                votes = all_predictions[:, sample_idx]
                vote_counts = np.bincount(votes, weights=self.weights)
                predictions_encoded.append(np.argmax(vote_counts))
            
            predictions_encoded = np.array(predictions_encoded)
        
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        return predictions
