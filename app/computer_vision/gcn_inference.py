"""
GCN Inference Engine for Hybrid GCN V2 Models
Replaces the TensorFlow/sklearn model inference in pose_analyzer.py
"""

import torch
import numpy as np
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List

# Add project root to path to ensure local imports work
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.models.gcn.model_architecture import HybridGCN, SKELETON_EDGES, CLASS_NAMES
from app.models.gcn.feature_extraction import (
    extract_node_features,
    compute_hybrid_features,
    extract_raw_features
)
from app.utils.resource_path import get_resource_path


class GCNInferenceEngine:
    """
    Manages loading and inference for 3 GCN specialist models.
    """

    def __init__(self, config_path: str = None,
                 device: str = 'cpu'):
        self.device = torch.device(device)
        self.models = {}
        self.current_viewpoint = 'front'
        self.templates = None
        self.edge_index = None

        if config_path is None:
            config_path = 'app/models/gcn_model_config.json'
            
        self._load_config(config_path)
        self._load_models()
        self._prepare_graph_structure()

    def _load_config(self, config_path: str):
        """Load model configuration"""
        # Resolve config path
        resolved_config_path = get_resource_path(config_path)
        print(f"[GCN] Loading config from {resolved_config_path}...")
        
        with open(resolved_config_path, 'r') as f:
            self.config = json.load(f)

        # Resolve templates path
        templates_path = self.config['feature_templates']
        resolved_templates_path = get_resource_path(templates_path)
        print(f"[GCN] Loading templates from {resolved_templates_path}...")
        
        with open(resolved_templates_path, 'r') as f:
            self.templates = json.load(f)

    def _load_models(self):
        """Load all 3 specialist models"""
        for viewpoint, model_info in self.config['models'].items():
            model_path = model_info['path']
            # Resolve model path
            resolved_model_path = get_resource_path(model_path)
            print(f"[GCN] Loading {viewpoint} model from {resolved_model_path}...")
            
            if not os.path.exists(resolved_model_path):
                print(f"[ERROR] Model file not found: {resolved_model_path}")
                continue
                
            checkpoint = torch.load(resolved_model_path, map_location=self.device)

            model = HybridGCN(
                node_in_channels=checkpoint['node_feat_dim'],
                hybrid_in_channels=checkpoint['hybrid_feat_dim'],
                hidden_channels=checkpoint['hidden_dim'],
                num_classes=len(CLASS_NAMES),
                num_layers=checkpoint['num_layers'],
                dropout=checkpoint['dropout']
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()

            self.models[viewpoint] = model
            print(f"[GCN] Loaded {viewpoint} specialist model "
                  f"(accuracy: {checkpoint.get('test_accuracy', 0):.2%})")

    def _prepare_graph_structure(self):
        """Prepare edge index for graph convolution"""
        self.edge_index = torch.tensor(SKELETON_EDGES, dtype=torch.long).t().to(self.device)

    def set_viewpoint(self, viewpoint: str):
        """Switch active viewpoint model"""
        if viewpoint not in self.models:
            print(f"[WARN] Unknown viewpoint: {viewpoint}, keeping current: {self.current_viewpoint}")
            return
        self.current_viewpoint = viewpoint
        print(f"[GCN] Active viewpoint set to: {viewpoint}")

    def predict(self, pose_keypoints: np.ndarray,
                stick_keypoints: np.ndarray,
                global_features: dict) -> Tuple[str, float, np.ndarray]:
        """
        Run GCN inference on extracted features.

        Returns:
            predicted_class_name: str
            confidence: float (0-1)
            all_probabilities: np.ndarray (12 classes)
        """
        # Run inference for EACH template hypothesis
        # We don't know the ground truth, so we must test the user's pose against 
        # each template and see which one yields the highest self-consistent confidence.
        
        # Pre-compute node features (shared across all hypotheses)
        # [35, 6] -> [33 body + 2 stick, 6 features]
        node_features = extract_node_features(pose_keypoints, stick_keypoints)
        x = torch.tensor(node_features, dtype=torch.float32).to(self.device)
        batch = torch.zeros(35, dtype=torch.long).to(self.device)
        
        model = self.models.get(self.current_viewpoint)
        if model is None:
            if not self.models:
                return "Unknown", 0.0, np.zeros(len(CLASS_NAMES))
            model = next(iter(self.models.values()))
            
        best_class = "No Technique Detected"
        best_conf = 0.0
        final_probs = np.zeros(len(CLASS_NAMES))
        
        # Iterate through all possible classes as "template hypotheses"
        # We ignore 'neutral' as a template source because it has no fixed geometry
        # but we still allow the model to predict 'neutral' if no other template fits well.
        candidate_classes = [c for c in CLASS_NAMES if c != 'neutral']
        
        with torch.no_grad():
            for candidate in candidate_classes:
                # 1. Hypothesize: "User is trying to do [candidate]"
                # Compute hybrid features measuring deviation from [candidate] template
                hybrid_features = compute_hybrid_features(
                    global_features,
                    self.templates,
                    viewpoint=self.current_viewpoint,
                    class_name=candidate
                )
                
                h = torch.tensor(hybrid_features, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # 2. Ask Model: "Given this deviation from [candidate], what is the class?"
                logits = model(x, self.edge_index, batch, h)
                probs = torch.softmax(logits, dim=1)[0]
                
                # 3. Check consistency: Did the model predict [candidate] with high confidence?
                # We look specifically at the probability of the candidate class
                candidate_idx = CLASS_NAMES.index(candidate)
                candidate_conf = probs[candidate_idx].item()
                
                if candidate_conf > best_conf:
                    best_conf = candidate_conf
                    best_class = candidate
                    final_probs = probs.cpu().numpy()

        # Apply per-viewpoint confidence threshold
        threshold = self.config['models'].get(self.current_viewpoint, {}).get('confidence_threshold', 0.50)
        
        # Filter neutral predictions or low confidence
        if best_class == 'neutral' or best_conf < threshold:
            return "No Technique Detected", 0.0, final_probs

        return best_class, best_conf, final_probs


# Global instance (lazy-loaded)
_gcn_engine: Optional[GCNInferenceEngine] = None


def get_gcn_engine(device: str = 'cpu') -> GCNInferenceEngine:
    """Get or create global GCN inference engine"""
    global _gcn_engine
    if _gcn_engine is None:
        try:
            _gcn_engine = GCNInferenceEngine(device=device)
        except Exception as e:
            print(f"[ERROR] Failed to initialize GCN Engine: {e}")
            raise e
    return _gcn_engine
