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


class GCNInferenceEngine:
    """
    Manages loading and inference for 3 GCN specialist models.
    """

    def __init__(self, config_path: str = 'app/models/gcn_model_config.json',
                 device: str = 'cpu'):
        self.device = torch.device(device)
        self.models = {}
        self.current_viewpoint = 'front'
        self.templates = None
        self.edge_index = None

        self._load_config(config_path)
        self._load_models()
        self._prepare_graph_structure()

    def _load_config(self, config_path: str):
        """Load model configuration"""
        print(f"[GCN] Loading config from {config_path}...")
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        templates_path = self.config['feature_templates']
        print(f"[GCN] Loading templates from {templates_path}...")
        with open(templates_path, 'r') as f:
            self.templates = json.load(f)

    def _load_models(self):
        """Load all 3 specialist models"""
        for viewpoint, model_info in self.config['models'].items():
            model_path = model_info['path']
            print(f"[GCN] Loading {viewpoint} model from {model_path}...")
            
            if not os.path.exists(model_path):
                print(f"[ERROR] Model file not found: {model_path}")
                continue
                
            checkpoint = torch.load(model_path, map_location=self.device)

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
            all_probabilities: np.ndarray (13 classes)
        """
        # Extract node features [35, 6]
        # pose_keypoints: [33, 4] (x, y, z, visibility)
        # stick_keypoints: [2, 4] (x, y, z, visibility)
        node_features = extract_node_features(pose_keypoints, stick_keypoints)

        # Compute hybrid features [30]
        # Using neutral_stance as reference (as per plan/training setup)
        hybrid_features = compute_hybrid_features(
            global_features,
            self.templates,
            viewpoint=self.current_viewpoint,
            class_name='neutral_stance'
        )

        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float32).to(self.device)
        hybrid = torch.tensor(hybrid_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        batch = torch.zeros(35, dtype=torch.long).to(self.device)

        # Run inference
        model = self.models.get(self.current_viewpoint)
        if model is None:
            # Fallback to first available model if current viewpoint not loaded
            if not self.models:
                return "Unknown", 0.0, np.zeros(len(CLASS_NAMES))
            model = next(iter(self.models.values()))

        with torch.no_grad():
            logits = model(x, self.edge_index, batch, hybrid)
            probabilities = torch.softmax(logits, dim=1)[0]

        # Get prediction
        pred_idx = probabilities.argmax().item()
        confidence = probabilities[pred_idx].item()
        predicted_class = CLASS_NAMES[pred_idx]

        return predicted_class, confidence, probabilities.cpu().numpy()


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
