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
        
        # Apply STD clamping to template features (D2)
        # Wide STDs make Gaussian similarity non-discriminative
        ANGLE_FEATURES = {
            'left_elbow_angle', 'right_elbow_angle', 
            'left_shoulder_angle', 'right_shoulder_angle',
            'left_knee_angle', 'right_knee_angle'
        }
        
        for template_key, template in self.templates.items():
            for feature_name, feature_data in template.items():
                if isinstance(feature_data, dict) and 'std' in feature_data:
                    old_std = feature_data['std']
                    
                    # Determine max STD based on feature type
                    if feature_name in ANGLE_FEATURES:
                        max_std = 20.0  # degrees
                    else:
                        max_std = 0.1   # normalized coordinates
                    
                    if old_std > max_std:
                        print(f"[GCN-CLAMP] {template_key}.{feature_name}: std={old_std:.2f} clamped to {max_std}")
                        feature_data['std'] = max_std

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
                global_features: dict,
                skip_threshold: bool = False) -> Tuple[str, float, np.ndarray]:
        """
        Run GCN inference on extracted features.

        Args:
            skip_threshold: If True, return the raw top prediction without
                            applying the confidence threshold filter.  Used by
                            the evaluation tool only.

        Returns:
            predicted_class_name: str
            confidence: float (0-1)
            all_probabilities: np.ndarray (12 classes)
        """
        # Detect missing stick for confidence penalty (D3)
        stick_missing = np.isnan(stick_keypoints).any()
        
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
        best_variance = 0.0  # Track variance for dynamic threshold (D4)
        best_hybrid_features = None  # Store winning hybrid features
        
        # Iterate through all possible classes as "template hypotheses"
        # All classes in CLASS_NAMES have templates (neutral was removed to match training)
        candidate_classes = CLASS_NAMES[:]
        
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
                    best_hybrid_features = hybrid_features  # Store for variance calc
                    best_variance = np.var(hybrid_features)   # Track variance (D4)

        # ── FIX #1: Post-loop argmax verification ──────────────────────
        # The hypothesis loop picks the candidate whose self-consistent
        # probability is highest.  But the final_probs distribution for
        # that winning hypothesis may assign an *even higher* marginal
        # probability to a different class.  Verify and correct.
        if best_conf > 0 and len(final_probs) > 0:
            argmax_idx = int(np.argmax(final_probs))
            argmax_class = CLASS_NAMES[argmax_idx]
            argmax_prob = final_probs[argmax_idx]

            if argmax_class != best_class and argmax_prob > best_conf:
                print(f"[GCN-FIX1] Overriding hypothesis winner: "
                      f"{best_class}({best_conf:.4f}) → {argmax_class}({argmax_prob:.4f}) "
                      f"(argmax of final_probs)")
                best_class = argmax_class
                best_conf = argmax_prob
                # Recompute variance for argmax winner (D4)
                argmax_hybrid = compute_hybrid_features(
                    global_features, self.templates,
                    viewpoint=self.current_viewpoint,
                    class_name=argmax_class
                )
                best_variance = np.var(argmax_hybrid)
                best_hybrid_features = argmax_hybrid

        # ── D3: Confidence penalty for missing stick ───────────────────
        if stick_missing and best_conf > 0:
            PENALTY_FACTOR = 0.7
            original_conf = best_conf
            best_conf = best_conf * PENALTY_FACTOR
            print(f"[GCN-PENALTY] Stick missing: confidence {original_conf:.4f} → {best_conf:.4f} (×{PENALTY_FACTOR})")

        # ── D4: Dynamic threshold based on feature variance ─────────────
        # Low variance (pose matches multiple templates) → lower threshold
        base_threshold = self.config['models'].get(self.current_viewpoint, {}).get('confidence_threshold', 0.50)
        variance_factor = 1 - 0.3 * (1 - best_variance)
        effective_threshold = base_threshold * variance_factor
        effective_threshold = max(0.45, min(0.70, effective_threshold))  # Clamp [0.45, 0.70]
        print(f"[GCN-THRESH] Base: {base_threshold:.4f}, Variance: {best_variance:.4f}, Effective: {effective_threshold:.4f}")

        # ── DIAGNOSTIC: raw GCN output before threshold filtering ──
        top3_indices = np.argsort(final_probs)[::-1][:3]
        top3_info = [(CLASS_NAMES[i], f"{final_probs[i]:.3f}") for i in top3_indices]
        print(f"[GCN-RAW] best_class={best_class} | best_conf={best_conf:.4f} | "
              f"viewpoint={self.current_viewpoint} | top3={top3_info}")

        # Apply per-viewpoint confidence threshold (unless caller opts out)
        # Uses effective_threshold which may be lowered for similar-looking poses (D4)
        if not skip_threshold:
            if best_conf < effective_threshold:
                print(f"[GCN-THRESHOLD] REJECTED: {best_class} conf={best_conf:.4f} < effective_threshold={effective_threshold:.4f}")
                return "No Technique Detected", 0.0, final_probs
            else:
                print(f"[GCN-THRESHOLD] ACCEPTED: {best_class} conf={best_conf:.4f} >= effective_threshold={effective_threshold:.4f}")

        return best_class, best_conf, final_probs

    def predict_for_class(self, pose_keypoints: np.ndarray,
                          stick_keypoints: np.ndarray,
                          global_features: dict,
                          target_class: str) -> float:
        """
        Run a single GCN forward pass using the target class's template
        hypothesis and return its self-consistent probability.

        This answers: "How confident is the GCN that this pose IS [target_class]
        when compared against [target_class]'s reference template?"

        Returns:
            probability (0-1) for the target class
        """
        if target_class not in CLASS_NAMES:
            return 0.0

        node_features = extract_node_features(pose_keypoints, stick_keypoints)
        x = torch.tensor(node_features, dtype=torch.float32).to(self.device)
        batch = torch.zeros(35, dtype=torch.long).to(self.device)

        model = self.models.get(self.current_viewpoint)
        if model is None:
            if not self.models:
                return 0.0
            model = next(iter(self.models.values()))

        hybrid_features = compute_hybrid_features(
            global_features, self.templates,
            viewpoint=self.current_viewpoint,
            class_name=target_class
        )
        h = torch.tensor(hybrid_features, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = model(x, self.edge_index, batch, h)
            probs = torch.softmax(logits, dim=1)[0]

        target_idx = CLASS_NAMES.index(target_class)
        return probs[target_idx].item()

    def get_feature_corrections(
        self,
        global_features: dict,
        target_class: str,
    ) -> dict:
        """
        Compute hybrid similarity scores comparing the user's pose against the
        target class template and return data needed for actionable corrections.

        Returns:
            {
              'hybrid_scores':   np.ndarray of 30 similarity scores (0=worst, 1=best),
              'feature_names':   list of 30 feature names in same order,
              'raw_values':      dict feature_name → user's raw value,
              'template_means':  dict feature_name → ideal mean from template,
            }
            or None if the target class template is unavailable.
        """
        key = f"{self.current_viewpoint}_{target_class}"
        if self.templates is None or key not in self.templates:
            return None

        template = self.templates[key]
        feature_names = list(global_features.keys())
        hybrid_scores = compute_hybrid_features(
            global_features,
            self.templates,
            viewpoint=self.current_viewpoint,
            class_name=target_class,
        )
        template_means = {
            fname: template[fname]['mean']
            for fname in feature_names
            if fname in template
        }

        return {
            'hybrid_scores':  hybrid_scores,
            'feature_names':  feature_names,
            'raw_values':     dict(global_features),
            'template_means': template_means,
        }


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
