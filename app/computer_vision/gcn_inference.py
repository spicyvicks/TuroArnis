"""
GCN Inference Engine for Hybrid GCN V2 / V5 / V6 Models
Mixed-version loader: v2 left/right + v5 front (default)
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
    extract_node_features_v6,
    compute_hybrid_features,
    compute_hybrid_features_v6,
    compute_hybrid_features_v5,
    create_node_mask,
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
        
        # NOTE: STD clamping removed for V5 deployment compatibility.
        # The v5 model (hybrid_gcn_v5_front.pth) was trained and validated
        # against unclamped templates. Clamping STDs at inference creates
        # a distribution mismatch that drops accuracy on real test images.
        # (Validated: 5-sample test went from 1/5 correct to 3/5 correct
        #  when clamping was disabled to match deployment_package behavior.)

    def _load_models(self):
        """Load all specialist models with auto-detected version."""
        self.model_meta = {}  # viewpoint -> {'version': 'v2'|'v5'|'v6', ...}
        
        for viewpoint, model_info in self.config['models'].items():
            model_path = model_info['path']
            resolved_model_path = get_resource_path(model_path)
            print(f"[GCN] Loading {viewpoint} model from {resolved_model_path}...")
            
            if not os.path.exists(resolved_model_path):
                print(f"[ERROR] Model file not found: {resolved_model_path}")
                continue
                
            checkpoint = torch.load(resolved_model_path, map_location=self.device)
            
            # ── Version detection ──
            version = 'v2'
            cfg = checkpoint.get('config', {})
            if isinstance(cfg, dict):
                ver_str = cfg.get('version', '')
                if ver_str.startswith('v6'):
                    version = 'v6'
                    print(f"[GCN] Detected V6 deployment checkpoint "
                          f"(node={cfg.get('num_node_features')}, "
                          f"hybrid={cfg.get('num_hybrid_features')})")
                elif ver_str.startswith('v5'):
                    version = 'v5'
                    print(f"[GCN] Detected V5 deployment checkpoint "
                          f"(node={cfg.get('num_node_features')}, "
                          f"hybrid={cfg.get('num_hybrid_features')})")
            
            if version == 'v6':
                from app.models.gcn.model_v6 import HybridGCN as HybridGCN_v6, load_deployment_model
                model, class_names_v6, config_v6 = load_deployment_model(
                    resolved_model_path, device=self.device
                )
                self.models[viewpoint] = model
                self.model_meta[viewpoint] = {
                    'version': 'v6',
                    'config': config_v6,
                    'class_names': class_names_v6,
                }
                acc = checkpoint.get('config', {}).get('source_val_acc', 0)
                print(f"[GCN] Loaded V6 {viewpoint} model (val_acc: {acc:.2%})")
            elif version == 'v5':
                from app.models.gcn.model_v5 import load_deployment_model as load_v5
                model, class_names_v5, config_v5 = load_v5(
                    resolved_model_path, device=self.device
                )
                self.models[viewpoint] = model
                self.model_meta[viewpoint] = {
                    'version': 'v5',
                    'config': config_v5,
                    'class_names': class_names_v5,
                }
                acc = cfg.get('source_val_acc', 0)
                real_acc = cfg.get('real_only_test_acc', 0)
                print(f"[GCN] Loaded V5 {viewpoint} model (val_acc: {acc:.2%}, real_test: {real_acc:.2%})")
            else:
                # Legacy V2 checkpoint format
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
                self.model_meta[viewpoint] = {'version': 'v2'}
                print(f"[GCN] Loaded V2 {viewpoint} model "
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
        Auto-detects v2 / v5 / v6 model and uses appropriate pipeline.

        Args:
            skip_threshold: If True, return the raw top prediction without
                            applying the confidence threshold filter.

        Returns:
            predicted_class_name: str
            confidence: float (0-1)
            all_probabilities: np.ndarray (13 classes)
        """
        stick_missing = np.isnan(stick_keypoints).any()
        has_stick_detected = not stick_missing
        model = self.models.get(self.current_viewpoint)
        meta = self.model_meta.get(self.current_viewpoint, {'version': 'v2'})
        
        if model is None:
            if not self.models:
                return "Unknown", 0.0, np.zeros(len(CLASS_NAMES))
            model = next(iter(self.models.values()))
            meta = self.model_meta.get(self.current_viewpoint, {'version': 'v2'})
        
        # ── V6 PATH ──────────────────────────────────────────────────
        if meta['version'] == 'v6':
            return self._predict_v6(
                model, pose_keypoints, stick_keypoints, global_features,
                has_stick_detected, skip_threshold
            )

        # ── V5 PATH ──────────────────────────────────────────────────
        if meta['version'] == 'v5':
            return self._predict_v5(
                model, pose_keypoints, stick_keypoints, global_features,
                has_stick_detected, skip_threshold
            )

        # ── V2 PATH (legacy left/right) ─────────────────────────────
        return self._predict_v2(
            model, pose_keypoints, stick_keypoints, global_features,
            stick_missing, skip_threshold
        )
    
    def _predict_v6(self, model, pose_keypoints, stick_keypoints, global_features,
                    has_stick_detected, skip_threshold):
        """V6 inference: PyG Data/Batch with node_mask, 7-dim nodes, 49-dim hybrid."""
        from torch_geometric.data import Data, Batch
        
        # Shared node features & mask
        node_features = extract_node_features_v6(
            pose_keypoints, stick_keypoints, has_stick_detected=has_stick_detected
        )
        node_tensor = torch.from_numpy(node_features).float().to(self.device)
        node_mask_np = create_node_mask(has_stick_detected)
        node_mask_tensor = torch.from_numpy(node_mask_np).float().to(self.device)
        
        # Build per-class hybrid features
        candidate_classes = [c for c in CLASS_NAMES if c != 'neutral']
        hybrid_per_class = []
        for candidate in candidate_classes:
            hf = compute_hybrid_features_v6(
                global_features, self.templates,
                viewpoint=self.current_viewpoint,
                class_name=candidate
            )
            hybrid_per_class.append(hf)
        
        hybrid_stack = torch.from_numpy(np.stack(hybrid_per_class, axis=0)).float().to(self.device)
        
        # Build batched PyG Data objects (one per class hypothesis)
        graphs = []
        for i in range(len(candidate_classes)):
            graphs.append(Data(
                x=node_tensor,
                edge_index=self.edge_index.to(self.device),
                hybrid_features=hybrid_stack[i],
                y=torch.tensor([i], device=self.device),
                node_mask=node_mask_tensor
            ))
        
        best_class = "No Technique Detected"
        best_conf = 0.0
        final_probs = np.zeros(len(CLASS_NAMES))
        best_variance = 0.0
        best_hybrid_features = None
        
        with torch.no_grad():
            batch_obj = Batch.from_data_list(graphs)
            logits = model(batch_obj)
            probs_all = torch.softmax(logits, dim=-1)
            
            for i, candidate in enumerate(candidate_classes):
                candidate_idx = CLASS_NAMES.index(candidate)
                candidate_conf = probs_all[i, candidate_idx].item()
                
                if candidate_conf > best_conf:
                    best_conf = candidate_conf
                    best_class = candidate
                    final_probs = probs_all[i].cpu().numpy()
                    best_hybrid_features = hybrid_per_class[i]
                    best_variance = np.var(hybrid_per_class[i])
        
        # Post-processing: argmax verification, stick penalty, dynamic threshold
        return self._post_process(
            best_class, best_conf, final_probs, best_variance, best_hybrid_features,
            global_features, not has_stick_detected, skip_threshold
        )

    def _predict_v5(self, model, pose_keypoints, stick_keypoints, global_features,
                    has_stick_detected, skip_threshold):
        """V5 inference: PyG Data/Batch, 6-dim nodes, 46-dim hybrid, no node_mask."""
        from torch_geometric.data import Data, Batch

        # V5 node features: 6-dim [x, y, z, vis, dist_to_hip, angle_from_hip]
        # Uses origin fallback for missing stick (same as v2 extract_node_features)
        node_features = extract_node_features(pose_keypoints, stick_keypoints)
        node_tensor = torch.from_numpy(node_features).float().to(self.device)

        # Build per-class hybrid features (46-dim)
        candidate_classes = [c for c in CLASS_NAMES if c != 'neutral']
        hybrid_per_class = []
        for candidate in candidate_classes:
            hf = compute_hybrid_features_v5(
                global_features, self.templates,
                viewpoint=self.current_viewpoint,
                class_name=candidate
            )
            hybrid_per_class.append(hf)

        hybrid_stack = torch.from_numpy(np.stack(hybrid_per_class, axis=0)).float().to(self.device)

        # Build batched PyG Data objects (one per class hypothesis)
        graphs = []
        for i in range(len(candidate_classes)):
            graphs.append(Data(
                x=node_tensor,
                edge_index=self.edge_index.to(self.device),
                hybrid_features=hybrid_stack[i],
                y=torch.tensor([i], device=self.device)
            ))

        best_class = "No Technique Detected"
        best_conf = 0.0
        final_probs = np.zeros(len(CLASS_NAMES))
        best_variance = 0.0
        best_hybrid_features = None

        with torch.no_grad():
            batch_obj = Batch.from_data_list(graphs)
            logits = model(batch_obj)
            probs_all = torch.softmax(logits, dim=-1)

            for i, candidate in enumerate(candidate_classes):
                candidate_idx = CLASS_NAMES.index(candidate)
                candidate_conf = probs_all[i, candidate_idx].item()

                if candidate_conf > best_conf:
                    best_conf = candidate_conf
                    best_class = candidate
                    final_probs = probs_all[i].cpu().numpy()
                    best_hybrid_features = hybrid_per_class[i]
                    best_variance = np.var(hybrid_per_class[i])

        # Post-processing: argmax verification, stick penalty, dynamic threshold
        return self._post_process(
            best_class, best_conf, final_probs, best_variance, best_hybrid_features,
            global_features, not has_stick_detected, skip_threshold
        )

    def _predict_v2(self, model, pose_keypoints, stick_keypoints, global_features,
                    stick_missing, skip_threshold):
        """V2 inference: manual tensors, 6-dim nodes, ~30-dim hybrid."""
        node_features = extract_node_features(pose_keypoints, stick_keypoints)
        x = torch.tensor(node_features, dtype=torch.float32).to(self.device)
        batch = torch.zeros(35, dtype=torch.long).to(self.device)
        
        best_class = "No Technique Detected"
        best_conf = 0.0
        final_probs = np.zeros(len(CLASS_NAMES))
        best_variance = 0.0
        best_hybrid_features = None
        
        candidate_classes = [c for c in CLASS_NAMES if c != 'neutral']
        
        with torch.no_grad():
            for candidate in candidate_classes:
                hybrid_features = compute_hybrid_features(
                    global_features, self.templates,
                    viewpoint=self.current_viewpoint,
                    class_name=candidate
                )
                h = torch.tensor(hybrid_features, dtype=torch.float32).unsqueeze(0).to(self.device)
                logits = model(x, self.edge_index, batch, h)
                probs = torch.softmax(logits, dim=1)[0]
                
                candidate_idx = CLASS_NAMES.index(candidate)
                candidate_conf = probs[candidate_idx].item()
                
                if candidate_conf > best_conf:
                    best_conf = candidate_conf
                    best_class = candidate
                    final_probs = probs.cpu().numpy()
                    best_hybrid_features = hybrid_features
                    best_variance = np.var(hybrid_features)
        
        return self._post_process(
            best_class, best_conf, final_probs, best_variance, best_hybrid_features,
            global_features, stick_missing, skip_threshold
        )
    
    def _post_process(self, best_class, best_conf, final_probs, best_variance,
                      best_hybrid_features, global_features, stick_missing, skip_threshold):
        """Shared post-processing: stick penalty, dynamic threshold.
        NOTE: argmax verification removed for V5 deployment compatibility.
        The v5 model's self-consistent per-template probability is sufficient;
        argmax override was causing left/right misclassifications on real
        test images (validated against deployment_package behavior)."""

        # D3: Confidence penalty for missing stick
        if stick_missing and best_conf > 0:
            PENALTY_FACTOR = 0.7
            original_conf = best_conf
            best_conf = best_conf * PENALTY_FACTOR
            print(f"[GCN-PENALTY] Stick missing: confidence {original_conf:.4f} → {best_conf:.4f} (×{PENALTY_FACTOR})")

        # D4: Dynamic threshold based on feature variance
        base_threshold = self.config['models'].get(self.current_viewpoint, {}).get('confidence_threshold', 0.50)
        variance_factor = 1 - 0.3 * (1 - best_variance)
        effective_threshold = base_threshold * variance_factor
        effective_threshold = max(0.45, min(0.70, effective_threshold))
        print(f"[GCN-THRESH] Base: {base_threshold:.4f}, Variance: {best_variance:.4f}, Effective: {effective_threshold:.4f}")

        # DIAGNOSTIC
        top3_indices = np.argsort(final_probs)[::-1][:3]
        top3_info = [(CLASS_NAMES[i], f"{final_probs[i]:.3f}") for i in top3_indices]
        print(f"[GCN-RAW] best_class={best_class} | best_conf={best_conf:.4f} | "
              f"viewpoint={self.current_viewpoint} | top3={top3_info}")

        # Threshold filtering
        if not skip_threshold:
            if best_class == 'neutral' or best_conf < effective_threshold:
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
        Auto-detects v2 / v5 / v6 model.
        """
        if target_class not in CLASS_NAMES:
            return 0.0

        model = self.models.get(self.current_viewpoint)
        meta = self.model_meta.get(self.current_viewpoint, {'version': 'v2'})
        if model is None:
            if not self.models:
                return 0.0
            model = next(iter(self.models.values()))
            meta = self.model_meta.get(self.current_viewpoint, {'version': 'v2'})

        # ── V6 PATH ──
        if meta['version'] == 'v6':
            from torch_geometric.data import Data
            has_stick_detected = not np.isnan(stick_keypoints).any()
            node_features = extract_node_features_v6(
                pose_keypoints, stick_keypoints, has_stick_detected=has_stick_detected
            )
            node_tensor = torch.from_numpy(node_features).float().to(self.device)
            node_mask_np = create_node_mask(has_stick_detected)
            node_mask_tensor = torch.from_numpy(node_mask_np).float().to(self.device)

            hybrid_features = compute_hybrid_features_v6(
                global_features, self.templates,
                viewpoint=self.current_viewpoint,
                class_name=target_class
            )
            h = torch.from_numpy(hybrid_features).float().to(self.device)

            data = Data(
                x=node_tensor,
                edge_index=self.edge_index.to(self.device),
                hybrid_features=h,
                y=torch.tensor([0], device=self.device),
                node_mask=node_mask_tensor,
                batch=torch.zeros(35, dtype=torch.long, device=self.device)
            )

            with torch.no_grad():
                logits = model(data)
                probs = torch.softmax(logits, dim=-1)[0]

            target_idx = CLASS_NAMES.index(target_class)
            return probs[target_idx].item()

        # ── V5 PATH ──
        if meta['version'] == 'v5':
            from torch_geometric.data import Data
            node_features = extract_node_features(pose_keypoints, stick_keypoints)
            node_tensor = torch.from_numpy(node_features).float().to(self.device)

            hybrid_features = compute_hybrid_features_v5(
                global_features, self.templates,
                viewpoint=self.current_viewpoint,
                class_name=target_class
            )
            h = torch.from_numpy(hybrid_features).float().to(self.device)

            data = Data(
                x=node_tensor,
                edge_index=self.edge_index.to(self.device),
                hybrid_features=h,
                y=torch.tensor([0], device=self.device)
            )

            with torch.no_grad():
                logits = model(data)
                probs = torch.softmax(logits, dim=-1)[0]

            target_idx = CLASS_NAMES.index(target_class)
            return probs[target_idx].item()

        # ── V2 PATH ──
        node_features = extract_node_features(pose_keypoints, stick_keypoints)
        x = torch.tensor(node_features, dtype=torch.float32).to(self.device)
        batch = torch.zeros(35, dtype=torch.long).to(self.device)

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
        # FIX: align feature_names with hybrid_scores order.
        # compute_hybrid_features iterates template keys (v2/v6 compatible),
        # so feature_names must be template keys to match hybrid_scores indices.
        # Exclude metadata keys (e.g., _count, _acceptance_rate).
        feature_names = [k for k in template.keys() if not k.startswith('_')]
        hybrid_scores = compute_hybrid_features(
            global_features,
            self.templates,
            viewpoint=self.current_viewpoint,
            class_name=target_class,
        )
        template_means = {
            fname: template[fname]['mean']
            for fname in feature_names
        }

        return {
            'hybrid_scores':  hybrid_scores,
            'feature_names':  feature_names,
            'raw_values':     {k: v for k, v in global_features.items() if k in template and not k.startswith('_')},
            'template_means': template_means,
        }

    def capture_similarity_snapshot(
        self,
        pose_keypoints: np.ndarray,
        stick_keypoints: np.ndarray,
        global_features: dict,
        target_class: str,
        approach: str = 'variance'
    ) -> dict:
        """
        Capture similarity scores for lesson mode snapshot.
        
        Calculates how close the user's pose is to the target technique using
        the specified similarity approach, with psychological buffer applied
        for display purposes.
        
        Args:
            pose_keypoints: [33, 4] MediaPipe pose keypoints
            stick_keypoints: [2, 4] YOLO stick keypoints
            global_features: Dict of computed geometric features
            target_class: Target technique class name
            approach: 'simple', 'variance', or 'multifactor'
        
        Returns:
            {
                'actual_score': float,      # Real similarity (0-100)
                'display_score': float,   # With +5% buffer applied (0-100)
                'passed': bool,           # actual_score >= 65
                'low_features': list,     # Features < 70% similarity
                'approach': str,          # Which approach was used
                'category_scores': dict   # For multifactor approach
            }
        """
        # Import SimilarityCalculator
        from .lesson_feedback import SimilarityCalculator
        
        # Get hybrid feature corrections vs target template
        corrections = self.get_feature_corrections(global_features, target_class)
        if corrections is None:
            return {
                'actual_score': 0.0,
                'display_score': 0.0,
                'passed': False,
                'low_features': [],
                'approach': approach,
                'category_scores': {}
            }
        
        # Get template for variance-weighted approach
        template_key = f"{self.current_viewpoint}_{target_class}"
        template = self.templates.get(template_key, {}) if self.templates else {}
        
        # Calculate similarity
        calculator = SimilarityCalculator(self.templates)
        result = calculator.calculate_similarity(
            corrections['hybrid_scores'],
            corrections['feature_names'],
            template,
            approach=approach
        )
        
        actual_score = result['actual_score']
        
        # Identify low-scoring features for tips
        low_features = calculator.identify_low_features(
            corrections['hybrid_scores'],
            corrections['feature_names'],
            threshold=0.70
        )
        
        # Apply psychological buffer for display
        display_score = calculator.apply_psychological_buffer(actual_score)
        
        return {
            'actual_score': actual_score,
            'display_score': display_score,
            'passed': actual_score >= 65.0,
            'low_features': low_features,
            'approach': approach,
            'category_scores': result.get('category_scores', {})
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
