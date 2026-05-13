"""
GCN Inference Engine for Hybrid GCN V6 Models
Mixed-version loader replaced by per-viewpoint V6 deployment engines.

All three viewpoints (front, left, right) now use V6 models with:
- node_mask for masked pooling
- 7-dim node features (has_stick binary)
- 49 hybrid features (33 Gaussian + 15 signed + 1 has_stick)
- True zero-stick fallback (no origin hack)

Backwards compatibility: GCNInferenceEngine keeps the same public interface.
Internal implementation delegates to MultiViewpointEngine.
"""

import torch
import numpy as np
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import sys

sys.path.append(str(Path(__file__).parent.parent))

from app.utils.resource_path import get_resource_path
from app.deployment.viewpoint_engine import MultiViewpointEngine


class GCNInferenceEngine:
    """
    Backwards-compatible wrapper around MultiViewpointEngine.

    Keeps all public methods (predict, predict_for_class, set_viewpoint,
    get_feature_corrections, capture_similarity_snapshot) and exposes
    .config / .templates for existing callers in app.py and test scripts.
    """

    def __init__(self, config_path: str = None, device: str = "cpu"):
        self.device = torch.device(device)

        # Load config for backwards compatibility (app.py reads thresholds)
        if config_path is None:
            config_path = "app/models/gcn_model_config.json"
        resolved_config_path = get_resource_path(config_path)
        print(f"[GCN] Loading config from {resolved_config_path}...")
        with open(resolved_config_path, "r") as f:
            self.config = json.load(f)

        # Create the per-viewpoint multi-engine (loads all 3 V6 models)
        self._multi_engine = MultiViewpointEngine(device=device)

        # Merge templates from all viewpoints for test scripts
        self.templates = {}
        for engine in self._multi_engine.engines.values():
            self.templates.update(engine.templates)
        print(f"[GCN] Merged {len(self.templates)} total templates from all viewpoints")

    # ------------------------------------------------------------------
    # Public API (delegated to MultiViewpointEngine)
    # ------------------------------------------------------------------

    def set_viewpoint(self, viewpoint: str):
        """Switch active viewpoint instantly."""
        self._multi_engine.set_viewpoint(viewpoint)

    def predict(
        self,
        pose_keypoints: np.ndarray,
        stick_keypoints: np.ndarray,
        global_features: dict,
        skip_threshold: bool = False,
    ) -> Tuple[str, float, np.ndarray]:
        """
        Run GCN inference on extracted features.
        Returns: (predicted_class_name, confidence, all_probabilities)
        """
        return self._multi_engine.predict(
            pose_keypoints, stick_keypoints, global_features, skip_threshold
        )

    def predict_for_class(
        self,
        pose_keypoints: np.ndarray,
        stick_keypoints: np.ndarray,
        global_features: dict,
        target_class: str,
    ) -> float:
        """
        Run a single GCN forward pass using the target class's template
        hypothesis and return its self-consistent probability.
        """
        return self._multi_engine.predict_for_class(
            pose_keypoints, stick_keypoints, global_features, target_class
        )

    def get_feature_corrections(
        self, global_features: dict, target_class: str
    ) -> Optional[dict]:
        """
        Compute hybrid similarity scores comparing the user's pose against the
        target class template and return data needed for actionable corrections.
        """
        return self._multi_engine.get_feature_corrections(
            global_features, target_class
        )

    def capture_similarity_snapshot(
        self,
        pose_keypoints: np.ndarray,
        stick_keypoints: np.ndarray,
        global_features: dict,
        target_class: str,
        approach: str = "variance",
    ) -> dict:
        """
        Capture similarity scores for lesson mode snapshot.
        """
        return self._multi_engine.capture_similarity_snapshot(
            pose_keypoints,
            stick_keypoints,
            global_features,
            target_class,
            approach,
        )


# Global instance (lazy-loaded)
_gcn_engine: Optional[GCNInferenceEngine] = None


def get_gcn_engine(device: str = "cpu") -> GCNInferenceEngine:
    """Get or create global GCN inference engine"""
    global _gcn_engine
    if _gcn_engine is None:
        try:
            _gcn_engine = GCNInferenceEngine(device=device)
        except Exception as e:
            print(f"[ERROR] Failed to initialize GCN Engine: {e}")
            raise e
    return _gcn_engine
