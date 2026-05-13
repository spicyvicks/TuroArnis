"""
Per-viewpoint deployment inference engines (V5 + V6 auto-detect).
Each viewpoint is self-contained: own model weights + templates.
Shared architecture/feature extraction code is imported from app.models.gcn
for fastest inference (no duplication, minimal memory).
"""

import torch
import numpy as np
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import sys

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.gcn.model_v6 import HybridGCN, SKELETON_EDGES
from app.models.gcn.model_v5 import load_deployment_model as load_v5_model
from app.models.gcn.model_architecture import CLASS_NAMES
from app.models.gcn.feature_extraction import (
    compute_hybrid_features_v6,
    compute_hybrid_features_v5,
    extract_node_features_v6,
    extract_node_features,
    create_node_mask,
    compute_hybrid_features,
)
from app.utils.resource_path import get_resource_path


class ViewpointInferenceEngine:
    """
    Self-contained inference engine for a single viewpoint.
    Auto-detects V5 vs V6 from checkpoint and uses correct pipeline.
    """

    def __init__(
        self,
        viewpoint: str,
        model_path: str,
        templates_path: str,
        device: str = "cpu",
        confidence_threshold: float = 0.70,
    ):
        self.viewpoint = viewpoint
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold

        # Detect version from checkpoint
        ckpt = torch.load(model_path, map_location=device)
        cfg = ckpt.get("config", {})
        ver_str = cfg.get("version", "")
        if ver_str.startswith("v6") or cfg.get("num_node_features") == 7:
            self.version = "v6"
        elif ver_str.startswith("v5") or cfg.get("num_hybrid_features") == 46:
            self.version = "v5"
        else:
            # Fallback: inspect model architecture
            state = ckpt["model_state_dict"]
            if "hybrid_mlp.0.weight" in state:
                hybrid_dim = state["hybrid_mlp.0.weight"].shape[1]
                self.version = "v6" if hybrid_dim >= 49 else "v5"
            else:
                self.version = "v5"
        print(
            f"[ViewpointEngine] Detected {self.version.upper()} model for {viewpoint}"
        )

        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        if self.version == "v6":
            from app.models.gcn.model_v6 import load_deployment_model

            self.model, self.class_names, self.config = load_deployment_model(
                model_path, device=self.device
            )
        else:
            self.model, self.class_names, self.config = load_v5_model(
                model_path, device=self.device
            )
        print(
            f"[ViewpointEngine] Loaded {viewpoint} {self.version.upper()} model from {model_path}"
        )

        # Load templates
        if not os.path.exists(templates_path):
            raise FileNotFoundError(f"Templates not found: {templates_path}")
        with open(templates_path, "r") as f:
            self.templates = json.load(f)
        print(
            f"[ViewpointEngine] Loaded {len(self.templates)} templates for {viewpoint}"
        )

        # Prepare graph structure
        self.edge_index = (
            torch.tensor(SKELETON_EDGES, dtype=torch.long).t().to(self.device)
        )

    def predict(
        self,
        pose_keypoints: np.ndarray,
        stick_keypoints: np.ndarray,
        global_features: dict,
        skip_threshold: bool = False,
    ) -> Tuple[str, float, np.ndarray]:
        """Run inference on extracted features."""
        if self.version == "v6":
            return self._predict_v6(
                pose_keypoints, stick_keypoints, global_features, skip_threshold
            )
        else:
            return self._predict_v5(
                pose_keypoints, stick_keypoints, global_features, skip_threshold
            )

    def _predict_v6(
        self, pose_keypoints, stick_keypoints, global_features, skip_threshold
    ):
        from torch_geometric.data import Data, Batch

        has_stick_detected = not np.isnan(stick_keypoints).any()
        node_features = extract_node_features_v6(
            pose_keypoints, stick_keypoints, has_stick_detected=has_stick_detected
        )
        node_tensor = torch.from_numpy(node_features).float().to(self.device)
        node_mask_np = create_node_mask(has_stick_detected)
        node_mask_tensor = torch.from_numpy(node_mask_np).float().to(self.device)

        candidate_classes = [c for c in CLASS_NAMES if c != "neutral"]
        hybrid_per_class = []
        for candidate in candidate_classes:
            hf = compute_hybrid_features_v6(
                global_features,
                self.templates,
                viewpoint=self.viewpoint,
                class_name=candidate,
            )
            hybrid_per_class.append(hf)

        hybrid_stack = (
            torch.from_numpy(np.stack(hybrid_per_class, axis=0))
            .float()
            .to(self.device)
        )

        graphs = []
        for i in range(len(candidate_classes)):
            graphs.append(
                Data(
                    x=node_tensor,
                    edge_index=self.edge_index.to(self.device),
                    hybrid_features=hybrid_stack[i],
                    y=torch.tensor([i], device=self.device),
                    node_mask=node_mask_tensor,
                )
            )

        best_class, best_conf, final_probs, best_variance, best_hybrid = (
            "No Technique Detected",
            0.0,
            np.zeros(len(CLASS_NAMES)),
            0.0,
            None,
        )

        with torch.no_grad():
            batch_obj = Batch.from_data_list(graphs)
            logits = self.model(batch_obj)
            probs_all = torch.softmax(logits, dim=-1)
            for i, candidate in enumerate(candidate_classes):
                candidate_idx = CLASS_NAMES.index(candidate)
                candidate_conf = probs_all[i, candidate_idx].item()
                if candidate_conf > best_conf:
                    best_conf = candidate_conf
                    best_class = candidate
                    final_probs = probs_all[i].cpu().numpy()
                    best_hybrid = hybrid_per_class[i]
                    best_variance = np.var(hybrid_per_class[i])

        return self._post_process(
            best_class,
            best_conf,
            final_probs,
            best_variance,
            best_hybrid,
            global_features,
            not has_stick_detected,
            skip_threshold,
        )

    def _predict_v5(
        self, pose_keypoints, stick_keypoints, global_features, skip_threshold
    ):
        from torch_geometric.data import Data, Batch

        has_stick_detected = not np.isnan(stick_keypoints).any()
        node_features = extract_node_features(pose_keypoints, stick_keypoints)
        node_tensor = torch.from_numpy(node_features).float().to(self.device)

        candidate_classes = [c for c in CLASS_NAMES if c != "neutral"]
        hybrid_per_class = []
        for candidate in candidate_classes:
            hf = compute_hybrid_features_v5(
                global_features,
                self.templates,
                viewpoint=self.viewpoint,
                class_name=candidate,
            )
            hybrid_per_class.append(hf)

        hybrid_stack = (
            torch.from_numpy(np.stack(hybrid_per_class, axis=0))
            .float()
            .to(self.device)
        )

        graphs = []
        for i in range(len(candidate_classes)):
            graphs.append(
                Data(
                    x=node_tensor,
                    edge_index=self.edge_index.to(self.device),
                    hybrid_features=hybrid_stack[i],
                    y=torch.tensor([i], device=self.device),
                )
            )

        best_class, best_conf, final_probs, best_variance, best_hybrid = (
            "No Technique Detected",
            0.0,
            np.zeros(len(CLASS_NAMES)),
            0.0,
            None,
        )

        with torch.no_grad():
            batch_obj = Batch.from_data_list(graphs)
            logits = self.model(batch_obj)
            probs_all = torch.softmax(logits, dim=-1)
            for i, candidate in enumerate(candidate_classes):
                candidate_idx = CLASS_NAMES.index(candidate)
                candidate_conf = probs_all[i, candidate_idx].item()
                if candidate_conf > best_conf:
                    best_conf = candidate_conf
                    best_class = candidate
                    final_probs = probs_all[i].cpu().numpy()
                    best_hybrid = hybrid_per_class[i]
                    best_variance = np.var(hybrid_per_class[i])

        return self._post_process(
            best_class,
            best_conf,
            final_probs,
            best_variance,
            best_hybrid,
            global_features,
            not has_stick_detected,
            skip_threshold,
        )

    def predict_for_class(
        self,
        pose_keypoints: np.ndarray,
        stick_keypoints: np.ndarray,
        global_features: dict,
        target_class: str,
    ) -> float:
        """Run single-class inference."""
        if target_class not in CLASS_NAMES:
            return 0.0

        from torch_geometric.data import Data

        if self.version == "v6":
            has_stick_detected = not np.isnan(stick_keypoints).any()
            node_features = extract_node_features_v6(
                pose_keypoints, stick_keypoints, has_stick_detected=has_stick_detected
            )
            node_tensor = torch.from_numpy(node_features).float().to(self.device)
            node_mask_np = create_node_mask(has_stick_detected)
            node_mask_tensor = torch.from_numpy(node_mask_np).float().to(self.device)
            hybrid_features = compute_hybrid_features_v6(
                global_features,
                self.templates,
                viewpoint=self.viewpoint,
                class_name=target_class,
            )
            h = torch.from_numpy(hybrid_features).float().to(self.device)
            data = Data(
                x=node_tensor,
                edge_index=self.edge_index.to(self.device),
                hybrid_features=h,
                y=torch.tensor([0], device=self.device),
                node_mask=node_mask_tensor,
                batch=torch.zeros(35, dtype=torch.long, device=self.device),
            )
        else:
            node_features = extract_node_features(pose_keypoints, stick_keypoints)
            node_tensor = torch.from_numpy(node_features).float().to(self.device)
            hybrid_features = compute_hybrid_features_v5(
                global_features,
                self.templates,
                viewpoint=self.viewpoint,
                class_name=target_class,
            )
            h = torch.from_numpy(hybrid_features).float().to(self.device)
            data = Data(
                x=node_tensor,
                edge_index=self.edge_index.to(self.device),
                hybrid_features=h,
                y=torch.tensor([0], device=self.device),
            )

        with torch.no_grad():
            logits = self.model(data)
            probs = torch.softmax(logits, dim=-1)[0]

        target_idx = CLASS_NAMES.index(target_class)
        return probs[target_idx].item()

    def get_feature_corrections(
        self, global_features: dict, target_class: str
    ) -> Optional[dict]:
        """Compute hybrid similarity scores."""
        key = f"{self.viewpoint}_{target_class}"
        if self.templates is None or key not in self.templates:
            return None

        template = self.templates[key]
        feature_names = [k for k in template.keys() if not k.startswith("_")]
        hybrid_scores = compute_hybrid_features(
            global_features,
            self.templates,
            viewpoint=self.viewpoint,
            class_name=target_class,
        )
        template_means = {
            fname: template[fname]["mean"] for fname in feature_names
        }

        return {
            "hybrid_scores": hybrid_scores,
            "feature_names": feature_names,
            "raw_values": {
                k: v
                for k, v in global_features.items()
                if k in template and not k.startswith("_")
            },
            "template_means": template_means,
        }

    def capture_similarity_snapshot(
        self,
        pose_keypoints: np.ndarray,
        stick_keypoints: np.ndarray,
        global_features: dict,
        target_class: str,
        approach: str = "variance",
    ) -> dict:
        """Capture similarity scores for lesson mode snapshot."""
        from app.computer_vision.lesson_feedback import SimilarityCalculator

        key = f"{self.viewpoint}_{target_class}"
        template = self.templates.get(key, {}) if self.templates else {}

        corrections = self.get_feature_corrections(global_features, target_class)
        if corrections is None:
            return {
                "actual_score": 0.0,
                "display_score": 0.0,
                "passed": False,
                "low_features": [],
                "approach": approach,
                "category_scores": {},
            }

        calculator = SimilarityCalculator(self.templates)
        result = calculator.calculate_similarity(
            corrections["hybrid_scores"],
            corrections["feature_names"],
            template,
            approach=approach,
        )

        actual_score = result["actual_score"]
        low_features = calculator.identify_low_features(
            corrections["hybrid_scores"],
            corrections["feature_names"],
            threshold=0.70,
        )
        display_score = calculator.apply_psychological_buffer(actual_score)

        return {
            "actual_score": actual_score,
            "display_score": display_score,
            "passed": actual_score >= 65.0,
            "low_features": low_features,
            "approach": approach,
            "category_scores": result.get("category_scores", {}),
        }

    def _post_process(
        self,
        best_class,
        best_conf,
        final_probs,
        best_variance,
        best_hybrid_features,
        global_features,
        stick_missing,
        skip_threshold,
    ):
        """Shared post-processing: stick penalty, dynamic threshold."""
        if stick_missing and best_conf > 0:
            PENALTY_FACTOR = 0.7
            original_conf = best_conf
            best_conf = best_conf * PENALTY_FACTOR
            print(
                f"[ViewpointEngine-{self.viewpoint}] Stick missing: "
                f"confidence {original_conf:.4f} -> {best_conf:.4f} (x{PENALTY_FACTOR})"
            )

        variance_factor = 1 - 0.3 * (1 - best_variance)
        effective_threshold = self.confidence_threshold * variance_factor
        effective_threshold = max(0.45, min(0.70, effective_threshold))
        print(
            f"[ViewpointEngine-{self.viewpoint}] Base: {self.confidence_threshold:.4f}, "
            f"Variance: {best_variance:.4f}, Effective: {effective_threshold:.4f}"
        )

        top3_indices = np.argsort(final_probs)[::-1][:3]
        top3_info = [
            (CLASS_NAMES[i], f"{final_probs[i]:.3f}") for i in top3_indices
        ]
        print(
            f"[ViewpointEngine-{self.viewpoint}] best_class={best_class} | "
            f"best_conf={best_conf:.4f} | top3={top3_info}"
        )

        if not skip_threshold:
            if best_class == "neutral" or best_conf < effective_threshold:
                print(
                    f"[ViewpointEngine-{self.viewpoint}] REJECTED: {best_class} "
                    f"conf={best_conf:.4f} < effective={effective_threshold:.4f}"
                )
                return "No Technique Detected", 0.0, final_probs
            else:
                print(
                    f"[ViewpointEngine-{self.viewpoint}] ACCEPTED: {best_class} "
                    f"conf={best_conf:.4f} >= effective={effective_threshold:.4f}"
                )

        return best_class, best_conf, final_probs


class MultiViewpointEngine:
    """
    Holds three inference engines (front/left/right) in memory.
    Auto-detects V5 vs V6 per viewpoint. Switches active viewpoint instantly.
    """

    VIEWPOINT_PATHS = {
        "front": {
            "model": "app/deployment/front/models/model_front_v5_deploy.pth",
            "templates": "app/deployment/front/templates/feature_templates_v5.json",
            "threshold": 0.70,
        },
        "left": {
            "model": "app/deployment/left/models/model_left_v5_mirrored.pth",
            "templates": "app/deployment/left/templates/feature_templates_v5.json",
            "threshold": 0.70,
        },
        "right": {
            "model": "app/deployment/right/models/model_right_v5_standard.pth",
            "templates": "app/deployment/right/templates/feature_templates_v5.json",
            "threshold": 0.70,
        },
    }

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.engines = {}
        self.current_viewpoint = "front"
        self._load_all_engines()

    def _load_all_engines(self):
        for vp, paths in self.VIEWPOINT_PATHS.items():
            model_path = get_resource_path(paths["model"])
            templates_path = get_resource_path(paths["templates"])
            self.engines[vp] = ViewpointInferenceEngine(
                viewpoint=vp,
                model_path=model_path,
                templates_path=templates_path,
                device=self.device,
                confidence_threshold=paths["threshold"],
            )
        print(
            f"[MultiViewpointEngine] All 3 engines loaded. Active: {self.current_viewpoint}"
        )

    def set_viewpoint(self, viewpoint: str):
        """Switch active viewpoint instantly."""
        vp = (
            viewpoint.lower()
            .replace(" ", "_")
            .replace("side", "")
            .strip("_")
        )
        if vp not in self.engines:
            print(
                f"[WARN] Unknown viewpoint: {viewpoint}, keeping current: {self.current_viewpoint}"
            )
            return
        self.current_viewpoint = vp
        print(f"[MultiViewpointEngine] Active viewpoint set to: {vp}")

    def predict(
        self,
        pose_keypoints: np.ndarray,
        stick_keypoints: np.ndarray,
        global_features: dict,
        skip_threshold: bool = False,
    ) -> Tuple[str, float, np.ndarray]:
        return self.engines[self.current_viewpoint].predict(
            pose_keypoints, stick_keypoints, global_features, skip_threshold
        )

    def predict_for_class(
        self,
        pose_keypoints: np.ndarray,
        stick_keypoints: np.ndarray,
        global_features: dict,
        target_class: str,
    ) -> float:
        return self.engines[self.current_viewpoint].predict_for_class(
            pose_keypoints, stick_keypoints, global_features, target_class
        )

    def get_feature_corrections(
        self, global_features: dict, target_class: str
    ) -> Optional[dict]:
        return self.engines[self.current_viewpoint].get_feature_corrections(
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
        return self.engines[self.current_viewpoint].capture_similarity_snapshot(
            pose_keypoints,
            stick_keypoints,
            global_features,
            target_class,
            approach,
        )
