# GCN Model Integration Plan
## Using Hybrid GCN V2 Specialist Models in TuroArnis Desktop App

**Date**: 2026-02-10  
**Version**: 1.0  
**Status**: Implementation Ready

---

## Executive Summary

**Objective**: Replace the current DNN/RF/XGBoost/Ensemble models in `app/models/` with the 3 Hybrid GCN V2 specialist models from `deployment_package/models/` while preserving all existing application functionality, UI components, and user experience.

### Current State vs Target State

| Aspect | Current | Target |
|--------|---------|--------|
| **Models** | DNN/RF/XGBoost/Ensemble (TensorFlow/sklearn) | 3 GCN specialists (Front/Left/Right) |
| **Framework** | TensorFlow 2.x + scikit-learn | PyTorch 2.10 + PyTorch Geometric |
| **Features** | 99 normalized coordinates (flattened) | 35 graph nodes (33 pose + 2 stick) + 30 hybrid scores |
| **Input** | Flat vector [99] | Graph: nodes[35,6] + edges + hybrid[30] |
| **Accuracy** | 55-65% | Expected 75%+ |
| **Inference Speed** | ~5ms | ~5ms (similar) |
| **FPS** | ~10 FPS | ~10 FPS (bottleneck is YOLO) |

---

## Architecture Overview

### Current Architecture
```
Input: Webcam Frame (640x480)
    ↓
YOLOv8n Person Detection + ByteTrack
    ↓
MediaPipe Pose (33 landmarks)
    ↓
YOLO Stick Detector (2 keypoints)
    ↓
Feature Extraction: 99 normalized coordinates
    ↓
Model: DNN/RF/XGBoost/Ensemble
    ↓
Prediction: Class name + confidence
    ↓
Feedback Analyzer
    ↓
UI: CustomTkinter
```

### Target Architecture (GCN)
```
Input: Webcam Frame (640x480)
    ↓
YOLOv8n Person Detection + ByteTrack (UNCHANGED)
    ↓
MediaPipe Pose (33 landmarks) (UNCHANGED)
    ↓
YOLO Stick Detector (2 keypoints) (UNCHANGED)
    ↓
NEW: Node Feature Extraction (35 nodes × 6 features)
     - x, y, z, visibility, distance_to_hip_3d, angle_from_hip
    ↓
NEW: Hybrid Feature Extraction (30 similarity scores)
     - Compare global features to reference templates
    ↓
NEW: Graph Construction (35 nodes + skeleton edges)
    ↓
NEW: GCN Model (Front/Left/Right specialist)
     - PyTorch model with GCNConv layers
     - Output: 13-class logits
    ↓
Prediction: Class name + confidence
    ↓
Feedback Analyzer (UNCHANGED)
    ↓
UI: CustomTkinter (UNCHANGED)
```

---

## Major Changes Required

### 1. Dependency Updates

**File**: `requirements.txt` (root)

**Additions**:
```txt
# PyTorch (CPU version for deployment)
torch==2.10.0+cpu
torch-geometric==2.4.0
torch-scatter==2.1.2+pt21cpu
torch-sparse==0.6.18+pt21cpu

# Keep existing:
# opencv-python>=4.8.0
# mediapipe>=0.10.0
# ultralytics>=8.0.0
# customtkinter>=5.2.0
# Pillow>=10.0.0
# joblib>=1.3.0
# numpy>=1.24.0
```

**Why**: GCN models require PyTorch and PyTorch Geometric for graph convolution operations.

---

### 2. New Model Configuration System

**File**: `app/models/gcn_model_config.json` (NEW)

```json
{
  "model_type": "gcn_hybrid_v2",
  "viewpoint": "front",
  "models": {
    "front": {
      "path": "deployment_package/models/hybrid_gcn_v2_front.pth",
      "accuracy": 0.85,
      "description": "Front viewpoint specialist"
    },
    "left": {
      "path": "deployment_package/models/hybrid_gcn_v2_left.pth",
      "accuracy": 0.82,
      "description": "Left side viewpoint specialist"
    },
    "right": {
      "path": "deployment_package/models/hybrid_gcn_v2_right.pth",
      "accuracy": 0.83,
      "description": "Right side viewpoint specialist"
    }
  },
  "feature_templates": "deployment_package/src/feature_templates.json",
  "stick_detector": "deployment_package/weights/best.pt",
  "class_names": [
    "crown_thrust_correct", "left_chest_thrust_correct", "left_elbow_block_correct",
    "left_eye_thrust_correct", "left_knee_block_correct", "left_temple_block_correct",
    "neutral_stance", "right_chest_thrust_correct", "right_elbow_block_correct",
    "right_eye_thrust_correct", "right_knee_block_correct", "right_temple_block_correct",
    "solar_plexus_thrust_correct"
  ]
}
```

**Why**: We need a new configuration system to manage 3 specialist models and specify viewpoint-specific feature templates.

---

### 3. Copy Deployment Package Source Files

**Files to Copy** (from `deployment_package/src/` to `app/models/gcn/`):

| Source | Destination | Purpose |
|--------|-------------|---------|
| `deployment_package/src/model_architecture.py` | `app/models/gcn/model_architecture.py` | HybridGCN class, SKELETON_EDGES, CLASS_NAMES |
| `deployment_package/src/feature_extraction.py` | `app/models/gcn/feature_extraction.py` | Node feature extraction, hybrid feature computation |
| `deployment_package/src/feature_templates.json` | `app/models/gcn/feature_templates.json` | Reference pose templates |

**Why**: These files contain the exact feature extraction logic and model architecture used during GCN training.

---

### 4. New GCN Inference Engine

**File**: `app/computer_vision/gcn_inference.py` (NEW)

```python
"""
GCN Inference Engine for Hybrid GCN V2 Models
Replaces the TensorFlow/sklearn model inference in pose_analyzer.py
"""

import torch
import numpy as np
import json
import os
from typing import Optional, Tuple, List
from torch_geometric.data import Data

from app.models.gcn.model_architecture import HybridGCN, SKELETON_EDGES, CLASS_NAMES
from app.models.gcn.feature_extraction import (
    extract_node_features,
    compute_hybrid_features,
    gaussian_similarity
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
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        templates_path = self.config['feature_templates']
        with open(templates_path, 'r') as f:
            self.templates = json.load(f)

    def _load_models(self):
        """Load all 3 specialist models"""
        for viewpoint, model_info in self.config['models'].items():
            checkpoint = torch.load(model_info['path'], map_location=self.device)

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
                  f"(accuracy: {checkpoint['test_accuracy']:.2%})")

    def _prepare_graph_structure(self):
        """Prepare edge index for graph convolution"""
        self.edge_index = torch.tensor(SKELETON_EDGES, dtype=torch.long).t().to(self.device)

    def set_viewpoint(self, viewpoint: str):
        """Switch active viewpoint model"""
        if viewpoint not in self.models:
            raise ValueError(f"Unknown viewpoint: {viewpoint}")
        self.current_viewpoint = viewpoint

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
        node_features = extract_node_features(pose_keypoints, stick_keypoints)

        # Compute hybrid features [30]
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
        model = self.models[self.current_viewpoint]
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
        _gcn_engine = GCNInferenceEngine(device=device)
    return _gcn_engine
```

**Why**: This module encapsulates all GCN-specific logic, handling model loading, viewpoint switching, feature extraction, and graph-based inference.

---

### 5. Modify pose_analyzer.py

**File**: `app/computer_vision/pose_analyzer.py`

#### A. Replace Model Loading Section

**Current** (lines 76-256):
```python
# Try to load from active_model.json...
# Loads TensorFlow Keras or sklearn models
self.pose_classifier_model = ...
self.label_encoder = ...
self.scaler = ...
```

**New**:
```python
# Import GCN inference engine
from app.computer_vision.gcn_inference import get_gcn_engine

# Load GCN models
try:
    self.gcn_engine = get_gcn_engine(device=self.device_info['torch_device'])
    self.pose_classifier_model = self.gcn_engine
    self.label_encoder = None  # Not needed
    self.scaler = None  # Not needed
    self.is_gcn = True
    self.is_ensemble = False
    print("[info] GCN specialist models loaded")
except Exception as e:
    print(f"[critical] Could not load GCN models: {e}")
    self.pose_classifier_model = None
    self.is_gcn = False
```

#### B. Replace Inference Section

**Current** (lines 554-593):
```python
# Feature extraction for current models
world_landmarks = pose_results.pose_world_landmarks.landmark
landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks])
hip_center = (landmarks_np[23] + landmarks_np[24]) / 2.0
coords = (landmarks_np - hip_center).flatten()

if self.scaler is not None:
    coords = self.scaler.transform(coords.reshape(1, -1))[0]

# Run DNN/RF/XGBoost/Ensemble prediction...
```

**New**:
```python
# GCN feature extraction
from app.models.gcn.feature_extraction import extract_raw_features, calculate_angle

# Prepare data for GCN
raw_data = {
    'pose_keypoints': np.array([[lm.x, lm.y, lm.z, lm.visibility]
                                for lm in pose_results.pose_landmarks.landmark]),
    'stick_keypoints': None,
    'global_features': {}
}

# Add stick keypoints if detected
if stick_endpoints:
    grip_pt, tip_pt = stick_endpoints
    h, w = frame.shape[:2]
    raw_data['stick_keypoints'] = np.array([
        [grip_pt[0]/w, grip_pt[1]/h, 0.0, 1.0],
        [tip_pt[0]/w, tip_pt[1]/h, 0.0, 1.0]
    ])
else:
    # Default stick positions
    raw_data['stick_keypoints'] = np.array([[0.5, 0.5, 0.0, 0.0],
                                             [0.5, 0.5, 0.0, 0.0]])

# Compute global geometric features
kpts = raw_data['pose_keypoints']
global_features = {
    'left_elbow_angle': calculate_angle(kpts[11], kpts[13], kpts[15]),
    'right_elbow_angle': calculate_angle(kpts[12], kpts[14], kpts[16]),
    'left_shoulder_angle': calculate_angle(kpts[13], kpts[11], kpts[23]),
    'right_shoulder_angle': calculate_angle(kpts[14], kpts[12], kpts[24]),
    'left_knee_angle': calculate_angle(kpts[23], kpts[25], kpts[27]),
    'right_knee_angle': calculate_angle(kpts[24], kpts[26], kpts[28]),
    # ... remaining features
}

# Run GCN inference
predicted_class, confidence, probs = self.gcn_engine.predict(
    raw_data['pose_keypoints'],
    raw_data['stick_keypoints'],
    global_features
)
```

**Why**: The GCN requires a completely different feature extraction pipeline - graph nodes with 6 features each + hybrid similarity scores.

---

### 6. Add Viewpoint Selector to UI

**File**: `app/main_app.py`

**Add after form selector** (around line 284):

```python
# Viewpoint selector for GCN specialist models
viewpoint_frame = ctk.CTkFrame(session_frame, fg_color="transparent")
viewpoint_frame.pack(fill="x", pady=5, padx=10)

ctk.CTkLabel(viewpoint_frame, text="Viewpoint:", font=("Inter", 11),
             text_color="#7f8c8d").pack(side="left", padx=(0, 10))

self.viewpoint_var = ctk.StringVar(value="front")
self.viewpoint_selector = ctk.CTkOptionMenu(
    viewpoint_frame,
    variable=self.viewpoint_var,
    values=["front", "left", "right"],
    command=self.on_viewpoint_changed,
    fg_color="#9b59b6",
    button_color="#8e44ad",
    button_hover_color="#7d3c98",
    width=120,
    font=("Inter", 12)
)
self.viewpoint_selector.pack(side="left")

def on_viewpoint_changed(self, viewpoint):
    """Handle viewpoint selection change"""
    if hasattr(self, 'analyzer') and self.analyzer.gcn_engine:
        self.analyzer.gcn_engine.set_viewpoint(viewpoint)
        self.activity_label.configure(
            text=f"Viewpoint switched to {viewpoint}",
            text_color="#9b59b6"
        )
```

**Why**: Users must select the appropriate viewpoint for their camera angle to use the correct specialist model.

---

### 7. Update Feedback Analyzer

**File**: `app/computer_vision/feedback_analyzer.py`

**Status**: No major changes required

The feedback analyzer works with:
- `predicted_class` (string)
- `confidence` (float)
- `live_angles` (dict of joint angles)
- `grip_angle` (float)
- `stick_endpoints` (tuple)

These are all still provided by the GCN pipeline. The feedback analyzer is model-agnostic.

---

## Directory Structure After Changes

```
TuroArnis/
├── app/
│   ├── main_app.py                    # MODIFIED: Add viewpoint selector
│   ├── computer_vision/
│   │   ├── pose_analyzer.py           # MODIFIED: Use GCN inference
│   │   ├── gcn_inference.py           # NEW: GCN inference engine
│   │   ├── feedback_analyzer.py       # UNCHANGED
│   │   └── ...
│   ├── models/
│   │   ├── gcn_model_config.json      # NEW: GCN model configuration
│   │   ├── gcn/                       # NEW: GCN source files
│   │   │   ├── model_architecture.py  # COPIED from deployment_package
│   │   │   ├── feature_extraction.py  # COPIED from deployment_package
│   │   │   └── feature_templates.json # COPIED from deployment_package
│   │   ├── active_model.json          # DEPRECATED
│   │   └── ... (legacy models)
│   ├── database/
│   │   └── db_manager.py              # UNCHANGED
│   └── ...
├── deployment_package/                # SOURCE OF MODELS & WEIGHTS
│   ├── models/
│   │   ├── hybrid_gcn_v2_front.pth   # USED by app
│   │   ├── hybrid_gcn_v2_left.pth    # USED by app
│   │   └── hybrid_gcn_v2_right.pth   # USED by app
│   ├── weights/
│   │   └── best.pt                    # USED by app
│   └── src/                           # COPIED to app/models/gcn/
├── requirements.txt                   # MODIFIED: Add PyTorch deps
└── ...
```

---

## Implementation Roadmap

### Phase 1: Setup & Dependencies (1-2 hours)

1. **Update requirements.txt**
   - Add PyTorch, PyTorch Geometric dependencies
   - Test installation in clean virtual environment

2. **Copy deployment package files**
   - Copy `src/` files to `app/models/gcn/`
   - Verify file integrity

### Phase 2: Core GCN Integration (4-6 hours)

3. **Create `gcn_model_config.json`**
   - Define paths to deployment_package models
   - Configure class names

4. **Create `gcn_inference.py`**
   - Implement GCNInferenceEngine class
   - Test model loading
   - Verify inference on sample data

5. **Modify `pose_analyzer.py`**
   - Replace model loading logic
   - Replace inference logic
   - Ensure backward compatibility flags

### Phase 3: UI Integration (2-3 hours)

6. **Add viewpoint selector to main_app.py**
   - Add dropdown for front/left/right
   - Wire up to GCN engine
   - Add visual indicator

7. **Update loading/splash screen**
   - Change "Loading AI models..." message

### Phase 4: Testing & Validation (4-6 hours)

8. **Unit tests**
   - Test feature extraction produces correct shapes
   - Test GCN inference returns valid predictions
   - Test viewpoint switching

9. **Integration tests**
   - Full pipeline: Camera → YOLO → MediaPipe → GCN → UI
   - Test all 13 classes
   - Test with/without stick detection

10. **Performance benchmark**
    - Measure FPS with GCN vs. old models
    - Compare accuracy on known test poses

---

## Key Technical Considerations

### 1. Hybrid Feature Computation

**Challenge**: Which template to use for inference?

**Solution**: Use `neutral_stance` as the base reference for all poses, or compute hybrid features for all 13 classes and use the max similarity. Start with `neutral_stance` reference for simplicity.

### 2. Stick Detection Dependency

**Challenge**: GCN expects 35 nodes (33 pose + 2 stick). What if stick detection fails?

**Solution**: Use default positions `[[0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]]` when stick not detected. Models were trained with this fallback.

### 3. Viewpoint Selection UX

**Options**:
1. **Manual dropdown** (Phase 1) - User selects front/left/right
2. **Auto-detection** (Future) - Train lightweight CNN to classify camera angle

**Recommendation**: Start with manual selection, add auto-detection later.

### 4. Performance Impact

- **GCN inference**: ~5ms (very fast)
- **Feature extraction**: ~5ms (calculating angles, distances)
- **Total overhead**: ~10ms vs. old pipeline
- **Expected FPS**: Still ~10 FPS (bottleneck is YOLO at ~60ms)

### 5. Backward Compatibility

Keep old model loading code as fallback:

```python
if os.path.exists('app/models/gcn_model_config.json'):
    # Use GCN
    self.gcn_engine = get_gcn_engine()
else:
    # Fallback to legacy models
    # ... existing code ...
```

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| PyTorch Geometric installation fails | Medium | High | Provide detailed install instructions; use conda if pip fails |
| GCN accuracy worse than expected | Low | High | Keep old models as fallback; A/B test before full deploy |
| Performance degradation | Low | Medium | Benchmark early; optimize feature extraction if needed |
| Stick detection still required | High | Medium | Models work with default stick positions; feature degradation |
| Viewpoint misclassification | Medium | Medium | Clear UI instructions; add warning if confidence low |

---

## Expected Benefits

1. **Higher Accuracy**: GCN specialist models should outperform generalist DNN (75%+ vs. 55-65%)
2. **Viewpoint Robustness**: Each model specialized for its angle (front/left/right)
3. **Better Spatial Understanding**: Graph structure captures body topology better than flattened coordinates
4. **Maintainable**: Clean separation of GCN logic in dedicated module
5. **Future-Proof**: Easy to add more specialist models (e.g., overhead view)

---

## Open Questions

1. **Should we keep the old models as fallback?**
   - Yes: Safer deployment, can rollback instantly
   - No: Cleaner codebase, less confusion

2. **Should we implement auto-viewpoint detection now or later?**
   - Now: More complex, requires additional model/training
   - Later: Manual selection is acceptable for MVP

3. **Should we support dynamic viewpoint switching during session?**
   - Yes: User can move around camera
   - No: Fixed camera position per session

4. **Should we log which viewpoint model was used for each prediction?**
   - Yes: Useful for analytics and debugging
   - Add `viewpoint` column to `performance` table

---

## Model Specifications Reference

### Hybrid GCN V2 Architecture
- **Input**: 35 nodes (33 MediaPipe + 2 YOLO stick keypoints)
- **Node Features**: 6 per node (x, y, z, visibility, distance_to_hip, angle_from_hip)
- **Global Features**: 30 similarity scores (hybrid features)
- **Hidden Dimension**: 256
- **Layers**: 3 GCN layers + MLP fusion
- **Output**: 13 classes (Arnis stances)
- **File Size**: ~1.35 MB per model (4.05 MB total)

### Classes (13 total)
1. `crown_thrust_correct`
2. `left_chest_thrust_correct`
3. `left_elbow_block_correct`
4. `left_eye_thrust_correct`
5. `left_knee_block_correct`
6. `left_temple_block_correct`
7. `neutral_stance`
8. `right_chest_thrust_correct`
9. `right_elbow_block_correct`
10. `right_eye_thrust_correct`
11. `right_knee_block_correct`
12. `right_temple_block_correct`
13. `solar_plexus_thrust_correct`

---

## Files Modified Summary

| File | Change Type | Lines Changed | Description |
|------|-------------|---------------|-------------|
| `requirements.txt` | Modify | +6 lines | Add PyTorch dependencies |
| `app/models/gcn_model_config.json` | Create | ~40 lines | New model configuration |
| `app/models/gcn/*.py` | Create | ~335 lines | Copy from deployment_package |
| `app/computer_vision/gcn_inference.py` | Create | ~150 lines | New inference engine |
| `app/computer_vision/pose_analyzer.py` | Modify | ~100 lines | Replace model loading & inference |
| `app/main_app.py` | Modify | ~40 lines | Add viewpoint selector |

**Total New Files**: 4  
**Total Modified Files**: 3  
**Estimated Implementation Time**: 12-18 hours

---

*Generated for TuroArnis Project - 2026-02-10*
