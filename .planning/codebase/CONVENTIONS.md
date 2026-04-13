# Coding Conventions

**Analysis Date:** 2025-04-13

## Overview

This codebase is a Python-based computer vision application for Arnis (Filipino martial arts) form correction. It uses a Graph Convolutional Network (GCN) for pose classification and provides real-time feedback through a customtkinter GUI.

---

## Language & Style

**Primary Language:** Python 3.x

**Code Style:** PEP 8 with some project-specific adaptations

### Key Style Patterns Observed:

- **Line Length:** Approximately 100-120 characters (relaxed from PEP 8's 79)
- **Indentation:** 4 spaces (consistent throughout)
- **Quotes:** Double quotes for strings, single quotes for dict keys where preferred
- **Blank Lines:** 
  - 2 blank lines between top-level functions and classes
  - 1 blank line between methods within classes

---

## Naming Conventions

### Files
- **Modules:** `snake_case.py` (e.g., `pose_analyzer.py`, `feedback_analyzer.py`)
- **Test Files:** `test_*.py` or `*_test.py` (e.g., `test_gcn_integration.py`, `lesson_module_test.py`)
- **Scripts:** Descriptive `snake_case.py` in `scripts/` directory

### Classes
- **PascalCase** with descriptive names
- Examples from codebase:
  ```python
  class PoseAnalyzer:
  class FeedbackAnalyzer:
  class HybridGCN(nn.Module):
  class GCNInferenceEngine:
  class DatabaseManager:
  class KioskApp(ctk.CTk):
  ```

### Functions & Methods
- **snake_case** with descriptive, action-oriented names
- Examples:
  ```python
  def process_frame(self, frame, skip_ml_inference=False, ...)
  def analyze(self, result: Dict, target_form: str, ...)
  def compute_global_features_from_kpts(kpts, stick_keypoints)
  def _calculate_all_angles_3d(self, landmark_list)  # private method
  ```

### Variables
- **snake_case** for local variables
- **UPPER_CASE** for constants at module level
- Type hints used extensively for function parameters and returns

### Constants (Module-Level)
```python
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
COLOR_BG = "#74b9ff"
COLOR_ACCENT = "#2980b9"
CLASS_NAMES = [...]
SKELETON_EDGES = [...]
```

---

## Import Organization

### Import Order (observed pattern):
1. **Standard library imports** (`os`, `sys`, `json`, `time`, `threading`)
2. **Third-party imports** (`cv2`, `numpy`, `torch`, `mediapipe`, `customtkinter`)
3. **Local project imports** (from `app.*`, `app.models.*`, etc.)

### Import Style Examples:
```python
# Standard library
import os
import sys
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party
import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
import customtkinter as ctk
from PIL import Image, ImageTk
from ultralytics import YOLO

# Local (with path bootstrap for PyInstaller compatibility)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.computer_vision.pose_analyzer import PoseAnalyzer
from app.models.gcn.model_architecture import CLASS_NAMES
from app.utils.resource_path import get_resource_path
```

### Path Bootstrap Pattern
All entry-point scripts use this pattern for PyInstaller compatibility:
```python
# Path bootstrap (same as main app)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
```

---

## Type Hints

**Extensive use of type hints** throughout the codebase:

### Function Signatures:
```python
def analyze(
    self,
    result: Dict,
    target_form: str,
    confidence_threshold: float = None,
    viewpoint: str = 'front',
    gcn_engine=None
) -> Dict:

def compute_global_features_from_kpts(kpts, stick_keypoints) -> dict:

def get_resource_path(relative_path: str) -> str:
```

### Type Aliases:
```python
from typing import Dict, List, Optional, Tuple
```

---

## Documentation & Comments

### Docstring Style
- **Google-style** docstrings for classes and functions
- Triple double quotes for docstrings

```python
"""
Feedback Analyzer - Provides detailed form correction feedback
Analyzes pose data against target form requirements and generates actionable corrections
"""

def analyze_image(
    image_path: str,
    pose_analyzer: PoseAnalyzer,
    feedback_analyzer: FeedbackAnalyzer,
    viewpoint: str = "front",
    target_pose: str = None,
    quiet: bool = False,
) -> dict:
    """
    Run the full pipeline on a single image and return a result dict.
    """
```

### Inline Comments
- Used extensively for explaining complex logic
- Prefix conventions observed:
  - `[info]`, `[DEBUG]`, `[ERROR]`, `[WARN]` - Log-like prefixes for print statements
  - `FIX #N:` - Reference to specific fixes
  - `# ── Section Name ──` - Decorative section separators

### Section Separators
```python
# ── Path bootstrap (same as main app) ─────────────────────────────────────
# ── Theme (matches kiosk) ──────────────────────────────────────────────────
# ── ML engines ─────────────────────────────────────────────────────────────
# ── State ─────────────────────────────────────────────────────────────────
# ── Root container ─────────────────────────────────────────────────────────
```

---

## Configuration Patterns

### JSON-Based Configuration
Configuration stored in JSON files, loaded at runtime:

**`app/models/gcn_model_config.json`**
```json
{
  "models": {
    "front": {
      "path": "deployment_package/models/hybrid_gcn_v2_front.pth",
      "confidence_threshold": 0.55
    },
    "left": {
      "path": "deployment_package/models/hybrid_gcn_v2_left.pth",
      "confidence_threshold": 0.55
    },
    "right": {
      "path": "deployment_package/models/hybrid_gcn_v2_right.pth",
      "confidence_threshold": 0.65
    }
  },
  "feature_templates": "app/models/gcn/feature_templates.json"
}
```

### PyInstaller Resource Path Pattern
```python
def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and PyInstaller."""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # Development mode
        base_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..'))
    return os.path.join(base_path, relative_path)
```

---

## Class Design Patterns

### Main Application Class
```python
class KioskApp(ctk.CTk):
    """Main application class inheriting from customtkinter.CTk"""
    
    def __init__(self):
        super().__init__()
        self.title("TuroArnis Kiosk")
        self.configure(fg_color=COLOR_BG)
        # ... initialization
```

### Analyzer Classes
- Encapsulate specific functionality
- Initialize expensive resources in `__init__`
- Provide clear public methods

```python
class PoseAnalyzer:
    """Handles pose detection, stick detection, and GCN classification."""
    
    def __init__(self, detection_interval=3, stick_model_path=None, debug_stick=False):
        self.yolo_model = YOLO(yolo_base_path)
        self.pose = self.mp_pose.Pose(...)
        # ... initialization
        
    def process_frame(self, frame, skip_ml_inference=False, mode='snapshot'):
        """Main processing method."""
        pass
        
    def close(self):
        """Cleanup resources."""
        self.pose.close()
```

---

## Error Handling

### Try-Except Pattern
```python
try:
    self.pose_analyzer = PoseAnalyzer(...)
    print("[Kiosk] Pose Analyzer initialized successfully")
except Exception as e:
    print(f"[Kiosk] Warning: Could not initialize Pose Analyzer: {e}")
    self.pose_analyzer = None
```

### Debug Print Pattern
```python
print(f"[DEBUG-YOLO] Processing frame: {w}x{h}, mode={mode}")
print(f"[DEBUG-STICK] Stick detected - Confidence: {confidence:.3f}")
print(f"[ERROR] GCN inference failed: {e}")
```

---

## GUI Patterns (customtkinter)

### Color Constants
```python
COLOR_BG = "#74b9ff"           # Light Blue (Splash Style)
COLOR_ACCENT = "#2980b9"       # Strong Blue for buttons
COLOR_ACCENT_HOVER = "#3498db" # Lighter Blue for hover
COLOR_SUCCESS = "#27ae60"      # Green
COLOR_WARNING = "#e74c3c"      # Red
COLOR_TEXT = "#2c3e50"         # Dark Text for Light BG
```

### Widget Creation Pattern
```python
btn = ctk.CTkButton(
    self.video_canvas,
    text="START PRACTICE",
    font=("Inter", 28, "bold"),
    fg_color=COLOR_ACCENT,
    hover_color=COLOR_ACCENT_HOVER,
    height=80,
    width=300,
    corner_radius=40,
    command=self.show_mode_select
)
```

---

## Naming Conventions for ML/Computer Vision

### Keypoint/Landmark Naming
- `landmarks_2d` - MediaPipe normalized coordinates
- `landmarks_absolute` - Pixel coordinates in frame
- `world_landmarks` - 3D world coordinates from MediaPipe
- `pose_kpts_array` - NumPy array of pose keypoints
- `stick_kpts_array` - NumPy array of stick keypoints

### Model Naming
- `yolo_model` - Main YOLO detector
- `stick_detector` - YOLO stick detection model
- `gcn_engine` - GCN inference engine
- `pose` / `pose_static` - MediaPipe pose instances

---

## Module Structure

### `__init__.py` Pattern
```python
"""
Initialize gcn package
"""
```
Minimal but present in package directories.

### Directory Layout
```
app/
├── __init__.py
├── app.py                    # Main application entry
├── eval_app.py               # Evaluation/testing GUI
├── computer_vision/
│   ├── __init__.py
│   ├── pose_analyzer.py      # Main CV pipeline
│   ├── feedback_analyzer.py  # Feedback generation
│   └── gcn_inference.py      # GCN model inference
├── models/
│   ├── gcn/
│   │   ├── __init__.py
│   │   ├── model_architecture.py
│   │   └── feature_extraction.py
│   └── gcn_model_config.json
├── gui/
│   ├── results_window.py
│   ├── user_dialog.py
│   └── ...
├── database/
│   └── db_manager.py
└── utils/
    ├── resource_path.py
    └── device_manager.py
```

---

*Conventions analysis: 2025-04-13*
