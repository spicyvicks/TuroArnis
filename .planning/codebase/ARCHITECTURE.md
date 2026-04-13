# TuroArnis System Architecture

**Analysis Date:** 2025-04-13

## High-Level Overview

TuroArnis is a real-time Arnis (Filipino martial arts) form correction system. It uses computer vision and machine learning to analyze practitioner movements, detect poses, and provide corrective feedback.

**Key Components:**
- Real-time pose detection using YOLOv8 + MediaPipe
- GCN (Graph Convolutional Network) pose classification (12 Arnis techniques)
- YOLO-based stick detection for weapon tracking
- Real-time feedback system with confidence scoring
- Multi-mode GUI (Free Practice, Guided Lessons)
- SQLite database for user/session tracking

---

## System Pattern

**Overall Architecture:** Layered Pipeline with Multiple Entry Points

**Key Characteristics:**
- **Multi-modal input**: Live camera feed, video files, or static images
- **Hybrid inference**: Pose detection (YOLO/Mediapipe) + Classification (GCN)
- **Viewpoint-aware models**: 3 specialist GCN models (front, left, right)
- **Multi-user support**: Up to 3 simultaneous practitioners
- **State-driven workflow**: Splash → Mode Select → Config → Zoning → Countdown → Feedback

---

## Core Components

### 1. Computer Vision Layer (`app/computer_vision/`)

**`pose_analyzer.py`** - Central analysis coordinator
- Purpose: Detects people, extracts pose keypoints, runs GCN classification
- Models used:
  - YOLOv8n (person detection)
  - YOLOv8n-Pose (live countdown visualization - 17 keypoints)
  - MediaPipe (pose extraction - 33 keypoints for classification)
  - Custom YOLO stick detector (`app/models/weights/best.pt`)
- Key methods:
  - `process_frame()` - Main entry point, mode-aware processing
  - `_detect_stick_with_yolo()` - Weapon endpoint detection
  - `_calculate_all_angles_3d()` - Joint angle computation
  - `clear_session_cache()` - Resets per-session state

**`feedback_analyzer.py`** - Form correction engine
- Purpose: Compares detected pose against target form, generates actionable feedback
- Thresholds: Per-viewpoint confidence thresholds (front: 0.60, left: 0.55, right: 0.65)
- Features:
  - Hybrid correction (GCN feature-based + hardcoded joint-angle fallback)
  - Body visibility gate (prevents false positives with partial body views)
  - Posture analysis (shoulder/hip alignment, spine check)
  - Stick detection status
  - Prioritized message system (errors → warnings → suggestions)

**`gcn_inference.py`** - Model inference manager
- Purpose: Loads and runs GCN specialist models
- Models: 3 .pth files (front, left, right viewpoints)
- Key methods:
  - `predict()` - Multi-hypothesis inference (tests all 12 technique templates)
  - `predict_for_class()` - Confidence for specific technique
  - `get_feature_corrections()` - Deviation analysis for feedback

**`feedback_mapper.py`** - Maps features to correction messages
- Hybrid scoring system for actionable feedback

---

### 2. GCN Model Architecture (`app/models/gcn/`)

**`model_architecture.py`** - Hybrid GCN V2
- Class: `HybridGCN`
- Architecture:
  - Node embeddings (35 nodes: 33 body + 2 stick)
  - 3-layer GCN with batch normalization
  - Global hybrid features MLP (30 geometric features)
  - Fusion layer combining GCN output + hybrid features
  - Classification head (12 classes + neutral)
- Edges: `SKELETON_EDGES` - 30 bidirectional connections (body + stick)
- Classes: 12 Arnis techniques (thrusts and blocks)

**`feature_extraction.py`** - Feature computation
- Functions:
  - `extract_raw_features()` - Complete feature extraction pipeline
  - `compute_global_features_from_kpts()` - 30 geometric features
  - `extract_node_features()` - Per-node features (6D: x,y,z,vis,dist_to_hip,angle_from_hip)
  - `compute_hybrid_features()` - Gaussian similarity scores vs templates
- Key features computed:
  - Joint angles (elbow, shoulder, knee)
  - Relative heights (wrists/elbows to hip center)
  - Horizontal positions (to hip center)
  - Stick orientation/length
  - Expert features (tip vs body landmarks)

**Model Config:** `app/models/gcn_model_config.json`
- 3 specialist model paths
- Per-viewpoint confidence thresholds
- Feature templates reference

---

### 3. Application Layer (`app/`)

**Entry Points:**
- `app.py` - Main kiosk application (fullscreen, multi-user)
- `main_app.py` - Development GUI (sidebar controls, debug mode)
- `main_video.py` - Video file analysis mode
- `main_image.py` - Static image analysis mode
- `eval_app.py` - Evaluation/testing interface

**`app.py` - Kiosk Application**
- Full-screen experience with state machine
- States: SPLASH → MODE_SELECT → USER_COUNT → CONFIG → ZONING → COUNTDOWN → SNAPSHOT → FEEDBACK
- Features:
  - Multi-user support (1-3 practitioners)
  - Guided lesson mode with 12 technique catalog
  - Animated GIF instruction display
  - Session pause/resume
  - Results screen
- UI: CustomTkinter canvas-based overlay system

**`main_app.py` - Desktop Application**
- Windowed interface with control panel
- Session management (start/end)
- Real-time confidence display
- Performance statistics
- Form selection dropdown (12 techniques)
- Viewpoint selector (front/left/right)
- Keyboard shortcuts (space to toggle session, F11 fullscreen)

---

### 4. GUI Components (`app/gui/`)

**`results_window.py`** - Performance statistics display
- 3-tab interface: Overview, Sessions, All Attempts
- 7-day statistics with accuracy breakdown
- Session history with duration/accuracy
- Detailed attempt log with correctness indicators

**`user_dialog.py`** - User management
- Create/select users
- Guest user support

**`toast.py`** - Notification system
**`loading_spinner.py`** - Initialization feedback
**`status_bar.py`** - Real-time system status
**`splash_screen.py`** - Application startup

---

### 5. Database Layer (`app/database/`)

**`db_manager.py`** - SQLite interface
- Tables:
  - `users` - Practitioner profiles
  - `sessions` - Practice sessions with target poses
  - `performances` - Individual pose attempts
- Features:
  - Automatic session timing
  - Joint angles storage (JSON)
  - Stick detection tracking
  - User statistics (7-day aggregation)
- Indexes: user_sessions, session_performances, user_performances

---

### 6. Utilities (`app/utils/`)

**`resource_path.py`** - Path resolution for PyInstaller
- Handles both development and packaged executable paths
- `get_resource_path()` - Relative to project root/_MEIPASS
- `get_app_data_path()` - User data directory

**`device_manager.py`** - CPU/GPU configuration
- TensorFlow/PyTorch device setup
- YOLO device selection

---

## Data Flow

### Image Processing Pipeline (Snapshot Mode)

```
Input Frame
    ↓
YOLO Person Detection → Bounding boxes + track IDs
    ↓
Person Crop (with padding)
    ↓
MediaPipe Pose (static_image_mode=True) → 33 keypoints (x,y,z,visibility)
    ↓
YOLO Stick Detection → Grip + Tip endpoints
    ↓
Feature Extraction:
    ├─ Node features: [35, 6] (pose + stick)
    ├─ Global features: 30 geometric measurements
    └─ Hybrid features: Gaussian similarity vs templates
    ↓
GCN Inference (viewpoint-specific model)
    ├─ Multi-hypothesis: Test against all 12 technique templates
    ├─ Best match wins
    └─ Confidence threshold filtering
    ↓
Feedback Analysis:
    ├─ Body visibility gate
    ├─ Hybrid corrections (GCN features + joint angles)
    ├─ Posture analysis
    └─ Message prioritization
    ↓
Output: {predicted_class, confidence, corrections[], is_correct}
```

### Live Video Pipeline (Countdown Mode)

```
Input Frame
    ↓
YOLO ByteTrack → Tracked person IDs
    ↓
YOLO-Pose (17 COCO keypoints) - Fast, no classification
    ↓
Stick detection (optional)
    ↓
Real-time visualization only
```

---

## Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `KioskApp` | `app/app.py` | Main fullscreen application |
| `PoseAnalyzer` | `computer_vision/pose_analyzer.py` | CV pipeline coordinator |
| `FeedbackAnalyzer` | `computer_vision/feedback_analyzer.py` | Form correction logic |
| `GCNInferenceEngine` | `computer_vision/gcn_inference.py` | Model inference |
| `HybridGCN` | `models/gcn/model_architecture.py` | Neural network |
| `DatabaseManager` | `database/db_manager.py` | Data persistence |
| `ResultsWindow` | `gui/results_window.py` | Statistics display |

---

## Model Architecture Summary

### GCN Classifier (Hybrid GCN V2)

**Input:**
- Pose: 33 MediaPipe landmarks (x, y, z, visibility)
- Stick: 2 endpoints (grip, tip) with confidence
- Graph: 35 nodes, 30 bidirectional edges

**Processing:**
1. Node embedding layer (35 → 8-dim)
2. GCN layers (input + 2 hidden, 128 channels)
3. Global mean pooling
4. Hybrid features MLP (30 → 128 → 128)
5. Fusion: Concat[GCN_out, Hybrid_out] → 128
6. Classification: 128 → 13 classes

**Output:**
- Class probabilities (softmax)
- Top prediction + confidence
- Per-class confidence for all 12 techniques

### Stick Detection (YOLO)

- Model: Custom trained YOLO pose model
- Input: Full frame or person crop
- Output: 2 keypoints (grip, tip) with confidence
- Correction: YOLO direction + shin-based length normalization

---

## Configuration Files

| File | Purpose |
|------|---------|
| `app/models/gcn_model_config.json` | Model paths, thresholds, templates |
| `app/models/gcn/feature_templates.json` | Per-class, per-viewpoint feature statistics |
| `app/assets/` | Logos, icons, UI graphics |
| `lesson/` | Instructional GIFs organized by viewpoint |

---

## Testing Tools

**`test_images_batch.py`** - CLI batch tester
- Analyzes single images or globs
- Viewpoint and target pose overrides
- JSON export for results
- Quiet mode for clean output
- Per-class softmax probability display

**`lesson_module_test.py`** - Lesson system validator

**`test_gcn_integration.py`** - GCN model tests

**`test_gcn_model_loading.py`** - Model loading verification

**`scripts/batch_test_trio.py`** - Multi-viewpoint batch testing

---

## Deployment

**Build System:**
- PyInstaller spec: `TuroArnis.spec`
- One-directory mode for asset access
- Includes: `.pth` models, database templates, lesson GIFs

**Packaged Assets:**
- GCN models: `app/models/hybrid_gcn_v2_*.pth`
- Stick detector: `app/models/weights/best.pt`
- YOLO base: `yolov8n.pt`, `yolov8n-pose.pt`
- Feature templates: `feature_templates.json`
- Lesson GIFs: `lesson/{front,left,right}_gif/*.gif`

---

*Architecture analysis: 2025-04-13*
