# TuroArnis Codebase Structure

**Analysis Date:** 2025-04-13

## Directory Layout

```
TuroArnis/
├── app/                           # Main application code
│   ├── __init__.py
│   ├── app.py                     # Fullscreen kiosk application
│   ├── main_app.py                # Desktop GUI (development)
│   ├── main_video.py              # Video file analysis mode
│   ├── main_image.py              # Static image analysis
│   ├── eval_app.py                # Evaluation interface
│   ├── app_1.py                   # Alternative GUI implementation
│   ├── computer_vision/           # CV pipeline components
│   │   ├── __init__.py
│   │   ├── pose_analyzer.py       # Main analyzer (YOLO + MediaPipe + GCN)
│   │   ├── feedback_analyzer.py   # Form correction logic
│   │   ├── gcn_inference.py       # GCN model inference
│   │   ├── feedback_mapper.py     # Feature-to-message mapping
│   │   └── gcn_processor.py       # GCN processing utilities
│   ├── models/                    # ML models
│   │   ├── __init__.py
│   │   ├── gcn_model_config.json  # Model paths & thresholds
│   │   ├── hybrid_gcn_v2_front.pth
│   │   ├── hybrid_gcn_v2_left.pth
│   │   ├── hybrid_gcn_v2_right.pth
│   │   ├── label_encoder.joblib
│   │   ├── weights/
│   │   │   └── best.pt            # YOLO stick detector
│   │   └── gcn/                   # GCN module
│   │       ├── __init__.py
│   │       ├── model_architecture.py
│   │       ├── feature_extraction.py
│   │       └── feature_templates.json
│   ├── database/                  # Data persistence
│   │   └── db_manager.py
│   ├── gui/                       # UI components
│   │   ├── __init__.py
│   │   ├── results_window.py
│   │   ├── user_dialog.py
│   │   ├── multi_user_dialog.py
│   │   ├── toast.py
│   │   ├── loading_spinner.py
│   │   ├── splash_screen.py
│   │   ├── status_bar.py
│   │   ├── TuroArnis_pyqt.py      # PyQt alternative
│   │   ├── TuroArnis_ttk.py       # Ttk alternative
│   │   └── draftTuroArnis.py      # Early prototype
│   └── utils/                     # Utilities
│       ├── __init__.py
│       ├── resource_path.py
│       └── device_manager.py
├── lesson/                        # Instructional media
│   ├── front_gif/                 # Front viewpoint GIFs (12 techniques)
│   ├── left_gif/                  # Left viewpoint GIFs
│   └── right_gif/                 # Right viewpoint GIFs
├── trio/                          # Training images by viewpoint
│   ├── front/                     # Front viewpoint images
│   ├── left/                      # Left viewpoint images
│   └── right/                     # Right viewpoint images
├── results_trio/                  # Analysis results output
├── scripts/                       # Utility scripts
│   ├── build_app.py               # PyInstaller build
│   ├── run_dist.py                # Run built distribution
│   ├── package_deployment.py      # Deployment packaging
│   ├── batch_test_trio.py         # Batch test all viewpoints
│   ├── test_yolo_pose_video.py    # YOLO-Pose testing
│   ├── test_stick_on_video.py     # Stick detection on video
│   ├── test_stick_on_image.py     # Stick detection on images
│   ├── test_stick_raw.py          # Raw stick testing
│   ├── test_user_tracking.py      # Multi-user tracking tests
│   ├── draw_mediapipe_landmarks.py
│   └── visualize_keypoint_methods.py
├── tests/                         # Test directory (inferred)
├── docs/                          # Documentation
├── build/                         # PyInstaller build output
├── dist/                          # Distribution packages
├── deployment_package/            # Deployment source
│   └── src/
├── eval_screenshots/              # Evaluation outputs
├── runs/                          # YOLO training runs
├── venv_311/                      # Python 3.11 virtual environment
├── .planning/                     # GSD planning documents
│   └── codebase/
│       ├── ARCHITECTURE.md
│       └── STRUCTURE.md
├── requirements.txt               # Python dependencies
├── TuroArnis.spec                 # PyInstaller specification
├── activate_env.bat               # Environment activation
├── .gitattributes
├── .gitignore
├── DEVELOPMENT_FLOWCHART.md       # Development process docs
├── GCN_INTEGRATION_SUMMARY.md     # GCN implementation notes
├── classification_failure_analysis.md
├── test_images_batch.py           # CLI batch tester
├── test_gcn_integration.py        # GCN integration tests
├── test_gcn_model_loading.py      # Model loading tests
├── lesson_module_test.py          # Lesson system tests
└── turoarnis.db                   # SQLite database (runtime)
```

---

## Directory Purposes

### `app/` - Core Application

**Entry Points (choose one):**
- `app.py` - Production kiosk (fullscreen, multi-user)
- `main_app.py` - Development GUI (windowed, sidebar controls)
- `main_video.py` - Video file testing
- `main_image.py` - Static image testing

**Computer Vision (`computer_vision/`):**
- `pose_analyzer.py` (850+ lines) - Main orchestrator
- `feedback_analyzer.py` (508 lines) - Correction logic
- `gcn_inference.py` (306 lines) - Model inference

**Models (`models/`):**
- GCN model weights: `hybrid_gcn_v2_{front,left,right}.pth`
- YOLO stick detector: `weights/best.pt`
- Configuration: `gcn_model_config.json`

**GUI (`gui/`):**
- `results_window.py` - Statistics display
- `user_dialog.py` - User management
- Various UI component modules

**Database (`database/`):**
- `db_manager.py` - SQLite interface (327 lines)

**Utils (`utils/`):**
- `resource_path.py` - PyInstaller path resolution
- `device_manager.py` - CPU/GPU configuration

---

### `lesson/` - Instructional Media

**Structure:** 3 subdirectories by viewpoint

```
lesson/
├── front_gif/          # 12 technique GIFs for front view
│   ├── crown.gif
│   ├── left_chest.gif
│   ├── left_elbow.gif
│   ├── left_eye.gif
│   ├── left_knee.gif
│   ├── left_temple.gif
│   ├── right_chest.gif
│   ├── right_elbow.gif
│   ├── right_eye.gif
│   ├── right_knee.gif
│   ├── right_temple.gif
│   └── solar_plexus.gif
├── left_gif/           # Same 12 for left view
└── right_gif/          # Same 12 for right view
```

**Usage:** Guided lesson mode displays these during instruction phase

---

### `trio/` - Training Data

Organized by camera viewpoint:
- `front/` - Front-facing training images
- `left/` - Left-side training images  
- `right/` - Right-side training images

Used for model training and batch testing

---

### `scripts/` - Development Tools

| Script | Purpose |
|--------|---------|
| `build_app.py` | Create PyInstaller executable |
| `batch_test_trio.py` | Test all 3 viewpoints |
| `test_stick_on_*.py` | Stick detection validation |
| `visualize_keypoint_methods.py` | Keypoint comparison |

---

## Key File Locations

### Configuration
- `app/models/gcn_model_config.json` - Model paths, thresholds
- `requirements.txt` - Python dependencies
- `TuroArnis.spec` - PyInstaller build config

### Models (Critical - Must be packaged)
- `app/models/hybrid_gcn_v2_front.pth`
- `app/models/hybrid_gcn_v2_left.pth`
- `app/models/hybrid_gcn_v2_right.pth`
- `app/models/weights/best.pt` (stick detector)
- `yolov8n.pt` (person detection)
- `yolov8n-pose.pt` (live visualization)

### Documentation
- `DEVELOPMENT_FLOWCHART.md` - Development process
- `GCN_INTEGRATION_SUMMARY.md` - GCN implementation
- `classification_failure_analysis.md` - Error analysis

---

## Test Files

**Root-level test scripts:**
- `test_images_batch.py` - CLI batch image tester (424 lines)
- `test_gcn_integration.py` - GCN integration tests (184 lines)
- `test_gcn_model_loading.py` - Model loading validation (92 lines)
- `lesson_module_test.py` - Lesson system tests (320 lines)

**Scripts directory:**
- `scripts/batch_test_trio.py` - Multi-viewpoint batch testing
- Various component-specific test scripts

---

## Naming Conventions

### Files
- Python modules: `snake_case.py`
- Test files: `test_*.py` or `*_test.py`
- Configuration: `*.json`, `*.yaml`
- Models: `hybrid_gcn_v2_{viewpoint}.pth`

### Directories
- Application code: singular (`app`, `gui`, `utils`)
- Resource directories: plural (`lessons`, `scripts`)
- Viewpoint variants: `front_gif`, `left_gif`, `right_gif`

---

## Where to Add New Code

### New Arnis Technique
1. **Model training data**: Add images to `trio/{front,left,right}/`
2. **GCN model**: Retrain `hybrid_gcn_v2_*.pth` models
3. **Feedback rules**: Add to `feedback_analyzer.py` joint targets
4. **Lesson GIFs**: Create GIFs in `lesson/{viewpoint}_gif/`
5. **App.py catalog**: Add to `TECHNIQUES` list in `app.py`

### New Computer Vision Feature
- Core logic: `app/computer_vision/pose_analyzer.py`
- Feature extraction: `app/models/gcn/feature_extraction.py`
- Visualization: Add drawing methods to `pose_analyzer.py`

### New GUI Component
- Main widgets: `app/gui/` new module
- Integration: `app.py` or `main_app.py` state handlers
- Styling: Use existing `COLOR_*` constants from `app.py`

### New Model Architecture
- Architecture: `app/models/gcn/model_architecture.py`
- Inference: `app/computer_vision/gcn_inference.py`
- Weights: `app/models/` directory
- Config: Update `gcn_model_config.json`

---

## Special Directories

### `build/` & `dist/`
- **Generated**: Yes (PyInstaller output)
- **Committed**: No (in `.gitignore`)
- **Purpose**: Executable builds

### `venv_311/`
- **Generated**: Virtual environment
- **Committed**: No
- **Purpose**: Python 3.11 dependencies

### `__pycache__/`
- **Generated**: Python bytecode
- **Committed**: No (in `.gitignore`)

### `.planning/codebase/`
- **Generated**: GSD planning documents
- **Committed**: Yes
- **Purpose**: Architecture and structure reference

---

## Database Schema

**SQLite file:** `turoarnis.db` (runtime)

**Tables:**
- `users` (id, name, created_at, is_active)
- `sessions` (id, user_id, target_pose, started_at, ended_at)
- `performances` (id, session_id, user_id, pose_detected, confidence, is_correct, timestamp, joint_angles, grip_angle, stick_detected)

**Indexes:**
- `idx_user_sessions` on sessions(user_id)
- `idx_session_performances` on performances(session_id)
- `idx_user_performances` on performances(user_id)

---

*Structure analysis: 2025-04-13*
