# External Integrations

**Analysis Date:** 2026-04-13

## Hardware Integration

**Camera System:**
- **DirectShow (Windows)** - Primary camera backend (`cv2.CAP_DSHOW`)
- **Fallback capture** - Standard OpenCV VideoCapture
- **Multi-camera search** - Iterates indices 0-2 for available cameras
- **Resolution support** - Processed at 480x360, displayed at screen resolution

**Input Devices:**
- **Keyboard** - Spacebar (toggle session), Enter (confirm), Escape (quit)
- **F11** - Fullscreen toggle
- **Ctrl+Q** - Application exit
- **Ctrl+R** - Results window

## Model Integrations

**YOLOv8 by Ultralytics**

*Person Detection:*
- **Model:** `yolov8n.pt` (YOLOv8 nano, 3.2MB)
- **Purpose:** Detect persons in frame for pose analysis
- **Classes:** Person only (class 0)
- **Inference:** `app/computer_vision/pose_analyzer.py`
- **Tracking:** ByteTrack algorithm for multi-person (up to 3 users)

*Pose Estimation:*
- **Model:** `yolov8n-pose.pt` (17 COCO keypoints)
- **Purpose:** Fast pose visualization during countdown
- **Used in:** Live tracking mode (not classification)
- **Speed:** Optimized for real-time feedback

*Stick Detection (Custom):*
- **Model:** `deployment_package/weights/best.pt` (or `app/models/weights/best.pt`)
- **Purpose:** Detect arnis stick (grip and tip keypoints)
- **Output:** 2 keypoints with confidence scores
- **Training:** Custom YOLO model on stick images
- **Correction:** Adaptive stick length correction using body proportions

**MediaPipe by Google**

*Pose Detection:*
- **Package:** `mediapipe==0.10.14`
- **Model:** BlazePose (33 keypoints, full-body)
- **Modes:**
  - Video mode: `static_image_mode=False` with temporal smoothing
  - Static mode: `static_image_mode=True` for snapshots
- **Output:** 33 landmarks with visibility scores and 3D coordinates
- **Integration:** `app/computer_vision/pose_analyzer.py` lines 70-90

**PyTorch Geometric**

*Graph Convolutional Networks:*
- **Architecture:** HybridGCN (custom implementation)
- **Specialist Models:** 3 viewpoint-specific models
  - Front: `app/models/hybrid_gcn_v2_front.pth`
  - Left: `app/models/hybrid_gcn_v2_left.pth`
  - Right: `app/models/hybrid_gcn_v2_right.pth`
- **Graph Structure:** 35 nodes (33 body + 2 stick), 30 edges
- **Features:** Node features (6-dim) + Hybrid global features (30-dim)
- **Classes:** 12 Arnis techniques + neutral (13 total)

## Data Storage

**SQLite Database**

*Location:* `turoarnis.db` (root and bundled in dist/)

*Schema:*
```sql
users (id, name, created_at, is_active)
sessions (id, user_id, target_pose, started_at, ended_at)
performances (id, session_id, user_id, pose_detected, confidence, 
              is_correct, timestamp, joint_angles, grip_angle, stick_detected)
```

*Integration:* `app/database/db_manager.py`
- User CRUD operations
- Session lifecycle management
- Performance logging with joint angles
- Statistics aggregation (7-day, session-level)

**JSON Data Files**

*Model Configuration:*
- **`app/models/gcn_model_config.json`**
  - Model paths per viewpoint
  - Confidence thresholds (0.70 default)
  - Class name mappings
  - Template file references

*Feature Templates:*
- **`app/models/gcn/feature_templates.json`**
  - 30 geometric features per technique
  - Mean, std, min, max per feature
  - 13 classes × 3 viewpoints = 39 templates
  - Used for hybrid similarity scoring in GCN

## Asset Integrations

**Lesson Content**

*Animated GIFs:*
- Location: `lesson/front_gif/`, `lesson/left_gif/`, `lesson/right_gif/`
- Content: 12 technique demonstrations per viewpoint
- Format: Animated GIFs, capped at 30 frames
- Loading: Async background thread with CTkImage conversion
- Display: CustomTkinter canvas with frame animation

*Static Reference Images:*
- Location: `app/assets/lesson_images/{front,left,right}/`
- Content: JPG technique references
- Purpose: Lesson instruction cards

**Application Assets**
- **`app/assets/TA.png`** - Application logo (150x150 splash)
- **`app/assets/TA.ico`** - Windows icon
- **`app/assets/lesson_images/`** - Static technique demonstrations

## Build Integration

**PyInstaller Packaging**

*Build Spec:* `TuroArnis.spec`
- **Mode:** One-directory (COLLECT)
- **Target:** `app/app.py` (Kiosk mode entry point)
- **Output:** `dist/TuroArnis/TuroArnis.exe`

*Bundled Data:*
```python
datas = [
    ('app/assets', 'app/assets'),
    ('lesson/front_gif', 'lesson/front_gif'),
    ('lesson/left_gif', 'lesson/left_gif'), 
    ('lesson/right_gif', 'lesson/right_gif'),
    ('app/models', 'app/models'),
    ('deployment_package', 'deployment_package'),
    ('yolov8n.pt', '.'),
]
```

*Hidden Imports:*
- sklearn, scipy, ultralytics, mediapipe, networkx
- PIL, customtkinter, torch_scatter, torch_sparse
- All app modules (GUI, CV, database)

*Exclusions:*
- `torch_geometric.distributed` (prevents RPC JIT error)
- `app.eval_app`, `app.main_app` (alternative entry points)

## Data Flow

**Real-time Pipeline:**
1. **Camera** → OpenCV VideoCapture
2. **Frame** → YOLO person detection
3. **Person crop** → MediaPipe (33 keypoints) OR YOLO-Pose (17 keypoints)
4. **Keypoints** → Feature extraction (30 geometric features)
5. **Features** → GCN inference (viewpoint-specific model)
6. **Classification** → Feedback analysis + Database logging
7. **Visualization** → CustomTkinter canvas with OpenCV overlay

**Offline/Batch Pipeline:**
1. **Image file** → OpenCV read
2. **Detection** → YOLO person detection
3. **Pose** → MediaPipe static mode
4. **Classification** → GCN with threshold filtering
5. **Results** → JSON/database storage

## Model I/O Specifications

**Input (GCN):**
- Pose keypoints: `[33, 4]` (x, y, z, visibility) normalized
- Stick keypoints: `[2, 4]` (grip, tip) with NaN sentinel when absent
- Global features: 30-dim dict (angles, heights, distances)

**Output (GCN):**
- Predicted class: One of 12 technique names or "No Technique Detected"
- Confidence: 0.0-1.0 probability
- All probabilities: 13-dim array (12 + neutral)

**Stick Detector I/O:**
- Input: Full frame image
- Output: Bounding box + 2 keypoints (grip, tip) with confidence
- Confidence threshold: 0.15 (lowered for occluded sticks)

## No External APIs

**Notable Absence:**
- No cloud services (AWS, Azure, GCP)
- No external REST APIs
- No authentication providers
- No telemetry/analytics services
- No update mechanisms
- Entirely offline/standalone operation

---

*Integration audit: 2026-04-13*
