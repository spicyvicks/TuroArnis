# Technology Stack

**Analysis Date:** 2026-04-13

## Languages

**Primary:**
- Python 3.11 - All application logic, ML inference, and GUI

**Configuration/Data:**
- JSON - Model configs (`app/models/gcn_model_config.json`), feature templates (`app/models/gcn/feature_templates.json`)
- SQL - SQLite database operations (`app/database/db_manager.py`)
- YAML - Ultralytics dataset configs (embedded in `dist/TuroArnis/_internal/ultralytics/cfg/`)

## Runtime

**Environment:**
- Python 3.11.0+
- Windows (primary target platform)
- CPU-optimized (GPU supported but optional)

**Package Manager:**
- pip (requirements.txt)
- Lockfile: Not committed (manual version pinning in requirements.txt)

## Frameworks

**GUI Framework:**
- **CustomTkinter 5.2.0+** - Primary modern UI framework (`app/app.py`, `app/main_app.py`)
  - Light/dark theme support (`ctk.set_appearance_mode("light")`)
  - Custom color themes and scaling
- **PyQt6 6.8.0** - Secondary/alternative GUI (`app/gui/TuroArnis_pyqt.py`)
- **Tkinter** - Base GUI layer (bundled with Python)

**Computer Vision:**
- **OpenCV 4.8.0+** (`cv2`) - Image processing, camera capture, frame manipulation
- **MediaPipe 0.10.14** - Pose detection (33 keypoints), landmark extraction
  - `mp.solutions.pose` for body pose detection
  - Static and video modes configured
- **Ultralytics/YOLOv8 8.3.25** - Object detection and tracking
  - YOLO for person detection (`yolov8n.pt`)
  - YOLO-Pose for fast keypoints (`yolov8n-pose.pt`)
  - Custom stick detection model (`best.pt`)
  - ByteTrack for person tracking

**Machine Learning / Deep Learning:**
- **PyTorch 2.1.0+cpu** - Primary ML framework
  - CPU-optimized build for deployment
  - Model inference for GCN classification
- **PyTorch Geometric 2.4.0** - Graph Convolutional Networks
  - `GCNConv`, `global_mean_pool` for pose classification
  - `torch-scatter 2.1.2+pt21cpu` - Sparse operations
  - `torch-sparse 0.6.18+pt21cpu` - Sparse tensor support
- **NumPy 1.26.4** - Numerical computations, array operations
- **SciPy 1.17.0** - Scientific computing, statistical operations

**Build/Packaging:**
- **PyInstaller 6.12.0** - Standalone executable creation
  - `TuroArnis.spec` - Build configuration
  - One-directory mode with `_internal` folder
  - UPX compression enabled
  - Custom hidden imports for ML libraries

**Database:**
- **SQLite3** - Embedded database (Python standard library)
  - User management, session tracking, performance records
  - Stored in `turoarnis.db`

**Image Processing:**
- **Pillow (PIL) 10.0.0+** - Image manipulation, format conversion
- **ImageTk** - Tkinter image integration

## Key Dependencies

**Critical Runtime:**
```
requirements.txt
├── opencv-python>=4.8.0    # Camera, image processing
├── mediapipe==0.10.14      # 33-keypoint pose detection
├── ultralytics==8.3.25     # YOLO detection, ByteTrack
├── customtkinter>=5.2.0    # Modern Tkinter UI
├── Pillow>=10.0.0          # Image manipulation
├── numpy==1.26.4           # Numerical ops
├── scipy==1.17.0           # Scientific computing
├── joblib>=1.3.0           # Model serialization
├── torch==2.1.0+cpu        # PyTorch CPU
├── torchvision==0.16.0+cpu # Vision utilities
├── torch-geometric==2.4.0  # GCN architecture
├── torch-scatter==2.1.2+pt21cpu  # Sparse ops
├── torch-sparse==0.6.18+pt21cpu  # Sparse tensors
├── PyQt6==6.8.0            # Alternative GUI
└── pyinstaller==6.12.0     # Executable build
```

**Development Environment:**
- `venv_311/` - Python 3.11 virtual environment (observed)
- `activate_env.bat` - Environment activation script

## Model Files

**Pre-trained Models:**
1. **`yolov8n.pt`** (root and `app/`) - YOLOv8 nano person detector
2. **`yolov8n-pose.pt`** - YOLOv8 nano pose estimator (17 keypoints)
3. **`deployment_package/weights/best.pt`** - Custom YOLO stick detector (2 keypoints: grip, tip)
4. **`app/models/hybrid_gcn_v2_*.pth`** - Three specialist GCN models:
   - `hybrid_gcn_v2_front.pth` - Front viewpoint classifier
   - `hybrid_gcn_v2_left.pth` - Left side viewpoint classifier  
   - `hybrid_gcn_v2_right.pth` - Right side viewpoint classifier

**Model Architecture:**
- **HybridGCN** class in `app/models/gcn/model_architecture.py`
  - 35 graph nodes (33 body + 2 stick)
  - Node-specific features + global hybrid context
  - 12 + 1 output classes (12 techniques + neutral)

## Configuration

**Environment:**
- No `.env` file required - self-contained application
- Device auto-detection in `app/utils/device_manager.py`
- GPU optional (CUDA checked but CPU fallback enforced)

**Build Configuration:**
- `TuroArnis.spec` - PyInstaller spec file
  - Target: `app/app.py` (Kiosk mode)
  - Assets bundled: `app/assets/`, `lesson/`, `app/models/`
  - Excludes: `torch_geometric.distributed` (RPC error prevention)

**Application Config:**
- `app/models/gcn_model_config.json` - Model paths, thresholds, class names
- `app/models/gcn/feature_templates.json` - 30 geometric features per technique

## Platform Requirements

**Development:**
- Windows 10/11
- Python 3.11
- 8GB+ RAM recommended (for ML inference)
- Webcam for live testing

**Production (Deployed):**
- Windows executable (`dist/TuroArnis/TuroArnis.exe`)
- Self-contained (all DLLs bundled in `_internal/`)
- SQLite database auto-created on first run
- No external dependencies required

---

*Stack analysis: 2026-04-13*
