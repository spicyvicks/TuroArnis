# TuroArnis System Development & GCN Integration Guide

## 1. System Overview

**TuroArnis** is a computer vision application designed to analyze and classify Arnis poses in real-time. It provides immediate feedback to practitioners, helping them improve their form. The system is built with a modular architecture, supporting both a Kiosk mode (for public deployments) and a Desktop mode (for personal use/testing).

### Key Features
- **Real-time Pose Detection**: Uses MediaPipe Pose for body landmarks.
- **Stick Detection**: Integrates a custom YOLOv8 model for Arnis stick tracking.
- **Form Classification**: Deploys specialist **Hybrid Graph Convolutional Networks (GCN)** to classify poses based on skeletal structure and stick position.
- **Viewpoint Robustness**: Automatically switches between specialist models (Front, Left, Right) based on user configuration.
- **Performance Feedback**: Visual and textual feedback on execution quality.

---

## 2. System Architecture

The application follows a modular design pattern, separating the User Interface, Business Logic, and Computer Vision components.

### 2.1 Frontend (User Interface)
- **Framework**: CustomTkinter (Modern UI wrapper for Tkinter).
- **Core Components**:
    - **Camera Feed**: Displays real-time video with skeletal overlays.
    - **Control Panel**: Allows users to select viewpoints, set timers, and view results.
    - **Feedback Overlay**: Shows classification confidence and corrective suggestions.
- **Entry Points**:
    - `app/app.py`: The **Kiosk** interface (optimized for touch/unattended use).
    - `app/main_app.py`: The **Desktop** interface (standard windowed application).

### 2.2 Backend & Data Management
- **Database**: SQLite (`app/database/db_manager.py`).
    - Stores session history, classification results, and user performance metrics.
- **Threading**: Heavy computations (CV pipeline) run on separate threads to keep the UI responsive.
- **State Management**: Handles application states (e.g., `IDLE`, `COUNTDOWN`, `RECORDING`, `FEEDBACK`).

### 2.3 Computer Vision Pipeline (`app/computer_vision/`)
The CV pipeline processes every video frame through several stages:
1.  **Frame Acquisition**: Captures raw frame from webcam (`cv2`).
2.  **Pose Detection**: MediaPipe extracts 33 body landmarks.
3.  **Stick Detection**: YOLOv8 detects the Arnis stick bounding box and keypoints.
4.  **Feature Extraction**: Combines pose and stick data into a unified feature set.
5.  **Inference**: The GCN model predicts the pose class.

---

## 3. GCN Model Integration

The core intelligence of TuroArnis lies in its **Hybrid Graph Convolutional Network (GCN)**, designed specifically to capture the structural relationship between body parts and the weapon (stick).

### 3.1 Model Architecture (`HybridGCN`)
The model is defined in `app/models/gcn/model_architecture.py`. It is a "Hybrid" GCN because it fuses two types of features:
1.  **Graph Features**: Processed by **GCN Layers (SAGEConv)**.
    - **Nodes**: 35 Keypoints (33 Body + 2 Stick Endpoints).
    - **Edges**: Physical connections (bones) + Stick-Hand connections.
    - **Input**: Spatial coordinates (x, y, z), visibility, and relative angles.
2.  **Global/Hybrid Features**: Processed by a separate **MLP (Multi-Layer Perceptron)**.
    - Captures high-level geometric relationships (e.g., shoulder alignment, stick angle relative to ground) via similarity to reference templates.

These branches are concatenated and passed through a final classifier to output probabilities for 13 Arnis classes.

### 3.2 Feature Engineering (`app/models/gcn/feature_extraction.py`)
Raw keypoints are not fed directly. They undergo rigorous preprocessing:
- **Normalization**: Keypoints are centered at the hip and scaled by torso size to be invariant to camera distance and user height.
- **Feature Computation**:
    - **Node Features**: `[x, y, z, visibility, dist_to_hip, angle_to_hip]` (6 dimensions per node).
    - **Hybrid Features**: Cosine similarity scores against "Gold Standard" templates stored in `feature_templates.json`.

### 3.3 Training Pipeline
The model development followed these steps:
1.  **Data Collection**: Recorded experts performing forms from 3 viewpoints (Front, Left, Right).
2.  **Augmentation**: Applied rotation, scaling, and jittering to synthetic data to improve robustness.
3.  **Training**:
    - Trained 3 distinct "Specialist" models: `hybrid_gcn_v2_front`, `hybrid_gcn_v2_left`, `hybrid_gcn_v2_right`.
    - Loss Function: CrossEntropyLoss.
    - Optimizer: AdamW.
4.  **Validation**: Evaluated on a held-out test set to ensure high accuracy (>90%) on unseen subjects.

### 3.4 Runtime Inference Flow
When the app runs, the `PoseAnalyzer` coordinates the inference:
1.  **Load Models**: On startup, all 3 specialist models are loaded into memory (`app/utils/device_manager.py` handles CPU/GPU).
2.  **Select Specialist**: Based on the UI "Viewpoint" setting, the corresponding model is activated.
3.  **Predict**:
    - Input Frame -> MediaPipe + YOLO -> Feature Vector.
    - Feature Vector -> GCN (Forward Pass) -> Logits.
    - Logits -> Softmax -> **Class & Confidence**.
4.  **Thresholding**: If confidence > 60%, the pose is accepted; otherwise classified as "Unknown" or "Incorrect".

---

## 4. Development & Deployment

### 4.1 Environment Setup
- **Python**: 3.11+
- **Dependencies**: Listed in `requirements.txt` (Key libs: `torch`, `torch_geometric`, `mediapipe`, `ultralytics`, `customtkinter`).
- **Hardware**: CPU-optimized (runs ~10 FPS on standard laptops), optional CUDA support.

### 4.2 Building the Application
We use **PyInstaller** to package the app into a standalone executable.
- **Spec File**: `TuroArnis.spec` defines the build configuration.
    - **Hidden Imports**: Explicitly includes dynamic libraries like `sklearn` and `torch_geometric`.
    - **Data Files**: bundles `models/` and `assets/` into the `_internal` directory.
- **Build Script**: `scripts/build_app.py` automates the clean build process.

### 4.3 Deployment Package
The final output is a structured zip file containing:
- `bin/`: The compiled `TuroArnis.exe` and dependencies.
- `models/`: Raw `.pth` model files (for reference or hot-swapping).
- `docs/`: Documentation for end-users.

---

## 5. Future Roadmap
- **Temporal Analysis**: Using LSTM/GRU to analyze the *sequence* of movements, not just static poses.
- **Multi-Person Tracking**: Optimizing the GCN to handle multiple skeletons in the scene simultaneously without performance degradation.
- **Web Dashboard**: Syncing local database results to a cloud platform for instructor review.
