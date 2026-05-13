# TuroArnis — Source Code Reference

> **Purpose**: Annotated code excerpts for thesis reviewers.  
> **Focus**: HybridGCN V5 training pipeline, feature extraction, inference, and feedback generation.  
> **Omitted**: Legacy GUI implementations (`TuroArnis_ttk.py`, `TuroArnis_pyqt.py`), utility helpers (`resource_path.py`, `device_manager.py`), and database layer (see User Manual Section 8).

---

## 1. Model Architecture — HybridGCN V5

**File**: `app/models/gcn/model_v5.py`  
**Role**: Deployment model definition. Exact match to the trained checkpoint.

### 1.1 Class Signature & Initialization

```python
class HybridGCN(nn.Module):
    """
    GCN with Node-Specific Features + Global Hybrid Context
    - GCN processes spatial node features (35 nodes × 6 features)
    - Global hybrid features provide 46 expert-knowledge geometric descriptors
    - Both branches are fused for final classification into 13 classes
    """
    def __init__(self, num_node_features, num_hybrid_features, num_classes=13,
                 hidden_dim=128, num_layers=3, dropout=0.5, node_embed_dim=8):
        super(HybridGCN, self).__init__()

        # Learnable identity embedding for each of the 35 graph nodes
        self.node_embedding = nn.Embedding(35, node_embed_dim)

        # GCN stack: 3 GCNConv layers + BatchNorm + ReLU + Dropout
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.convs.append(
            GCNConv(num_node_features + node_embed_dim, hidden_dim)
        )
        self.batch_norms.append(
            nn.BatchNorm1d(hidden_dim, track_running_stats=False)
        )
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(
                nn.BatchNorm1d(hidden_dim, track_running_stats=False)
            )

        # Hybrid MLP: compresses 46 global features into hidden_dim//2
        self.hybrid_mlp = nn.Sequential(
            nn.Linear(num_hybrid_features, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU()
        )

        # Fusion + classification head
        self.fc1 = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout_layer = nn.Dropout(dropout)
```

> **Design rationale**: The graph branch captures topological body structure; the hybrid branch injects domain-specific geometric knowledge (angles, distances, stick position). Fusion occurs at the penultimate layer so both modalities influence classification jointly.

---

### 1.2 Forward Pass

```python
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        hybrid_features = data.hybrid_features
        batch_size = batch.max().item() + 1

        # --- Node identity embedding ---
        node_indices = torch.arange(35, device=x.device) \
                         .unsqueeze(0).expand(batch_size, -1)
        node_emb = self.node_embedding(node_indices) \
                       .view(-1, self.node_embed_dim)
        x = torch.cat([x, node_emb], dim=-1)   # [batch*35, 6+8]

        # --- GCN layers with residual connections ---
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout_layer(x_new)
            # Residual connection where dimensions match
            if x_new.size(-1) == x.size(-1):
                x = x_new + x
            else:
                x = x_new

        # --- Global mean pooling over nodes ---
        x_pool = global_mean_pool(x, batch)   # [batch, hidden_dim]

        # --- Hybrid feature processing ---
        if hybrid_features.dim() == 1:
            hybrid_features = hybrid_features.view(batch_size, -1)
        hybrid_out = self.hybrid_mlp(hybrid_features)   # [batch, hidden_dim//2]

        # --- Fusion & classification ---
        combined = torch.cat([x_pool, hybrid_out], dim=-1)
        x_out = F.relu(self.fc1(combined))
        x_out = self.dropout_layer(x_out)
        logits = self.fc2(x_out)
        return logits
```

> **Key point**: `track_running_stats=False` in BatchNorm is intentional — the training batch size is small (32), and running statistics become unstable. Inference therefore uses batch statistics directly.

---

### 1.3 Skeleton Graph Edges

```python
SKELETON_EDGES = [
    # Torso
    (11, 12), (12, 11),          # Shoulders ↔
    (11, 23), (23, 11),          # L shoulder ↔ L hip
    (12, 24), (24, 12),          # R shoulder ↔ R hip
    (23, 24), (24, 23),          # Hips ↔
    # Arms
    (11, 13), (13, 11), (13, 15), (15, 13),   # Left
    (12, 14), (14, 12), (14, 16), (16, 14),   # Right
    # Legs
    (23, 25), (25, 23), (25, 27), (27, 25),   # Left
    (24, 26), (26, 24), (26, 28), (28, 26),   # Right
    # Stick (nodes 33=grip, 34=tip)
    (15, 33), (33, 15), (16, 33), (33, 16),   # Wrists ↔ grip
    (33, 34), (34, 33),                       # Grip ↔ tip
]
```

> **Graph topology**: 35 nodes = 33 MediaPipe body landmarks + 2 stick keypoints (grip, tip). Bidirectional edges allow message passing in both directions. The stick is anchored to whichever wrist is closer during inference.

---

### 1.4 Deployment Model Loader (Auto-Inference)

```python
def load_deployment_model(checkpoint_path, device='cpu'):
    """Load model with automatic hyperparameter inference from state dict.
    Handles checkpoints where config metadata may be incomplete."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get('config', {})
    state = ckpt['model_state_dict']

    defaults = {
        'num_node_features': 6,
        'num_hybrid_features': 46,
        'num_classes': 13,
        'hidden_dim': 128,
        'num_layers': 3,
        'dropout': 0.5,
        'node_embed_dim': 8,
    }
    merged = {**defaults, **config}

    # Infer from actual weight shapes when config is missing
    if 'hybrid_mlp.0.weight' in state:
        merged['num_hybrid_features'] = state['hybrid_mlp.0.weight'].shape[1]
        merged['hidden_dim'] = state['hybrid_mlp.0.weight'].shape[0] * 2
    if 'fc2.weight' in state:
        merged['num_classes'] = state['fc2.weight'].shape[0]
    if 'convs.0.lin.weight' in state:
        conv_in = state['convs.0.lin.weight'].shape[1]
        merged['num_node_features'] = conv_in - merged.get('node_embed_dim', 8)

    model = HybridGCN(**merged).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt.get('class_names', []), merged
```

> **Robustness**: This loader ensures a single checkpoint file is sufficient for deployment — no external config file is strictly required.

---

## 2. Training Pipeline — HybridGCN V2/V5

**File**: `scripts/4c_train_hybrid_gcn_v2_optimized.py`  
**Role**: End-to-end training with class-imbalance handling and overfitting detection.

### 2.1 Hyperparameters & Optimizations

```python
# =============================================================================
# CONFIGURATION — OPTIMIZED FOR ~1,871 TRAINING SAMPLES
# =============================================================================

HIDDEN_DIM = 128              # REDUCED: was 256 → mitigates overfitting
NUM_LAYERS = 3                # Depth retained for hierarchical features
DROPOUT = 0.7                 # INCREASED: was 0.5 → fixes 21% train-test gap
NODE_EMBED_DIM = 16           # INCREASED: was 8 → better node differentiation

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4           # ADDED: L2 regularization
EPOCHS = 150
PATIENCE = 15                 # REDUCED: was 20 → faster overfit detection
BATCH_SIZE = 32
MAX_OVERFIT_GAP = 25.0        # ADDED: hard stop if train-test gap > 25%
```

> **Phase 5.7 optimizations**: The model was originally over-parameterized for the dataset size (~1,871 samples). Reducing hidden dimension and increasing dropout tightened the generalization gap from 21% to <5%.

---

### 2.2 Graph Dataset & Weighted Sampling

```python
class GraphDataset(Dataset):
    """Loads pre-computed node features and hybrid features per sample."""
    def __init__(self, features_path, viewpoint=None):
        self.data = torch.load(features_path, map_location='cpu')
        # Optional viewpoint filtering (front / left / right)
        if viewpoint:
            mask = [v == viewpoint for v in self.data['viewpoints']]
            self.node_features = self.data['node_features'][mask]
            self.hybrid_features = self.data['hybrid_features'][mask]
            self.labels = self.data['labels'][mask]
        else:
            self.node_features = self.data['node_features']
            self.hybrid_features = self.data['hybrid_features']
            self.labels = self.data['labels']

    def __getitem__(self, idx):
        edge_index = torch.tensor(SKELETON_EDGES, dtype=torch.long).t()
        return Data(
            x=self.node_features[idx],           # [35, num_node_features]
            edge_index=edge_index,                # [2, 56] (bidirectional)
            hybrid_features=self.hybrid_features[idx],  # [46]
            y=self.labels[idx]
        )

def create_weighted_sampler(dataset):
    """WeightedRandomSampler fixes 6.7:1 class imbalance (neutral dominates)."""
    labels = dataset.labels.numpy()
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
```

> **Class imbalance**: The dataset contains far more "neutral stance" frames than active techniques. Without `WeightedRandomSampler`, the model would converge to predicting neutral for all inputs.

---

### 2.3 Training Loop with Early Stopping

```python
def train_model(train_dataset, val_dataset, viewpoint=None, merged=False):
    # --- Data loaders ---
    sampler = create_weighted_sampler(train_dataset)
    train_loader = GeoDataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        sampler=sampler, collate_fn=collate_fn, drop_last=True
    )
    val_loader = GeoDataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn
    )

    # --- Model & optimizer ---
    sample = train_dataset[0]
    model = HybridGCN(
        num_node_features=sample.x.size(1),
        num_hybrid_features=sample.hybrid_features.size(0),
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM
    ).to(DEVICE)

    # Xavier initialization for better convergence
    model.apply(lambda m: nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) else None)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    class_weights = compute_class_weights(train_dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --- Training loop ---
    best_val_acc = 0.0
    patience_counter = 0
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, preds, labels = evaluate(model, val_loader, criterion)

        # Early stopping: check overfit gap
        overfit_gap = train_acc - val_acc
        if overfit_gap > MAX_OVERFIT_GAP:
            print(f"STOP: Overfit gap {overfit_gap:.1f}% exceeds {MAX_OVERFIT_GAP}%")
            break

        # Checkpoint best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': history['config'],
                'class_names': CLASS_NAMES,
                'test_accuracy': val_acc / 100.0,
            }, best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

        scheduler.step(val_acc)
```

> **Dual stopping criteria**: (1) `ReduceLROnPlateau` decays LR when validation plateaus; (2) a hard `MAX_OVERFIT_GAP` threshold prevents the model from memorizing the small training set.

---

## 3. Feature Extraction Pipeline

**File**: `app/models/gcn/feature_extraction.py`  
**Role**: Converts raw image → MediaPipe pose + YOLO stick → node features + 46 hybrid features.

### 3.1 Raw Feature Extraction

```python
def extract_raw_features(image, stick_detector):
    """Extract pose keypoints (MediaPipe) and stick keypoints (YOLO) from image."""
    img = cv2.imread(image) if isinstance(image, str) else image
    if img is None:
        return None
    h, w = img.shape[:2]

    # --- MediaPipe Pose (33 keypoints) ---
    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(static_image_mode=True, model_complexity=2)
    results = pose_detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pose_detector.close()
    if not results.pose_landmarks:
        return None

    kpts = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                     for lm in results.pose_landmarks.landmark])   # [33, 4]

    # --- YOLO Stick Detector (grip + tip) ---
    stick_results = stick_detector(img, verbose=False)[0]
    if stick_results.keypoints is not None and len(stick_results.keypoints.data) > 0:
        sk = stick_results.keypoints.data[0].cpu().numpy()
        stick_grip = [sk[0,0]/w, sk[0,1]/h, 0.0, sk[0,2]]
        stick_tip  = [sk[1,0]/w, sk[1,1]/h, 0.0, sk[1,2]]
    else:
        # NaN sentinel when stick is not detected
        stick_grip = [np.nan, np.nan, 0.0, 0.0]
        stick_tip  = [np.nan, np.nan, 0.0, 0.0]

    stick_keypoints = np.array([stick_grip, stick_tip])   # [2, 4]

    # --- Global geometric features ---
    features = compute_global_features_from_kpts(kpts, stick_keypoints)

    return {
        'pose_keypoints': kpts,
        'stick_keypoints': stick_keypoints,
        'global_features': features,
        'has_stick_detected': stick_results.keypoints is not None
    }
```

> **V5 adaptation**: For V5/V6 models, NaN stick coordinates are sanitized to `[0,0,0,0]` so all hybrid features remain computable. The model learned the zero-offset distribution during training.

---

### 3.2 Hybrid Feature Computation (V5)

```python
def compute_global_features_from_kpts(kpts, stick_keypoints, world_landmarks=None,
                                      has_stick_detected=None, version='v5'):
    """Compute 46 hybrid features: 33 Gaussian similarity + 13 signed direction."""
    stick_grip = stick_keypoints[0]
    stick_tip  = stick_keypoints[1]

    # V5: sanitize NaN → origin (model learned constant offsets)
    if version in ('v5', 'v6'):
        if np.isnan(stick_grip).any():
            stick_grip = np.array([0.0, 0.0, 0.0, 0.0])
        if np.isnan(stick_tip).any():
            stick_tip = np.array([0.0, 0.0, 0.0, 0.0])
        stick_available = True
    else:
        stick_available = not (np.isnan(stick_grip[0]) or np.isnan(stick_tip[0]))

    features = {}

    # --- 3D joint angles (using world landmarks for v5/v6) ---
    if version in ('v5', 'v6') and world_landmarks is not None:
        get_point = lambda idx: [world_landmarks[idx].x,
                                 world_landmarks[idx].y,
                                 world_landmarks[idx].z]
    else:
        get_point = lambda idx: kpts[idx]

    features['left_elbow_angle']  = calculate_angle(get_point(11), get_point(13), get_point(15))
    features['right_elbow_angle']  = calculate_angle(get_point(12), get_point(14), get_point(16))
    features['left_shoulder_angle'] = calculate_angle(get_point(13), get_point(11), get_point(23))
    features['right_shoulder_angle']= calculate_angle(get_point(14), get_point(12), get_point(24))
    features['left_knee_angle']   = calculate_angle(get_point(23), get_point(25), get_point(27))
    features['right_knee_angle']   = calculate_angle(get_point(24), get_point(26), get_point(28))

    # --- Height & position relative to hip center ---
    hip_center_y = (kpts[23][1] + kpts[24][1]) / 2
    hip_center_x = (kpts[23][0] + kpts[24][0]) / 2
    features['left_wrist_height']  = hip_center_y - kpts[15][1]
    features['right_wrist_height'] = hip_center_y - kpts[16][1]
    features['left_wrist_x']       = kpts[15][0] - hip_center_x
    features['right_wrist_x']      = kpts[16][0] - hip_center_x

    # --- Stick-dependent geometric features ---
    if stick_available:
        features['stick_tip_height']   = hip_center_y - stick_tip[1]
        features['stick_grip_height']  = hip_center_y - stick_grip[1]
        features['stick_angle']        = calculate_angle(stick_grip[:3], stick_tip[:3], [stick_tip[0]+1, stick_tip[1], 0])
        features['stick_dx']           = stick_tip[0] - stick_grip[0]
        features['stick_dy']           = stick_tip[1] - stick_grip[1]
        features['stick_length']       = calculate_distance(stick_grip[:2], stick_tip[:2])
        features['tip_vs_nose']         = calculate_distance(stick_tip[:2], kpts[0][:2])
        features['tip_vs_shoulder']     = calculate_distance(stick_tip[:2], kpts[12][:2])

    # --- Signed direction features (13 features for V5) ---
    features['stick_tip_signed_x']    = np.sign(stick_tip[0] - hip_center_x)
    features['grip_signed_x']         = np.sign(stick_grip[0] - hip_center_x)
    features['tip_vs_nose_signed']      = np.sign(stick_tip[1] - kpts[0][1])
    features['tip_vs_shoulder_signed']  = np.sign(stick_tip[1] - kpts[12][1])
    features['left_elbow_angle_signed'] = np.sign(features['left_elbow_angle'] - 90)
    features['right_elbow_angle_signed']= np.sign(features['right_elbow_angle'] - 90)
    # ... (additional signed features omitted for brevity)

    return features
```

> **Feature engineering**: 33 base features are normalized distances/angles; 13 signed features encode directional information (left/right, above/below). The model thus receives both magnitude and semantic direction cues.

---

## 4. Inference Pipeline

### 4.1 GCN Inference Engine (Public API)

**File**: `app/computer_vision/gcn_inference.py`  
**Role**: Backwards-compatible wrapper around the multi-viewpoint engine.

```python
class GCNInferenceEngine:
    """Backwards-compatible wrapper around MultiViewpointEngine.
    Exposes .config and .templates for app.py and test scripts."""

    def __init__(self, config_path=None, device="cpu"):
        self.device = torch.device(device)
        # Load per-viewpoint confidence thresholds
        config_path = config_path or "app/models/gcn_model_config.json"
        with open(get_resource_path(config_path), "r") as f:
            self.config = json.load(f)

        # Create per-viewpoint multi-engine (loads all 3 V5/V6 models)
        self._multi_engine = MultiViewpointEngine(device=device)

        # Merge templates from all viewpoints for similarity computation
        self.templates = {}
        for engine in self._multi_engine.engines.values():
            self.templates.update(engine.templates)

    def set_viewpoint(self, viewpoint: str):
        """Switch active viewpoint instantly."""
        self._multi_engine.set_viewpoint(viewpoint)

    def predict(self, pose_keypoints, stick_keypoints, global_features,
                skip_threshold=False):
        """Run GCN inference on extracted features.
        Returns: (predicted_class_name, confidence, all_probabilities)"""
        return self._multi_engine.predict(
            pose_keypoints, stick_keypoints, global_features, skip_threshold
        )
```

> **Dynamic viewpoint switching**: The kiosk app calls `set_viewpoint()` when the user changes from "Front" to "Left Side" or "Right Side", routing inference to the corresponding specialist model without reloading.

---

### 4.2 Viewpoint Inference Engine (Auto-Version Detection)

**File**: `app/deployment/viewpoint_engine.py`  
**Role**: Per-viewpoint engine that auto-detects V5 vs V6 from checkpoint metadata.

```python
class ViewpointInferenceEngine:
    """Self-contained inference engine for a single viewpoint.
    Auto-detects V5 vs V6 from checkpoint and uses correct pipeline."""

    def __init__(self, viewpoint, model_path, templates_path,
                 device="cpu", confidence_threshold=0.70):
        self.viewpoint = viewpoint
        self.device = torch.device(device)

        # --- Auto-detect version from checkpoint ---
        ckpt = torch.load(model_path, map_location=device)
        cfg = ckpt.get("config", {})
        ver_str = cfg.get("version", "")
        if ver_str.startswith("v6") or cfg.get("num_node_features") == 7:
            self.version = "v6"
        elif ver_str.startswith("v5") or cfg.get("num_hybrid_features") == 46:
            self.version = "v5"
        else:
            # Fallback: inspect weight shapes
            state = ckpt["model_state_dict"]
            if "hybrid_mlp.0.weight" in state:
                hybrid_dim = state["hybrid_mlp.0.weight"].shape[1]
                self.version = "v6" if hybrid_dim >= 49 else "v5"
            else:
                self.version = "v5"
        print(f"[ViewpointEngine] Detected {self.version.upper()} for {viewpoint}")

        # Load model & templates
        if self.version == "v6":
            from app.models.gcn.model_v6 import load_deployment_model
            self.model, self.class_names, _ = load_deployment_model(
                model_path, device=device
            )
            self.extract_features = extract_node_features_v6
            self.compute_hybrid = compute_hybrid_features_v6
        else:
            from app.models.gcn.model_v5 import load_deployment_model
            self.model, self.class_names, _ = load_deployment_model(
                model_path, device=device
            )
            self.extract_features = extract_node_features
            self.compute_hybrid = compute_hybrid_features_v5

        with open(templates_path, "r") as f:
            self.templates = json.load(f)

    def predict(self, pose_keypoints, stick_keypoints, global_features):
        """Full inference: extract node features → build graph → run model."""
        node_features = self.extract_features(pose_keypoints, stick_keypoints)
        hybrid_features = self.compute_hybrid(global_features, self.templates,
                                               self.viewpoint)
        edge_index = torch.tensor(SKELETON_EDGES, dtype=torch.long).t()

        graph = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=edge_index,
            hybrid_features=torch.tensor(hybrid_features, dtype=torch.float32)
        )
        batch = torch.zeros(35, dtype=torch.long)   # single graph

        with torch.no_grad():
            logits = self.model(graph)
            probs = torch.softmax(logits, dim=1)
            pred_class = probs.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()

        return self.class_names[pred_class], confidence, probs[0].cpu().numpy()
```

> **Runtime polymorphism**: The same `predict()` method works for both V5 and V6 checkpoints. Version detection inspects both metadata (`config.version`) and weight tensor shapes (`hybrid_mlp.0.weight.shape[1]`), making it resilient to incomplete checkpoints.

---

### 4.3 Pose Analyzer (Real-Time Integration)

**File**: `app/computer_vision/pose_analyzer.py`  
**Role**: Orchestrates MediaPipe, YOLO, and GCN into a single `process_frame()` call.

```python
class PoseAnalyzer:
    def __init__(self, detection_interval=3, stick_model_path=None,
                 debug_stick=False, disable_stick_correction=False):
        # --- Device configuration (GPU if available) ---
        self.device_info = configure_device(verbose=True)
        self.yolo_device = get_yolo_device(self.device_info)

        # --- YOLO person detector (for zone assignment) ---
        self.yolo_model = YOLO(get_resource_path('yolov8n.pt'))
        self.yolo_model.to(self.yolo_device)

        # --- YOLO stick detector ---
        self.stick_detector = None
        if stick_model_path and os.path.exists(stick_model_path):
            self.stick_detector = YOLO(stick_model_path)
            self.stick_detector.to(self.yolo_device)

        # --- MediaPipe Pose ---
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,          # balance accuracy vs. CPU speed
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            smooth_landmarks=True        # temporal smoothing for video
        )

        # --- GCN inference engine ---
        self.gcn_engine = get_gcn_engine()

        # --- Temporal smoothing buffers ---
        self.stick_buffer = []
        self.stick_buffer_size = 5
        self.track_history = {}

    def detect_stick(self, frame, person_bbox=None, debug=False,
                      skip_smoothing=False):
        """Run YOLO stick detector; return (grip, tip) keypoints."""
        if self.stick_detector is None:
            return None, None
        try:
            result = self.stick_detector(frame, verbose=False)[0]
            if len(result.boxes) == 0:
                return None, None

            # --- Select best stick: highest IoU with person bbox ---
            best_stick_idx = 0
            if person_bbox:
                px1, py1, px2, py2 = person_bbox
                best_iou = -1.0
                for i in range(min(3, len(result.boxes))):
                    bx1, by1, bx2, by2 = map(int, result.boxes[i].xyxy[0].tolist())
                    inter = max(0, min(px2,bx2)-max(px1,bx1)) * \
                            max(0, min(py2,by2)-max(py1,by1))
                    if inter > best_iou:
                        best_iou = inter
                        best_stick_idx = i

            # --- Extract grip & tip ---
            if result.keypoints is not None and len(result.keypoints) > 0:
                kpts = result.keypoints[best_stick_idx].data[0]
                grip_point = (int(kpts[0][0]), int(kpts[0][1]))
                tip_point  = (int(kpts[1][0]), int(kpts[1][1]))

                # Apply temporal smoothing (video mode only)
                if not skip_smoothing:
                    smoothed = self._smooth_stick_keypoints(grip_point, tip_point)
                    if smoothed:
                        grip_point, tip_point = smoothed
                return (grip_point, tip_point), stick_bbox
        except Exception as e:
            print(f"[ERROR-STICK] Detection failed: {e}")
            return None, None

    def process_frame(self, frame, skip_ml_inference=False,
                      skip_stick_detection=False, mode='snapshot'):
        """Full pipeline for one frame: detect person → detect stick
        → extract features → run GCN → return results per zone."""
        # (Implementation: runs MediaPipe, YOLO person tracking,
        #  YOLO stick detection, feature extraction, GCN inference,
        #  and returns structured results for each detected person.)
        ...
```

> **Performance note**: `model_complexity=1` in MediaPipe balances accuracy and speed. On the target hardware (Intel Core i7-150U CPU), the full pipeline achieves ~10 FPS. YOLO stick detection is the bottleneck (~60 ms/frame); MediaPipe runs at ~30 ms/frame.

---

## 5. Feedback Generation

**File**: `app/computer_vision/feedback_analyzer.py`  
**Role**: Converts raw pose results into human-readable corrective instructions.

### 5.1 Feedback Analysis Pipeline

```python
class FeedbackAnalyzer:
    def __init__(self):
        # Per-viewpoint confidence thresholds
        self.confidence_thresholds = self._load_confidence_thresholds()

        # Joint angle targets per form: {form: {joint: (min_deg, max_deg, importance)}}
        self.joint_angle_targets = self._initialize_joint_targets()

    def analyze(self, result, target_form, confidence_threshold=None,
                viewpoint='front', gcn_engine=None):
        """Analyze pose result and generate prioritized feedback.
        Returns dict with errors, warnings, suggestions, and severity."""
        errors, warnings, suggestions, corrections = [], [], [], []
        used_hybrid = False

        # --- 1. Hybrid-based corrections (preferred, if available) ---
        if gcn_engine and hasattr(gcn_engine, 'get_feature_corrections'):
            correction_data = gcn_engine.get_feature_corrections(
                result, target_form, viewpoint
            )
            if correction_data:
                raw_corrections = get_corrections(
                    raw_features=correction_data['raw_values'],
                    hybrid_scores=correction_data['hybrid_scores'],
                    feature_names=correction_data['feature_names'],
                    template_means=correction_data['template_means'],
                    max_corrections=3,
                )
                for msg, score in raw_corrections:
                    errors.append(msg)
                    corrections.append({
                        'joint': 'body', 'action': 'adjust',
                        'value': round(1.0 - score, 2), 'message': msg
                    })
                used_hybrid = True

        # --- 2. Fallback: joint-angle heuristic analysis ---
        if not used_hybrid:
            joint_errs, joint_corrs = self._analyze_joint_angles(result, target_form)
            errors.extend(joint_errs)
            corrections.extend(joint_corrs)
            grip_errs, grip_corrs = self._analyze_grip_angle(result, target_form)
            errors.extend(grip_errs)
            corrections.extend(grip_corrs)

        # --- 3. Always-on checks ---
        body_warning = self._check_body_visibility(result)
        if body_warning:
            warnings.insert(0, body_warning)
        else:
            warnings.extend(self._analyze_posture(result))
        warnings.extend(self._analyze_stick_detection(result))
        suggestions.extend(self._analyze_confidence(
            result.get('predicted_class'), target_form, result.get('confidence')
        ))

        # Severity classification
        error_count = len(errors)
        warning_count = len(warnings)
        if error_count == 0 and warning_count == 0:
            severity = 'ok'
        elif error_count == 0:
            severity = 'minor'
        elif error_count <= 2:
            severity = 'major'
        else:
            severity = 'critical'

        return {
            'errors': errors,
            'corrections': corrections,
            'warnings': warnings,
            'suggestions': suggestions,
            'error_count': error_count,
            'severity': severity
        }
```

> **Two-tier feedback**: The analyzer first attempts **template-based corrections** (comparing live features against stored reference templates via the GCN engine). If templates are unavailable, it falls back to **hardcoded joint-angle heuristics** with form-specific target ranges.

---

### 5.2 Joint-Angle Heuristic Rules

```python
    def _analyze_joint_angles(self, result, target_form):
        """Compare live joint angles against target ranges for the form."""
        errors = []
        corrections = []
        live_angles = result.get('live_angles', {})
        targets = self.joint_angle_targets.get(target_form, {})

        for joint, (target_min, target_max, importance) in targets.items():
            if joint not in live_angles:
                continue
            angle = live_angles[joint]
            if angle < target_min:
                msg = self._joint_direction(joint, 'extend')
                errors.append(msg)
                corrections.append({
                    'joint': joint, 'action': 'extend',
                    'value': target_min - angle, 'message': msg
                })
            elif angle > target_max:
                msg = self._joint_direction(joint, 'flex')
                errors.append(msg)
                corrections.append({
                    'joint': joint, 'action': 'flex',
                    'value': angle - target_max, 'message': msg
                })
        return errors, corrections

    @staticmethod
    def _joint_direction(joint_name, action):
        """Convert joint + action into natural language."""
        parts = joint_name.split('_')
        side = parts[0].title() if parts else ''
        region = parts[1] if len(parts) > 1 else joint_name

        if region == 'elbow':
            return f"Straighten your {side} elbow" if action == 'extend' \
                   else f"Bend your {side} elbow more"
        elif region == 'shoulder':
            return f"Raise your {side} arm higher" if action == 'extend' \
                   else f"Lower your {side} arm"
        elif region == 'knee':
            return f"Straighten your {side} knee" if action == 'extend' \
                   else f"Bend your {side} knee more"
```

> **Example output**: If the user performing "Left Chest Thrust" has a left elbow angle of 140° (target: 160–180°), the system emits: `"Straighten your left elbow"`.

---

### 5.3 Grip Angle Analysis

```python
    def _analyze_grip_angle(self, result, target_form):
        """Analyze stick grip angle against form-specific ranges."""
        errors = []
        corrections = []
        grip_angle = result.get('grip_angle')
        if grip_angle is None:
            return errors, corrections

        # Form-specific ranges
        if 'thrust' in target_form:
            target_min, target_max = (85, 115)
        elif 'block' in target_form:
            target_min, target_max = (75, 125)
        else:
            target_min, target_max = (80, 120)

        if grip_angle < target_min:
            errors.append("Bring stick grip closer to you")
            corrections.append({
                'joint': 'wrist', 'action': 'extend_grip',
                'value': target_min - grip_angle,
                'message': "Bring stick grip closer to you"
            })
        elif grip_angle > target_max:
            errors.append("Extend stick grip outward")
            corrections.append({
                'joint': 'wrist', 'action': 'retract_grip',
                'value': grip_angle - target_max,
                'message': "Extend stick grip outward"
            })
        return errors, corrections
```

> **Grip angle**: Defined as the angle between the stick and the forearm. Thrusts require a more acute angle (85–115°) for proper striking mechanics; blocks allow a wider range (75–125°) for defensive coverage.

---

## Appendix: File Index

| File | Lines | Thesis Relevance |
|------|-------|-----------------|
| `scripts/4c_train_hybrid_gcn_v2_optimized.py` | 628 | Training loop, hyperparameter tuning, class imbalance handling |
| `app/models/gcn/model_v5.py` | 170 | Deployment model architecture (GCN + hybrid fusion) |
| `app/models/gcn/feature_extraction.py` | 546 | Pose/stick detection, 46-dim hybrid feature computation |
| `app/computer_vision/gcn_inference.py` | 141 | Public inference API, backwards-compatible wrapper |
| `app/deployment/viewpoint_engine.py` | 565 | Per-viewpoint engine, V5/V6 auto-detection |
| `app/computer_vision/pose_analyzer.py` | 855 | Real-time pipeline: MediaPipe → YOLO → GCN |
| `app/computer_vision/feedback_analyzer.py` | 541 | Form correction logic, heuristic rules, severity scoring |

---

*End of Source Code Reference*
