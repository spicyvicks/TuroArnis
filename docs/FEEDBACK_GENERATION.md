# Feedback Generation in TuroArnis Kiosk

## Overview
The TuroArnis Kiosk generates feedback through a multi-stage pipeline that combines computer vision, machine learning (GCN models), and rule-based analysis.

---

## Feedback Pipeline

### 1. **Snapshot Capture** (app.py)
```
User in position → Countdown ends → SNAP! → Frame captured
```
- When countdown reaches 0, `capture_snapshot()` is triggered
- Current camera frame is frozen for analysis
- This frozen frame is sent to the analysis pipeline

### 2. **Zone-Based Analysis** (app.py → analyze_zones)
```
Frozen Frame → Split into zones (1-3 users) → Analyze each zone independently
```

For each user zone:
```python
def analyze_zones(self, frame):
    col_w = w // self.num_users
    for i in range(self.num_users):
        zone_frame = frame[:, x_start:x_end]  # Crop zone
        # Set viewpoint model (front/left/right)
        viewpoint = self.user_configs[i]['viewpoint'].get().lower()
        self.pose_analyzer.gcn_engine.set_viewpoint(viewpoint)
        # Analyze
        results = self.pose_analyzer.process_frame(zone_frame)
```

### 3. **Pose Detection** (pose_analyzer.py)

#### Step 3a: Person Detection
- **YOLOv8n** detects people in the frame
- **ByteTrack** tracks person across frames
- Bounding box extracted for each person

#### Step 3b: Pose Keypoint Extraction
- **MediaPipe Pose** extracts 33 body keypoints:
  - Head: nose, eyes, ears
  - Upper body: shoulders, elbows, wrists
  - Core: hips
  - Lower body: knees, ankles
- Each keypoint: `[x, y, z, visibility]`

#### Step 3c: Stick Detection
- **YOLO Stick Detector** (best.pt) finds arnis stick
- Extracts 2 keypoints: **grip** and **tip**
- Each keypoint: `[x, y, confidence]`

### 4. **Feature Extraction** (gcn/feature_extraction.py)

#### Node Features (35 nodes × 6 features)
```python
def extract_node_features(pose_keypoints, stick_keypoints):
    # Combine: 33 body + 2 stick = 35 nodes
    for each node:
        x, y, z                    # 3D position
        visibility                 # Confidence
        distance_to_hip_3d         # Distance from hip center
        angle_from_hip             # Angular position
```

#### Global Features (30 geometric measurements)
```python
def compute_global_features_from_kpts(kpts, stick_keypoints):
    # Joint angles
    - left_elbow_angle, right_elbow_angle
    - left_shoulder_angle, right_shoulder_angle
    - left_knee_angle, right_knee_angle
    
    # Heights (relative to hip)
    - left_wrist_height, right_wrist_height
    - stick_tip_height, stick_grip_height
    
    # Horizontal positions
    - left_wrist_x, right_wrist_x
    - stick_tip_x, stick_grip_x
    
    # Stick orientation
    - stick_angle, stick_dx, stick_dy
    
    # Expert features
    - tip_vs_nose, tip_vs_shoulder, tip_vs_hip
    - r_hand_vs_nose, r_hand_vs_shoulder
    - foot_stagger
    
    # Distances
    - hands_distance, stick_length
```

#### Hybrid Features (30 similarity scores)
```python
def compute_hybrid_features(raw_features, templates, viewpoint, class_name):
    # Compare each feature to reference template
    for feature_name, feature_value:
        mean = template[feature_name]['mean']
        std = template[feature_name]['std']
        similarity = gaussian_similarity(feature_value, mean, std)
    # Returns 30 similarity scores (0-1)
```

### 5. **GCN Classification** (gcn_inference.py)

#### Model Architecture
```
Input:
  - Node features: [35 nodes, 6 features]
  - Hybrid features: [30 similarity scores]
  - Graph edges: Skeleton connections

Processing:
  1. Node embeddings (learn node-specific patterns)
  2. GCN layers (3 layers, 64 hidden dims)
     - Aggregate info from neighbor nodes
     - Learn spatial relationships
  3. Global pooling (graph → vector)
  4. Hybrid MLP (process similarity features)
  5. Fusion (combine GCN + hybrid features)
  6. Classification head → 13 classes

Output:
  - predicted_class: "left_temple_block_correct"
  - confidence: 0.87
  - all_probabilities: [13 values]
```

#### Viewpoint-Specific Models
```python
# 3 specialist models loaded:
models = {
    'front': hybrid_gcn_v2_front.pth   (85% accuracy)
    'left':  hybrid_gcn_v2_left.pth    (82% accuracy)
    'right': hybrid_gcn_v2_right.pth   (83% accuracy)
}
```

Each model is expert in its viewpoint, trained on thousands of examples.

### 6. **Feedback Generation** (app.py → show_feedback)

#### Step 6a: Retrieve GCN Results
```python
zone_result = self.analysis_results.get(i, {})
predicted_class = zone_result.get('predicted_class', 'N/A')
confidence = zone_result.get('confidence', 0.0)
stick_detected = zone_result.get('stick_detected', False)
```

#### Step 6b: Compare with Target
```python
# Map UI name to class name
class_name_mapping = {
    'Pugay': 'neutral_stance',
    'Left Temple Block': 'left_temple_block_correct',
    'Right Temple Block': 'right_temple_block_correct',
}

target_pose = config['form'].get()  # What user selected
expected_class = class_name_mapping.get(target_pose)

# Check if correct
is_correct = (predicted_class == expected_class) and (confidence > 0.6)
```

#### Step 6c: Generate Feedback Message
```python
if predicted_class == 'N/A':
    feedback_msg = "No pose detected"
elif is_correct:
    feedback_msg = "Perfect form!"  # Green ✓
else:
    feedback_msg = f"Detected: {predicted_class}"  # Yellow warning
```

#### Step 6d: Visual Display
```
┌─────────────────────────┐
│  User Name              │
│  87%  (big number)      │ ← Confidence score
│  Perfect form!          │ ← Feedback message
└─────────────────────────┘
```

Color coding:
- **Green (SUCCESS)**: Correct pose with confidence > 60%
- **Yellow (WARNING)**: Wrong pose or low confidence

### 7. **Database Storage** (db_manager.py)
```python
self.db.save_performance(
    session_id=config['session_id'],
    user_id=config['user']['id'],
    pose_detected=predicted_class,        # Actual GCN prediction
    confidence=confidence,                # Real confidence (0-1)
    is_correct=is_correct,               # Boolean match
    stick_detected=stick_detected        # Stick found or not
)
```

---

## Summary Flow Diagram

```
Camera Frame
    │
    ├─> YOLOv8n Person Detection
    │       │
    │       └─> Person Bounding Box
    │
    ├─> MediaPipe Pose
    │       │
    │       └─> 33 Body Keypoints
    │
    └─> YOLO Stick Detector
            │
            └─> 2 Stick Keypoints (grip, tip)

Combined Keypoints (35 total)
    │
    ├─> Node Features (35×6)
    │       - x, y, z, vis, dist, angle
    │
    ├─> Global Features (30)
    │       - angles, heights, distances
    │
    └─> Hybrid Features (30)
            - similarity to reference templates

All Features → GCN Model (Front/Left/Right)
    │
    ├─> GCN Layers (graph convolution)
    ├─> Hybrid MLP (similarity processing)
    └─> Fusion + Classification
            │
            └─> 13-Class Prediction

Prediction + Target → Feedback
    │
    ├─> Match? → "Perfect form!" (Green)
    └─> No match → "Detected: X" (Yellow)

Feedback + Data → Database
```

---

## Example Session

**User selects**: "Left Temple Block" | **Viewpoint**: "Front"

1. **Countdown**: 5...4...3...2...1...SNAP!
2. **Capture**: Frame frozen
3. **Detection**:
   - Person detected ✓
   - 33 pose keypoints extracted ✓
   - Stick detected ✓ (grip + tip)
4. **Feature Extraction**:
   - Node features: [35, 6] ✓
   - Global features: 30 values ✓
   - Hybrid features: 30 similarities ✓
5. **GCN Inference** (Front model):
   - Input → GCN processing → Output
   - **Predicted**: `left_temple_block_correct`
   - **Confidence**: `0.87` (87%)
6. **Comparison**:
   - Target: `left_temple_block_correct` ✓
   - Predicted: `left_temple_block_correct` ✓
   - Match: `True` ✓
7. **Feedback Display**:
   ```
   John Doe
   87%
   Perfect form!
   ```
   Color: GREEN
8. **Database Save**:
   ```json
   {
     "pose_detected": "left_temple_block_correct",
     "confidence": 0.87,
     "is_correct": true,
     "stick_detected": true
   }
   ```

---

## Confidence Score Meaning

| Range | Meaning | Feedback |
|-------|---------|----------|
| 90-100% | Excellent execution | "Perfect form!" |
| 70-89% | Good execution | "Perfect form!" |
| 60-69% | Acceptable (threshold) | "Perfect form!" |
| 40-59% | Needs improvement | "Detected: X" (if wrong class) |
| 0-39% | Poor detection | "No pose detected" |

Threshold is **60%** - below this, poses are marked as incorrect even if class matches.

---

## Why GCN for Arnis?

**Graph structure** is ideal because:
1. **Spatial relationships** matter (arm relative to head, stick relative to hands)
2. **Skeleton is naturally a graph** (joints connected by bones)
3. **Context-aware** (left elbow angle affects left shoulder interpretation)
4. **Rotation-invariant** (graph structure doesn't change with camera angle)

Traditional CNNs treat image as pixels. GCNs treat pose as **connected structure**.

---

## Improving Feedback Quality

### Current Implementation:
- Binary: "Perfect" or "Wrong"
- Shows detected class when wrong

### Future Enhancements:
1. **Detailed Corrections**:
   ```
   "Raise your left arm 15° higher"
   "Stick angle should be 45° not 30°"
   "Widen your stance by 10cm"
   ```

2. **Component Scoring**:
   ```
   Stance:     ★★★★☆ (80%)
   Arm Position: ★★★★★ (95%)
   Stick Angle:  ★★★☆☆ (60%)
   ```

3. **Progressive Feedback**:
   ```
   Attempt 1: "Move stick higher"
   Attempt 2: "Better! Now rotate 10° left"
   Attempt 3: "Perfect!"
   ```

4. **Video Replays**: Show side-by-side with expert reference

---

## Technical Notes

- **FPS**: ~10 FPS (bottleneck is YOLO detection)
- **Latency**: ~110ms per frame (GCN is only ~5ms)
- **Accuracy**: 75-85% depending on viewpoint
- **Classes**: 13 Arnis poses (can be expanded)
- **Dataset**: Trained on thousands of labeled Arnis poses

---

For more details, see:
- Model Architecture: [model_architecture.py](../app/models/gcn/model_architecture.py)
- Feature Extraction: [feature_extraction.py](../app/models/gcn/feature_extraction.py)
- GCN Inference: [gcn_inference.py](../app/computer_vision/gcn_inference.py)
- Main App: [app.py](../app/app.py)
