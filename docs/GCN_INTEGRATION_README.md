# GCN Integration in TuroArnis Kiosk App

## Overview
This document describes the integration of Hybrid GCN V2 specialist models into the TuroArnis Kiosk application for real-time Arnis pose classification.

## Architecture

### Components Integrated
1. **GCN Inference Engine** (`app/computer_vision/gcn_inference.py`)
   - Manages loading and inference for 3 specialist models (front, left, right)
   - Handles viewpoint switching
   - Performs real-time pose classification

2. **Feature Extraction** (`app/models/gcn/feature_extraction.py`)
   - Extracts node features (35 nodes x 6 features)
   - Computes global geometric features (joint angles, heights, distances)
   - Calculates hybrid similarity features using templates

3. **Model Architecture** (`app/models/gcn/model_architecture.py`)
   - HybridGCN neural network
   - Graph Convolutional Network with node embeddings
   - Fuses graph features with global hybrid features

4. **PoseAnalyzer Integration** (`app/computer_vision/pose_analyzer.py`)
   - Seamlessly integrates GCN with existing pose analysis pipeline
   - Falls back to legacy models if GCN unavailable
   - Handles MediaPipe pose detection + YOLO stick detection

5. **Kiosk App Integration** (`app/app.py`)
   - Multi-zone pose analysis (up to 3 users)
   - Viewpoint-specific model selection
   - Real-time feedback with GCN predictions

## Model Files

### Required Files
```
deployment_package/models/
├── hybrid_gcn_v2_front.pth   (Front viewpoint specialist)
├── hybrid_gcn_v2_left.pth    (Left viewpoint specialist)
└── hybrid_gcn_v2_right.pth   (Right viewpoint specialist)

deployment_package/weights/
└── best.pt                    (YOLO stick detector)

app/models/
├── gcn_model_config.json      (Model configuration)
└── gcn/
    ├── feature_templates.json (Hybrid feature templates)
    ├── feature_extraction.py  (Feature computation)
    └── model_architecture.py  (GCN model definition)
```

## How It Works

### 1. Initialization
```python
# In app.py
self.pose_analyzer = PoseAnalyzer(
    detection_interval=3,
    stick_model_path='deployment_package/weights/best.pt',
    debug_stick=False
)
# PoseAnalyzer automatically loads GCN models on init
```

### 2. Frame Analysis
For each user zone:
1. Extract zone from camera frame
2. Set appropriate viewpoint model (front/left/right)
3. Run pose detection (MediaPipe + YOLO)
4. Extract features:
   - Node features: [35 nodes, 6 features] (x, y, z, visibility, distance, angle)
   - Global features: 30 geometric measurements (angles, heights, distances)
   - Hybrid features: 30 similarity scores vs. reference templates
5. Run GCN inference
6. Return predicted class + confidence

### 3. Feedback Display
```python
# Results include:
{
    'predicted_class': 'left_temple_block_correct',
    'confidence': 0.87,
    'stick_detected': True
}
```

## Viewpoint Models

### Front Viewpoint
- Best for frontal poses
- Optimal for: neutral stance, forward/back stances, frontal thrusts
- Accuracy: ~85%

### Left Viewpoint
- Best for left-side poses
- Optimal for: left blocks, left-side thrusts
- Accuracy: ~82%

### Right Viewpoint
- Best for right-side poses
- Optimal for: right blocks, right-side thrusts
- Accuracy: ~83%

## Class Names (13 Classes)
1. `neutral_stance`
2. `forward_stance_correct` (not in current UI mapping)
3. `left_temple_block_correct`
4. `right_temple_block_correct`
5. `left_chest_thrust_correct`
6. `right_chest_thrust_correct`
7. `left_eye_thrust_correct`
8. `right_eye_thrust_correct`
9. `left_elbow_block_correct`
10. `right_elbow_block_correct`
11. `left_knee_block_correct`
12. `right_knee_block_correct`
13. `crown_thrust_correct`
14. `solar_plexus_thrust_correct`

## Configuration

### UI Form to Class Mapping
```python
class_name_mapping = {
    'Pugay': 'neutral_stance',
    'Forward Stance': 'forward_stance_correct',
    'Left Temple Block': 'left_temple_block_correct',
    'Right Temple Block': 'right_temple_block_correct',
}
```

### GCN Model Config (`app/models/gcn_model_config.json`)
```json
{
    "model_type": "gcn_hybrid_v2",
    "models": {
        "front": {
            "path": "deployment_package/models/hybrid_gcn_v2_front.pth",
            "accuracy": 0.85
        },
        ...
    },
    "feature_templates": "app/models/gcn/feature_templates.json",
    "stick_detector": "deployment_package/weights/best.pt",
    "class_names": [...]
}
```

## Performance

### Inference Speed
- ~5ms per frame (CPU)
- Bottleneck: YOLO person detection (~100ms)
- Overall FPS: ~10 FPS for real-time video

### Memory Usage
- Models: ~50MB total
- Feature templates: ~2MB
- Runtime: ~200MB peak

## Testing

Run the integration test:
```bash
python test_gcn_integration.py
```

This will verify:
- [ ] All model files exist
- [ ] GCN engine loads successfully
- [ ] Feature extraction works
- [ ] Model inference produces valid predictions
- [ ] PoseAnalyzer integration is functional

## Troubleshooting

### Issue: "Model file not found"
**Solution**: Ensure all `.pth` files are in `deployment_package/models/`

### Issue: "Could not load GCN models"
**Solution**: Check PyTorch and torch-geometric installation:
```bash
pip install torch==2.1.0+cpu torch-geometric==2.4.0
```

### Issue: "No pose detected"
**Solution**: 
- Ensure person is fully visible in frame
- Check lighting conditions
- Verify camera is working

### Issue: Low confidence scores
**Solution**:
- Select correct viewpoint (front/left/right)
- Ensure user is performing target pose correctly
- Check stick detection is working

## Future Enhancements

1. **Add More Poses**: Train models for additional Arnis techniques
2. **Improve Stick Detection**: Fine-tune YOLO model for better accuracy
3. **Multi-Person Tracking**: Better handle overlapping zones
4. **Temporal Smoothing**: Average predictions over multiple frames
5. **Confidence Calibration**: Improve confidence score accuracy

## Dependencies
```
torch>=2.1.0
torch-geometric>=2.4.0
mediapipe>=0.10.14
ultralytics>=8.3.25
opencv-python>=4.8.0
numpy>=1.26.4
```

## References
- GCN Integration Plan: `docs/GCN_INTEGRATION_PLAN.md`
- Model Architecture: `app/models/gcn/model_architecture.py`
- Feature Templates: `app/models/gcn/feature_templates.json`
