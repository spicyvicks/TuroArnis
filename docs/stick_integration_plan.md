# Implementation Plan: Stick Detection Integration

Integrate YOLOv8 stick keypoint features with existing MediaPipe pose features for training RF/XGBoost models.

## Background

The TuroArnis system currently:
- Uses MediaPipe to extract 54 pose features (angles, positions, distances, symmetry)
- Trains RF/XGBoost models on these features for pose classification
- Has a **separate** YOLOv8 stick detector (already trained at `runs/pose/arnis_stick_detector/`) that detects stick grip and tip points
- Uses stick detection only during **inference** (real-time app) but **not during training**

**Goal**: Add stick features to the training pipeline so models learn from both body pose and stick position/orientation.

## Proposed Changes

### 1. Feature Extraction Enhancement

#### [MODIFY] [feature_extraction.py](file:///c:/Users/HP/Documents/GitHub/TuroArnis/training/feature_extraction.py)

Add stick feature extraction alongside existing MediaPipe pose extraction:

**New function**: `extract_stick_features_from_image(image_path, stick_model_path)`
- Load YOLOv8 stick detector
- Detect stick grip and tip keypoints  
- Extract 10 stick features:
  1. `stick_grip_x` - Grip X coordinate (normalized)
  2. `stick_grip_y` - Grip Y coordinate (normalized)
  3. `stick_tip_x` - Tip X coordinate (normalized)
  4. `stick_tip_y` - Tip Y coordinate (normalized)
  5. `stick_angle` - Stick orientation angle (degrees)
  6. `stick_length` - Distance between grip and tip
  7. `grip_to_wrist_dist` - Distance from grip to nearest wrist
  8. `tip_to_shoulder_dist` - Distance from tip to nearest shoulder
  9. `stick_to_body_angle` - Angle between stick and torso vertical
  10. `stick_confidence` - Detection confidence score

**Updated function**: `extract_combined_features_from_image(image_path, stick_model_path)`
- Calls `extract_angles_from_image()` → 54 pose features
- Calls `extract_stick_features_from_image()` → 10 stick features
- Concatenates into **64-feature vector**
- Returns `None` if either pose or stick detection fails

**Updated function**: `extract_features_from_dataset(..., use_stick=False, stick_model_path=None)`
- Add parameters for stick detection
- Support three modes:
  - `use_stick=False`: Original 54 pose features only (backward compatible)
  - `use_stick=True`: Combined 64 features (pose + stick)
- Use multiprocessing with stick model loading in worker processes
- Save to separate CSV: `arnis_poses_angles_with_stick.csv`

---

### 2. Training Script Updates

#### [MODIFY] [training.py](file:///c:/Users/HP/Documents/GitHub/TuroArnis/training/training.py)

Update training script to support stick-enhanced features:

- Add command-line argument: `--use-stick` flag
- Add command-line argument: `--stick-model <path>` (default: `runs/pose/arnis_stick_detector/weights/best.pt`)
- When `--use-stick` is enabled:
  - Use `arnis_poses_angles_with_stick.csv` for training
  - Pass 64 features to model training
  - Save model with suffix indicating stick usage (e.g., `v042_random_forest_with_stick`)
- Update metadata to record stick feature usage

#### [MODIFY] [model_manager.py](file:///c:/Users/HP/Documents/GitHub/TuroArnis/training/model_manager.py)

- Add `uses_stick_features` field to model metadata
- Store expected feature count (54 or 64) in metadata
- Display stick feature usage in model list view

---

### 3. Application Runtime Support (Future - Not in this phase)

> [!NOTE]
> The real-time application already has stick detection integrated in `pose_analyzer.py`. However, it currently extracts stick features **separately** from the classifier features. In a future update, we'll need to modify the inference pipeline to pass stick features to the classifier if the model was trained with stick support.

**Future work** (not implemented now):
- Modify `pose_analyzer.py` to extract same 10 stick features during inference
- Check model metadata for `uses_stick_features` flag
- If true, append stick features to pose features before classification
- Maintain backward compatibility with non-stick models

---

## Verification Plan

### Automated Tests

> [!WARNING]
> No existing unit tests were found for the feature extraction pipeline. Manual verification will be required.

### Manual Verification Steps

#### Step 1: Feature Extraction Test

Run feature extraction with stick detection on sample dataset:

```bash
cd c:\Users\HP\Documents\GitHub\TuroArnis
python -c "from training.feature_extraction import extract_features_from_dataset; extract_features_from_dataset('dataset', 'arnis_poses_angles_with_stick.csv', feature_mode='angles', use_stick=True, stick_model_path='runs/pose/arnis_stick_detector/weights/best.pt')"
```

**Expected outcome**:
- CSV file `arnis_poses_angles_with_stick.csv` created
- File contains 65 columns (1 class + 64 features)
- Column headers include stick feature names
- Row count matches non-stick CSV (same number of valid samples)

**Validation**:
```bash
# Check CSV shape
python -c "import pandas as pd; df = pd.read_csv('arnis_poses_angles_with_stick.csv'); print(f'Shape: {df.shape}'); print(f'Columns: {list(df.columns)}')"
```

---

#### Step 2: Model Training Test

Train Random Forest model with combined features:

```bash
cd c:\Users\HP\Documents\GitHub\TuroArnis\training
python training.py --model-type random_forest --use-stick --stick-model ../runs/pose/arnis_stick_detector/weights/best.pt --csv ../arnis_poses_angles_with_stick.csv
```

**Expected outcome**:
- Model trains successfully on 64 features
- Accuracy comparable to or better than pose-only models
- Model saved with descriptive name (e.g., `v042_random_forest_with_stick`)
- Metadata includes `uses_stick_features: true`

**Validation**:
```bash
# Check model metadata
python -c "import json; import os; versions = sorted([d for d in os.listdir('c:/Users/HP/Documents/GitHub/TuroArnis/models') if d.startswith('v')]); latest = versions[-1]; meta = json.load(open(f'c:/Users/HP/Documents/GitHub/TuroArnis/models/{latest}/metadata.json')); print(json.dumps(meta, indent=2))"
```

---

#### Step 3: Ensemble Creation Test

Create ensemble combining stick-enabled RF and XGBoost:

```bash
cd c:\Users\HP\Documents\GitHub\TuroArnis\training
python ensemble_model.py
# Select option 1 (create ensemble)
# Choose models trained with stick features
# Use soft voting, optimize weights
```

**Expected outcome**:
- Ensemble combines stick-enabled models successfully
- Performance metrics displayed
- Ensemble model saved

---

#### Step 4: Feature Comparison (Optional)

Compare model performance with and without stick features:

```bash
# Train without stick (baseline)
python training.py --model-type random_forest --csv ../arnis_poses_angles.csv

# Train with stick (enhanced)
python training.py --model-type random_forest --use-stick --csv ../arnis_poses_angles_with_stick.csv
```

**Analysis**: Compare test accuracies to quantify improvement from stick features.

---

## Notes

- Backward compatibility maintained: existing models continue to work with 54 features
- Stick model must be trained before using this feature (already available at `runs/pose/arnis_stick_detector/weights/best.pt`)
- If stick detection fails on specific images, those samples are excluded (same as current pose detection failure handling)
- Feature normalization handled by existing scaler in training pipeline
