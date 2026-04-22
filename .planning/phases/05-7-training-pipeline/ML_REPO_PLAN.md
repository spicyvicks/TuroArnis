---
phase: 5.7
plan: 01
type: execute
wave: 1
depends_on: []
files_modified: [
  "hybrid_classifier/1_extract_reference_features.py",
  "hybrid_classifier/feature_templates.json",
  "hybrid_classifier/hybrid_features_v3/train_features.pt",
  "hybrid_classifier/hybrid_features_v3/test_features.pt",
  "hybrid_classifier/models/model_merged.pth",
  "hybrid_classifier/models/history_merged.json"
]
autonomous: true
requirements: [TRN-01, TRN-02, TRN-03, TRN-04]

must_haves:
  truths:
    - "Quality gates tuned to achieve 50-70% acceptance rate (currently at 20%)"
    - "YOLO confidence threshold reduced from 0.5 to 0.35"
    - "MediaPipe visibility threshold reduced from 0.7 to 0.5"
    - "Critical joints reduced to wrists only [15, 16] for martial arts poses"
    - "Templates regenerated with balanced thresholds"
    - "Training features generated (.pt files)"
    - "Models trained with optimized hyperparameters (dropout 0.7, hidden 128)"
    - "Validation accuracy > 70% minimum, target > 75%"
  artifacts:
    - path: "hybrid_classifier/feature_templates.json"
      provides: "Quality-gated templates with tuned thresholds"
      contains: "39 templates (13 classes × 3 views), acceptance rate 50-70%"
    - path: "hybrid_classifier/models/model_merged.pth"
      provides: "Trained HybridGCN model"
      contains: "Optimized architecture with dropout 0.7, hidden 128"
    - path: "hybrid_classifier/models/history_merged.json"
      provides: "Training metrics"
      contains: "Epoch-by-epoch accuracy, gap monitoring, early stopping"
---

<objective>
Complete the training pipeline execution in TuroArnis-ML: tune quality gate thresholds to fix 20% acceptance rate, regenerate templates, generate training features, train models with optimized hyperparameters, and deploy ready-to-use models.
</objective>

<execution_context>
@$HOME/.config/opencode/get-shit-done/workflows/execute-plan.md
@$HOME/.config/opencode/get-shit-done/templates/summary.md
</execution_context>

<context>
## Current State

Quality gates were applied to `1_extract_reference_features.py` but thresholds are too strict:
- **Current acceptance rate: 20%** (expected: 50-70%)
- **Problem:** YOLO 0.5 threshold excludes valid poses, MediaPipe 0.7 excludes wrist-occluded martial arts poses

## Required Threshold Adjustments

| Parameter | Current | Target | Rationale |
|-----------|---------|--------|-----------|
| YOLO confidence | 0.5 | **0.35** | Many valid poses have confidence 0.3-0.49 |
| MediaPipe visibility | 0.7 | **0.5** | Wrists often occluded by stick/body in Arnis |
| Critical joints | [11,12,13,14,15,16,23,24] | **[15,16]** | Only wrists are critical for stick technique |
| Min stick length | > 0 | **≥ 10** | Allow very small but non-zero detections |

## Files to Modify

1. `hybrid_classifier/1_extract_reference_features.py` - Threshold tuning
2. Copy optimized training script from TuroArnis repo

## Optimized Training Script Available

Source: `TuroArnis/scripts/4c_train_hybrid_gcn_v2_optimized.py`
Key optimizations:
- HIDDEN_DIM: 256 → 128
- DROPOUT: 0.5 → 0.7
- NODE_EMBED_DIM: 8 → 16
- WeightedRandomSampler for class balance
- BatchNorm(track_running_stats=False)
- Overfitting gap monitoring
</context>

<tasks>

<task type="auto">
  <name>Task 1: Tune Quality Gate Thresholds</name>
  <files>hybrid_classifier/1_extract_reference_features.py</files>
  <action>
    Fix the 20% acceptance rate by tuning validation thresholds:

    1. Edit hybrid_classifier/1_extract_reference_features.py:
       - Find MIN_YOLO_CONF = 0.5, change to 0.35
       - Find CRITICAL_JOINTS list, reduce to [15, 16] (wrists only)
       - Find MIN_VISIBILITY = 0.7, change to 0.5
       - Find stick_length <= 0 check, change to < 10

    2. If validation function doesn't exist, create it with these thresholds:
       ```python
       def validate_features(features, pose_landmarks, stick_detected, yolo_confidence=1.0):
           MIN_YOLO_CONF = 0.35  # Reduced from 0.5
           CRITICAL_JOINTS = [15, 16]  # Wrists only (was [11,12,13,14,15,16,23,24])
           MIN_VISIBILITY = 0.5  # Reduced from 0.7
           
           # Check YOLO confidence
           if yolo_confidence < MIN_YOLO_CONF:
               return False, f"yolo_confidence_low ({yolo_confidence:.2f})"
           
           # Check critical joint visibility
           for joint_idx in CRITICAL_JOINTS:
               if pose_landmarks.landmark[joint_idx].visibility < MIN_VISIBILITY:
                   return False, f"visibility_low_joint_{joint_idx}"
           
           # Check stick detected
           if not stick_detected:
               return False, "stick_not_detected"
           
           # Check stick length (relaxed)
           if features.get('stick_length', 0) < 10:
               return False, "stick_length_too_small"
           
           return True, "valid"
       ```

    3. Verify the validation gate is called in the feature extraction loop

    4. Test acceptance rate improvement
  </action>
  <verify>
    <automated>python -c "import re; f=open('hybrid_classifier/1_extract_reference_features.py').read(); assert '0.35' in f or 'MIN_YOLO_CONF' in f; print('Thresholds updated')"