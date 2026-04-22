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
    <automated>python -c "import re; f=open('hybrid_classifier/1_extract_reference_features.py').read(); assert '0.35' in f or 'MIN_YOLO_CONF' in f; print('Thresholds updated')"</automated>
  </verify>
  <done>Quality gate thresholds tuned: YOLO 0.35, visibility 0.5, critical joints wrists only</done>
</task>

<task type="auto">
  <name>Task 2: Regenerate Templates with Tuned Thresholds</name>
  <files>hybrid_classifier/feature_templates.json</files>
  <action>
    Run the feature extraction script to regenerate templates with relaxed thresholds:

    1. Execute:
       ```bash
       cd C:\Users\HP\Documents\GitHub\TuroArnis-ML
       python hybrid_classifier\1_extract_reference_features.py
       ```

    2. Verify acceptance rate improved from 20% to 50-70%:
       - Check console output for "Accepted: X/Y (Z%)"
       - If still < 40%, lower thresholds further: YOLO 0.35 → 0.25, visibility 0.5 → 0.4

    3. Verify templates generated:
       - Check hybrid_classifier/feature_templates.json exists
       - Check file size is reasonable (should be ~200KB+)

    4. Log rejection breakdown to understand remaining rejects
  </action>
  <verify>
    <automated>python -c "import json; t=json.load(open('hybrid_classifier/feature_templates.json')); print(f'Templates: {len(t)}')"</automated>
  </verify>
  <done>Templates regenerated with 50-70% acceptance rate, 39 templates present</done>
</task>

<task type="auto">
  <name>Task 3: Generate Training Features</name>
  <files>hybrid_classifier/hybrid_features_v3/train_features.pt, hybrid_classifier/hybrid_features_v3/test_features.pt</files>
  <action>
    Generate PyTorch tensors from training images using quality-gated templates:

    1. Execute:
       ```bash
       python hybrid_classifier\2b_generate_node_hybrid_features.py
       ```

    2. Verify outputs created:
       - hybrid_classifier/hybrid_features_v3/train_features.pt
       - hybrid_classifier/hybrid_features_v3/test_features.pt

    3. Check tensor shapes are correct:
       - node_features: [N, 35, 6]
       - hybrid_features: [N, 30]
       - labels: [N]

    4. Note sample count N (will determine if class balance is needed)
  </action>
  <verify>
    <automated>Test-Path "hybrid_classifier/hybrid_features_v3/train_features.pt"</automated>
  </verify>
  <done>Training features generated: train_features.pt and test_features.pt ready</done>
</task>

<task type="auto">
  <name>Task 4: Copy Optimized Training Script</name>
  <files>hybrid_classifier/4c_train_hybrid_gcn_v2.py</files>
  <action>
    Copy the optimized training script from TuroArnis repo:

    1. Source file: TuroArnis/scripts/4c_train_hybrid_gcn_v2_optimized.py
    
    2. Copy to:
       ```bash
       copy "..\TuroArnis\scripts\4c_train_hybrid_gcn_v2_optimized.py" "hybrid_classifier\4c_train_hybrid_gcn_v2.py"
       ```
       Or adjust path based on your directory structure

    3. Verify the script has optimizations:
       - HIDDEN_DIM = 128 (line ~71)
       - DROPOUT = 0.7 (line ~73)
       - NODE_EMBED_DIM = 16 (line ~74)
       - WeightedRandomSampler implementation
       - BatchNorm(track_running_stats=False)

    4. Test import (no syntax errors):
       ```bash
       python -c "import hybrid_classifier.4c_train_hybrid_gcn_v2"
       ```
  </action>
  <verify>
    <automated>python -c "import sys; sys.path.insert(0, 'hybrid_classifier'); import importlib.util; spec = importlib.util.spec_from_file_location('train', 'hybrid_classifier/4c_train_hybrid_gcn_v2.py'); assert spec is not None; print('Script syntax OK')"</automated>
  </verify>
  <done>Optimized training script copied and verified</done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 5: Train Models</name>
  <what-built>
    All preprocessing complete:
    - Quality gates tuned (acceptance rate 50-70%)
    - Templates regenerated with tight STDs
    - Training features generated (.pt files)
    - Optimized training script ready
  </what-built>
  <how-to-verify>
    Train the HybridGCN models with optimized hyperparameters:

    ```bash
    cd C:\Users\HP\Documents\GitHub\TuroArnis-ML
    python hybrid_classifier\4c_train_hybrid_gcn_v2.py --merged
    ```

    Expected training output:
    - Total parameters: ~185,000
    - Hidden dim: 128, Dropout: 0.7
    - Xavier initialization applied
    - WeightedRandomSampler for class balance
    - Training 150 epochs max (early stopping patience 15)
    - Gap monitoring (stops if train-test gap > 25%)

    Monitor for:
    - Training accuracy should increase steadily
    - Validation accuracy should track within 10-15% of training
    - No severe overfitting (gap stays < 25%)
  </how-to-verify>
  <resume-signal>
    Training completed. Report:
    - Best validation accuracy: ___%
    - Epochs trained: ___
    - Final train-test gap: ___%
    - Model saved: hybrid_classifier/models/model_merged.pth
    
    Decision: PASS (if ≥70%) / RETRY (if 60-70%) / INVESTIGATE (if <60%)
  </resume-signal>
</task>

<task type="auto">
  <name>Task 6: Copy Models to TuroArnis App</name>
  <files>../TuroArnis/app/models/gcn/model_merged.pth, ../TuroArnis/app/models/gcn/feature_templates.json</files>
  <action>
    Deploy trained models and templates to TuroArnis application:

    1. Copy model:
       ```bash
       copy "hybrid_classifier\models\model_merged.pth" "..\TuroArnis\app\models\gcn\"
       ```

    2. Copy quality-gated templates:
       ```bash
       copy "hybrid_classifier\feature_templates.json" "..\TuroArnis\app\models\gcn\"
       ```

    3. Verify files in TuroArnis:
       - app/models/gcn/model_merged.pth (should be ~2-3MB)
       - app/models/gcn/feature_templates.json (should be ~200KB)

    4. Update template metadata to mark as quality-gated
  </action>
  <verify>
    <automated>Test-Path "../TuroArnis/app/models/gcn/model_merged.pth"</automated>
  </verify>
  <done>Models and templates deployed to TuroArnis app</done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <name>Task 7: Integration Test in TuroArnis</name>
  <what-built>
    Complete training pipeline with:
    - Tuned quality gates (50-70% acceptance)
    - Regenerated templates with relaxed thresholds
    - Trained HybridGCN model (dropout 0.7, hidden 128)
    - Models deployed to app/models/gcn/
  </what-built>
  <how-to-verify>
    Test classification in TuroArnis application:

    ```bash
    cd C:\Users\HP\Documents\GitHub\TuroArnis
    python app\test_classification.py
    ```

    Or run full application:
    ```bash
    python app\app.py
    ```

    Test scenarios:
    1. Free Mode: Perform thrusts and blocks - should classify correctly (>70% accuracy)
    2. Lesson Mode: Start any lesson, check similarity scores make sense
    3. Neutral class: Stand still/walk around - should trigger neutral (not false positive)
    4. Stability: Run for 2-3 minutes without crashes

    Pay attention to:
    - Thrust vs block discrimination
    - Per-viewpoint accuracy (front/left/right)
    - Neutral false positive rate
  </how-to-verify>
  <resume-signal>
    Integration test result: PASS / ISSUES FOUND
    
    If issues found, describe:
    - Specific techniques misclassified
    - Confidence scores observed
    - UI/UX problems
    - Performance issues
  </resume-signal>
</task>

</tasks>

<success_criteria>
## Success Criteria

### Threshold Tuning
- [ ] YOLO confidence: 0.35 (reduced from 0.5)
- [ ] MediaPipe visibility: 0.5 (reduced from 0.7)
- [ ] Critical joints: wrists only [15, 16]
- [ ] Template acceptance rate: 50-70%

### Model Training
- [ ] Training features generated (.pt files)
- [ ] Model trained with optimized hyperparameters:
  - Hidden dim: 128
  - Dropout: 0.7
  - Node embed: 16
  - Weight decay: 1e-4
  - WeightedRandomSampler active
- [ ] Validation accuracy: ≥70% (target: >75%)
- [ ] Train-test gap: <15%

### Deployment
- [ ] Model file copied to TuroArnis app/models/gcn/
- [ ] Templates copied with version metadata
- [ ] Integration test passes (classification works)

### Completion
- [ ] Training pipeline fully executed
- [ ] Models ready for Phase 5.6 and Phase 6
</success_criteria>

<output>
After completion, create this summary in TuroArnis:
`.planning/phases/05-7-training-pipeline/05-7-01-SUMMARY.md`

Include:
1. Quality gate threshold adjustments and acceptance rate achieved
2. Template statistics (before/after if available)
3. Model training results (accuracy, epochs, final gap)
4. Any issues encountered and resolutions
5. Next steps: Phase 5.6 (Lesson Similarity) or Phase 6 (Packaging)
</output>