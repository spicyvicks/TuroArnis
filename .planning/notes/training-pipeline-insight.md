---
title: Training Pipeline Insight — Visual Perfection ≠ Computational Validity
date: 2026-04-18
context: TuroArnis GCN classifier training pipeline
source: /gsd-explore training pipeline optimization
---

## Core Insight

Manual curation of training images captures human-judged "perfect form," but feature extraction failures can silently corrupt the computed representation. The training pipeline must validate the *extracted features*, not just the *visual appearance*.

## What We Observed

1. **Human Curation Process**
   - Manually selected "perfect textbook pose" images for each technique
   - Selected based on visual assessment of form correctness
   - Images were full-frame (not cropped), judged in context

2. **Template Generation Process**
   - Ran script to extract features from selected images
   - Computed mean/std/min/max statistics → `feature_templates.json`
   - **No verification that stick detection or pose estimation succeeded**

3. **Resulting Template Corruption**
   - `front_crown_thrust_correct.left_elbow_angle.std = 60.2°` (full 0-175° range considered valid)
   - `front_left_chest_thrust_correct.stick_angle.std = 116.4°` (effectively ±180° — completely uninformative)
   - Template statistics include `stick_length=0` as valid value

4. **Root Cause**
   - Silent YOLO stick detection failures on some "perfect" training images
   - Silent MediaPipe low-confidence joint detections
   - Template generation treated all extracted features as valid ground truth

## Why This Matters

The classifier compares user poses to these templates using Gaussian similarity scores. When templates have wide STDs:
- Any user pose gets near-1.0 similarity to all classes
- Model can't discriminate between similar techniques (thrusts vs. blocks)
- Confidence thresholds (0.70) become impossible to meet even for correct poses

Lesson mode failures are not just inference bugs — they're a **training data quality problem** cascading through the pipeline.

## Guiding Principle

> Ground truth must be valid at the representation layer, not just the pixel layer.

Future template generation must include:
1. **Detection validation**: Verify YOLO/Mediapipe succeeded before accepting features
2. **Physical plausibility**: Reject impossible values (zero stick length, 180° angles that should be bent)
3. **Tightened statistics**: Outlier rejection so template STDs reflect true class variance, not detection noise

## Related Artifacts

- Research: `.planning/research/questions.md` — validation criteria investigation
- Todo: `.planning/todos/pending/training-quality-gates.md` — implementation task
