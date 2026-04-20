---
title: Validation Criteria for Training Image Quality
date: 2026-04-18
context: Exploration session on training pipeline optimization for TuroArnis GCN classifier
source: /gsd-explore training pipeline optimization
---

## Research Question

What detection confidence thresholds, joint visibility requirements, and physical plausibility checks should gate template inclusion in the training pipeline?

## Background

Current template generation accepts all manually-curated "perfect" images without validating that feature extraction succeeded. This allows silent detection failures (stick undetected, low MediaPipe confidence, impossible feature values) to corrupt templates with near-zero stick lengths or missing joint data. These corrupted features widen template STDs and reduce discriminative power.

## Specific Unknowns to Investigate

1. **Stick Detection Reliability Thresholds**
   - What YOLO confidence threshold balances false rejection vs. false inclusion?
   - Does stick detection failure correlate with specific pose types (overhead thrusts, stick close to body)?

2. **MediaPipe Joint Visibility Requirements**
   - Which joints are critical for Arnis technique discrimination? (likely: wrists, elbows, shoulders)
   - What visibility confidence threshold ensures reliable angle calculations?
   - How many joints can be "low confidence" before rejecting the image?

3. **Physical Plausibility Checks**
   - What are valid ranges for `stick_length` given known Arnis stick dimensions?
   - Can we detect corrupted angles (e.g., 0° when joints are collinear but should be bent)?
   - Should there be coherence checks between body pose and stick position?

4. **Per-Technique Sensitivity**
   - Do different techniques have different critical joints? (e.g., knee blocks need visible legs)
   - Should quality gates vary by viewpoint (front vs. side)?

## Success Criteria

- Define quantitative acceptance criteria that would have filtered the corrupted templates in `feature_templates.json` (e.g., `stick_length=0` with claimed valid detection)
- Validate that applying these gates tightens template STDs by >30% for critical features (stick_angle, stick_length)
- Confirm that lesson mode classification improves on a holdout test set after regenerating templates with quality gates

## Research Approach

Analyze existing training images (both accepted and rejected by human curation) to:
1. Measure actual detection confidence distributions
2. Identify feature values that correlate with "obviously wrong" classifications
3. Propose thresholds that would have caught the current template corruption
