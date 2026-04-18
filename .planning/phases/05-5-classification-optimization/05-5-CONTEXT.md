# Phase Context: Classification Algorithm Optimization

**Phase:** Phase 5.5 — Classification Algorithm Optimization  
**Date:** 2026-04-15  
**Status:** Ready for Research & Planning  

---

## User's Goal

> """I need help ensuring that my inference is using the best algorithm to get the best classifications. I've been dissatisfied with it so far."""

**Specific Issue:** Thrust vs block confusion — the system incorrectly classifies thrusting techniques as blocking techniques and vice versa.

---

## Background: Current System

**Pipeline:** Camera ? YOLO person detect ? MediaPipe 33 kpts ? GCN inference

**Architecture:**
- HybridGCN model with 35 nodes (33 body + 2 stick)
- 3 specialist models (front, left, right viewpoints)
- 12 Arnis technique classes + neutral
- Gaussian similarity-based hybrid features (30 features)

**Current State:**
- FIX #1 implemented: Post-loop argmax verification in `gcn_inference.py`
- FIX #2 partially implemented: NaN sentinels for missing stick in `feature_extraction.py`
- FIX #6 partially implemented: Separate `pose_static` instance created in `pose_analyzer.py`

**Source Analysis:** `classification_failure_analysis.md` identified 7 root causes

---

## DECISIONS LOCKED

These decisions are final. Downstream agents MUST NOT re-ask the user about these choices.

### D1: Angle Dimensionality — USE 3D ANGLES

**Decision:** Upgrade `calculate_angle()` to use 3D coordinates [x, y, z]

**Rationale:** 
- Thrusts extend forward (toward camera) ? large z variation
- Blocks move sideways ? minimal z variation  
- In 2D projection from front view, these look identical
- 3D angles will better distinguish thrust vs block

**Implementation:**
- Modify `calculate_angle(p1, p2, p3)` in `feature_extraction.py`
- Use 3D vectors and proper 3D angle calculation
- Keep feature count the same (don't add 2D+3D, replace 2D with 3D)

---

### D2: Template Standard Deviations — CLAMP/CAP MAXIMUM

**Decision:** Apply maximum STD ceiling to all template features

**Rationale:**
- Current `front_crown_thrust_correct.left_elbow_angle` has std=60.2°, rendering it non-discriminative
- Gaussian with 60° std gives near-1.0 similarity for any angle 0°-175°
- Must tighten templates without requiring retraining

**Implementation:**
- Apply cap at load time in `GCNInferenceEngine._load_config()`
- Angle features: max std = 20°
- Normalized coordinate features: max std = 0.1
- Log warnings when caps are applied (so we know which features were too loose)

---

### D3: Stick Failure Handling — CONFIDENCE PENALTY

**Decision:** Apply confidence penalty factor when stick not detected

**Rationale:**
- NaN sentinel already implemented (FIX #2)
- Zero-filling features creates ambiguous signal
- Hard reject would be frustrating for users
- Body-only classification is better than no classification

**Implementation:**
- In `gcn_inference.py` predict() method
- Detect NaN/zeroed stick features
- Apply penalty factor: multiply final confidence by 0.7
- Still respect threshold, but lowered effective threshold for stickless poses

---

### D4: Confidence Threshold Strategy — DYNAMIC/ADAPTIVE

**Decision:** Compute threshold dynamically based on template similarity spread

**Rationale:**
- Uniform 0.70 threshold fails for geometrically similar classes
- Structurally similar classes (thrusts) need lower effective thresholds
- Adaptive approach handles this automatically

**Implementation:**
- After computing hybrid similarity scores for a hypothesis
- Calculate `score_variance` across all 30 features
- If variance is low (pose matches multiple templates closely), lower threshold
- Formula: `effective_threshold = base_threshold * (1 - 0.3 * (1 - score_variance))`
- Clamp between 0.45 and 0.70

---

### D5: MediaPipe Mode Verification — AUDIT CALL SITES

**Decision:** Review all `process_frame()` call sites to verify mode usage

**Rationale:**
- `pose_static` instance exists but may not be used correctly
- Video mode (temporal smoothing) causes """temporal bleed""" between lesson frames
- Lesson mode must use static mode

**Implementation:**
- Search codebase for all `process_frame()` calls
- Identify which mode each call needs
- Add inline comments documenting expected mode
- Add runtime check: `assert mode == expected_mode` during development
- Do NOT add explicit mode parameter (keep current design)

---

### D6: Phase Scope — ALGORITHM FIXES ONLY

**Decision:** Implement algorithmic fixes only. NO retraining, NO new data collection.

**Rationale:**
- Faster time to evaluation
- Can assess algorithm fix effectiveness before committing to retraining
- Retraining can be Phase 5.6 or Phase 8 (Performance Optimization) if needed

**Scope INCLUDES:**
- 3D angle calculation
- STD clamping
- Confidence penalty
- Dynamic thresholds
- MediaPipe mode audit

**Scope EXCLUDES:**
- New training data collection
- Template regeneration from new data
- Model retraining
- New model architectures

---

## REMAINING OPEN QUESTIONS

These questions may be answered during research/planning if analysis reveals them, or left for later:

1. **Model warm-up/burn-in:** Should we add a """burn-in""" period for GCN models before accepting predictions?
2. **Ensemble voting:** Would running all 3 viewpoint models and voting improve accuracy vs current viewpoint-switching?
3. **Feature importance analysis:** Which of the 30 features contribute most to thrust/block discrimination?

---

## FILES TO MODIFY

| File | Changes |
|------|---------|
| `app/models/gcn/feature_extraction.py` | 3D angle calculation, update `calculate_angle()` |
| `app/computer_vision/gcn_inference.py` | STD clamping in _load_config(), dynamic threshold in predict(), confidence penalty |
| `app/computer_vision/pose_analyzer.py` | MediaPipe mode audit comments, verify pose_static usage |
| `app/models/gcn_model_config.json` | Document new threshold behavior (comment, not value change) |

---

## SUCCESS CRITERIA

- [ ] 3D angle calculation implemented and tested
- [ ] STD clamping applied to all templates (logged)
- [ ] Confidence penalty applied for missing stick
- [ ] Dynamic threshold formula implemented
- [ ] MediaPipe mode usage audited and documented
- [ ] Thrust vs block classification accuracy improved (measured)

---

## RELATED DOCUMENTS

- `classification_failure_analysis.md` — Root cause analysis
- `app/models/gcn/feature_templates.json` — Template data (wide STDs)
- `app/models/gcn_model_config.json` — Current thresholds

---

*Context generated: 2026-04-15*  
*For: Researcher and Planner agents*
