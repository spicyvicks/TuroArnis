# Classification Failure Analysis — TuroArnis GCN Pipeline

> [!IMPORTANT]
> This analysis is based on code-reading only (no live run). Every issue below is traceable to a specific file and line number.

---

## Overview of the Classification Path

```
Camera → YOLO person detect → MediaPipe 33 kpts → GCN inference
                                      ↓
                              Stick YOLO detect → stick_kpts_array
                                      ↓
                     compute_global_features_from_kpts()
                                      ↓
                     gcn_engine.predict() [hypothesis loop]
                                      ↓
                     confidence threshold filter → result
```

---

## Root Cause #1 — The Hypothesis Loop Selects the Wrong Winner (Critical)

**File:** [gcn_inference.py](file:///c:/Users/HP/Documents/GitHub/TuroArnis/app/computer_vision/gcn_inference.py) lines 147–172

The [predict()](file:///c:/Users/HP/Documents/GitHub/TuroArnis/app/computer_vision/gcn_inference.py#105-190) method loops over each candidate class and asks:
> *"Given deviation from [candidate] template, what class does the model predict?"*

It records `candidate_conf = probs[candidate_idx]` — **only the probability of that specific candidate** — and picks whichever candidate self-reports the highest single-class probability.

**The problem:** A class can get a high self-consistent probability *not because the pose matches it well*, but because the model is generally uncertain and spreads probability across similar classes. Two structurally similar techniques (e.g., `left_eye_thrust` vs `right_eye_thrust`, or any two thrusts) will produce nearly identical hybrid feature vectors. Under those hybrid features, the model can assign both candidate classes a 0.65 probability, meaning **the winning class is whichever one the model by-default biases toward for that hybrid input mix** — not necessarily the one the user is performing.

**Concrete symptom:** The model says "crown_thrust" when the user is doing "left_eye_thrust" because both produce a very similar soft probability mass on the thrust cluster.

**Fix direction:** After the loop, verify that `best_class` actually wins over *all other classes* in the final `final_probs` array — i.e., `final_probs[best_class_idx] == max(final_probs)`. Currently, it's possible that another class has a higher marginal probability in `final_probs` but was not selected as `best_class` because it wasn't the hypothesis being tested at that iteration.

---

## Root Cause #2 — Stick Not Detected → Silently Falls Back to Center (High Impact)

**File:** [pose_analyzer.py](file:///c:/Users/HP/Documents/GitHub/TuroArnis/app/computer_vision/pose_analyzer.py) lines 720–722 / [feature_extraction.py](file:///c:/Users/HP/Documents/GitHub/TuroArnis/app/models/gcn/feature_extraction.py) lines 76–78

When stick detection fails, `stick_kpts_array` defaults to:
```python
np.array([[0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]])
```
Both grip and tip are placed at the image center with **zero confidence**.

[compute_global_features_from_kpts()](file:///c:/Users/HP/Documents/GitHub/TuroArnis/app/models/gcn/feature_extraction.py#92-164) then computes:
- `stick_angle = arctan2(0, 0) = 0.0` (horizontal)
- `stick_dx = 0`, `stick_dy = 0` (normalised zero-length vector)
- `stick_length = 0`
- `stick_tip_height`, `stick_tip_x`, etc. — all become hip-relative measures of center-of-frame

These values will match *no legitimate template* well, producing **near-zero Gaussian similarity scores for all stick-related features**. Since the GCN model relies heavily on stick features to separate thrust direction and height, the result is that all hypotheses get equally penalised on stick dimensions, and the model defaults to whichever arm-angle cluster happens to be highest — often `neutral` or a mismatched technique.

> [!WARNING]
> `stick_length = 0` gets passed to [extract_node_features()](file:///c:/Users/HP/Documents/GitHub/TuroArnis/app/models/gcn/feature_extraction.py#206-238) (line 161 of feature_extraction.py) and produces zero-length node-level features for nodes 33 and 34. This is a **silent corruption** of the node embedding.

---

## Root Cause #3 — Feature Coordinate Space Mismatch (High Impact)

**File:** [feature_extraction.py](file:///c:/Users/HP/Documents/GitHub/TuroArnis/app/models/gcn/feature_extraction.py) lines 109–163 vs [pose_analyzer.py](file:///c:/Users/HP/Documents/GitHub/TuroArnis/app/computer_vision/pose_analyzer.py) lines 716–719

[compute_global_features_from_kpts()](file:///c:/Users/HP/Documents/GitHub/TuroArnis/app/models/gcn/feature_extraction.py#92-164) expects [kpts](file:///c:/Users/HP/Documents/GitHub/TuroArnis/app/models/gcn/feature_extraction.py#92-164) in **normalized MediaPipe coordinates** (0–1 range, image-relative x/y).

The angle calculation at line 109:
```python
features['left_elbow_angle'] = calculate_angle(kpts[11], kpts[13], kpts[15])
```
…uses `kpts[i][0]` and `kpts[i][1]` (x, y) — which are MediaPipe normalized coordinates. This is fine.

However, the **stick keypoints passed in** are:
```python
stick_kpts_array = np.array([
    [grip_pt[0]/w_frame, grip_pt[1]/h_frame, 0.0, 1.0],
    [tip_pt[0]/w_frame, tip_pt[1]/h_frame, 0.0, 1.0]
])
```
These are normalized by **frame (camera) dimensions**, while MediaPipe landmarks are normalized to the **detected person crop**. If YOLO finds the person occupying 40% of the frame, the MediaPipe [(0.5, 0.5)](file:///c:/Users/HP/Documents/GitHub/TuroArnis/app/computer_vision/pose_analyzer.py#971-974) wrist means *wrist is 50% across the person crop* — perhaps 20% into the actual frame. But stick [(0.5, 0.5)](file:///c:/Users/HP/Documents/GitHub/TuroArnis/app/computer_vision/pose_analyzer.py#971-974) means *center of the full frame*.

**Result:** `stick_grip_height = hip_center_y (MediaPipe-scale) - stick_grip[1] (frame-scale)`. These are in **incompatible units**. The computed height/position features for the stick will be wrong by a factor proportional to how much of the frame the person occupies.

This is one of the strongest contributors to misclassification, especially for smaller/farther subjects.

---

## Root Cause #4 — Extremely Wide Template Standard Deviations Kill Discriminability

**File:** [feature_templates.json](file:///c:/Users/HP/Documents/GitHub/TuroArnis/app/models/gcn/feature_templates.json)

Looking at the templates for `front_crown_thrust_correct`:
| Feature | Mean | STD |
|---|---|---|
| `left_elbow_angle` | 115.2° | **60.2°** |
| `stick_angle` | -39.9° | **48.2°** |
| `stick_dx` | 0.015 | **0.305** |
| `stick_dy` | -0.397 | **0.464** |

A `std` of 60° on elbow angle means the Gaussian similarity score will be near 1.0 for almost any elbow angle from 0° to 175°. This feature **contributes nothing to discrimination**.

`stick_dx` having `std = 0.305` when normalized to [-1, 1] is similarly flat — the Gaussian is so wide that both `stick_dx = 0` and `stick_dx = 0.9` get similarity ≈ 0.95.

Wide STDs arise from training data that included many varied images. But the consequence is that the hybrid feature vector (30-dimensional similarity scores) is **near-all-ones** for most users, and all class hypotheses receive nearly identical hybrid vectors → the model can't separate them.

> [!NOTE]
> `left_chest_thrust` has `stick_angle std = 116°` (essentially the full ±180° range). This feature is completely uninformative for that class.

---

## Root Cause #5 — Angle Computation Uses 2D Normalized Coordinates

**File:** [feature_extraction.py](file:///c:/Users/HP/Documents/GitHub/TuroArnis/app/models/gcn/feature_extraction.py) lines 12–19

```python
def calculate_angle(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])  # only x, y
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
```

The [kpts](file:///c:/Users/HP/Documents/GitHub/TuroArnis/app/models/gcn/feature_extraction.py#92-164) array has 4 columns: `[x, y, z, vis]`. But only `[0]` (x) and `[1]` (y) are used — **z-depth is ignored**.

For Arnis, arm positions differ substantially in depth (a thrust extends toward the camera; a block moves sideways). Ignoring z means:
- A `crown_thrust` arm extended forward vs a `left_knee_block` arm pointing sideways-and-down can look *identical* in 2D projection from the front viewpoint.
- The model's GCN nodes do receive `z` via [extract_node_features](file:///c:/Users/HP/Documents/GitHub/TuroArnis/app/models/gcn/feature_extraction.py#206-238), but the **30 global hybrid features** — which are what drive the class-specific branch of the model — use 2D-only angles.

---

## Root Cause #6 — MediaPipe Running in Video Mode (`static_image_mode=False`) for Snapshots

**File:** [pose_analyzer.py](file:///c:/Users/HP/Documents/GitHub/TuroArnis/app/computer_vision/pose_analyzer.py) line 75

```python
self.pose = self.mp_pose.Pose(
    static_image_mode=False,   # ← video mode
    ...
)
```

In snapshot / classification mode, each call to [process_frame()](file:///c:/Users/HP/Documents/GitHub/TuroArnis/app/computer_vision/pose_analyzer.py#275-765) is an independent static image. `static_image_mode=False` (video mode) enables inter-frame tracking and temporal smoothing. When applied to **unrelated sequential images** (e.g., different users striking different poses), the temporal smoothing carries over ghost keypoints from the previous image into the current one.

**Symptom:** If User A just struck `crown_thrust` and User B is striking `left_knee_block`, MediaPipe in video mode will "remember" the previous arm position and blend it into the new detection. This softens the keypoints toward the previous pose, reducing discriminative signal.

**Fix:** Use `static_image_mode=True` for snapshot classification, or create a separate [Pose](file:///c:/Users/HP/Documents/GitHub/TuroArnis/app/computer_vision/pose_analyzer.py#21-974) instance for snapshot mode.

---

## Root Cause #7 — Confidence Threshold Set Uniformly at 0.70 (Policy Issue)

**File:** [gcn_model_config.json](file:///c:/Users/HP/Documents/GitHub/TuroArnis/app/models/gcn_model_config.json) lines 8, 14, 20

All three viewpoint models use `confidence_threshold: 0.70`. Given:
1. Many hypothesis-loop iterations returning candidate_conf < 0.70 (due to the wide-STD problem above)
2. Stick failures pushing all features to zero
3. Coordinate space mismatch penalising stick features

…the majority of valid poses will never clear 0.70, producing "No Technique Detected" even when the model is moderately confident. This is what generates the "red no detected" feedback in lesson mode.

> [!TIP]
> The threshold should be tuned per-class, not per-viewpoint. Visually similar classes (e.g., all four thrust directions) should have a *lower* threshold because none can achieve 0.70 self-consistent probability when the target techniques are geometrically close.

---

## Summary Table

| # | Issue | Severity | File |
|---|---|---|---|
| 1 | Hypothesis loop doesn't verify marginal winner | 🔴 High | `gcn_inference.py:169` |
| 2 | Stick miss → silent center-frame fallback corrupts all stick features | 🔴 High | `pose_analyzer.py:720` |
| 3 | Stick coordinates in frame-space, body kpts in crop-space | 🔴 High | `pose_analyzer.py:716` |
| 4 | Template STDs too wide → Gaussian scores near 1.0 everywhere | 🟠 Medium | [feature_templates.json](file:///c:/Users/HP/Documents/GitHub/TuroArnis/app/models/gcn/feature_templates.json) |
| 5 | 2D-only angle ignores Z depth for thrust vs block separation | 🟠 Medium | `feature_extraction.py:14` |
| 6 | MediaPipe in video mode for static snapshots (temporal bleed) | 🟡 Medium | `pose_analyzer.py:75` |
| 7 | 0.70 confidence threshold too high given above errors | 🟡 Low | `gcn_model_config.json:8` |

---

## Recommended Fix Priority

1. **Fix #3 first** — normalize stick keypoints to crop-space, not frame-space. Single-line fix in [pose_analyzer.py](file:///c:/Users/HP/Documents/GitHub/TuroArnis/app/computer_vision/pose_analyzer.py).
2. **Fix #2** — when stick is not detected, do not zero-fill; instead flag the inference as "stick-unavailable" and optionally skip or reweight stick features.
3. **Fix #6** — instantiate a separate [Pose(static_image_mode=True)](file:///c:/Users/HP/Documents/GitHub/TuroArnis/app/computer_vision/pose_analyzer.py#21-974) for the snapshot classification path.
4. **Fix #1** — after the hypothesis loop, confirm the selected class is the argmax of `final_probs`, not just the best self-consistent candidate.
5. **Retrain templates (#4)** with cleaner, more consistent training data to tighten STDs.
