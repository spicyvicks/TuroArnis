---
title: Implement Quality-Filtered Template Generation
priority: high
date: 2026-04-18
source: /gsd-explore training pipeline optimization
---

## Objective

Add validation layer to training pipeline that filters out images with failed stick detection, low MediaPipe confidence, or impossible feature values before they pollute templates.

## Context

Current template generation in the TuroArnis training pipeline generates statistics (mean, std, min, max) from manually-curated training images, but does not validate that feature extraction actually succeeded. Silent failures in:
- YOLO stick detection (no box, low confidence, or misdetection)
- MediaPipe pose estimation (low confidence on critical joints)
- Feature extraction (impossible values like `stick_length=0`)

...corrupt templates with garbage values, which widens STDs and reduces discriminative power. This cascades into lesson mode classification failures even when users pose correctly.

## Scope

Modify the template generation script(s) to add quality gates that reject images before including them in template statistics.

## Acceptance Criteria

- [ ] Pipeline rejects images where YOLO stick detection confidence < threshold (TBD by research)
- [ ] Pipeline rejects images where MediaPipe visibility < threshold for critical joints (wrists, elbows, at minimum)
- [ ] Pipeline rejects images with physically impossible feature values (e.g., `stick_length=0` but detection claimed success)
- [ ] Rejection reasons are logged per-image for debugging
- [ ] Remaining accepted images produce templates with tighter STDs (target: >30% reduction in critical feature STDs)
- [ ] Regenerated templates improve lesson mode classification accuracy on holdout test set

## Implementation Notes

The validation layer should run **after** `extract_raw_features()` and **before** accumulating statistics. Current workflow appears to be:

```
Image → extract_raw_features() → compute_global_features_from_kpts() → accumulate to template stats
```

Insert validation gate:

```
Image → extract_raw_features() → [VALIDATION GATE] → compute_global_features_from_kpts() → accumulate to template stats
                                        ↓
                                    reject + log reason
```

## Dependencies

- Research: Validation criteria thresholds (see `.planning/research/questions.md`)
- Files likely to modify: template generation scripts (location TBD — may be in `scripts/` or ad-hoc)
- Related: `app/models/gcn/feature_extraction.py` (for understanding feature computation)

## Estimated Effort

Medium — requires both implementation and validation against existing corrupted templates to prove improvement.
