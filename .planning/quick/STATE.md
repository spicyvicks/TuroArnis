# Quick Tasks

## Active
- training-pipeline-fixes: Fix 5 incompatibilities between training and app
  - [x] Issue #1: Class names aligned (13 classes including neutral)
  - [x] Issue #2: Quality validation gates (ML repo)
  - [x] Issue #3: Coordinate space alignment (full frame)
  - [x] Issue #4: 3D angle calculation (ML repo)
  - [x] Issue #5: NaN fallback for missing stick (ML repo)

## Completed
- training-pipeline-fixes: All incompatibilities fixed!
  - TuroArnis app: Class count restored to 13 (1d298305)
  - TuroArnis-ML: Class count restored to 13 (7340e8d)
  - Note: 'neutral' acts as buffer for wrong/incorrect poses

## Summary
Training and app now both use 13 classes (12 correct techniques + 1 neutral).
The neutral class serves as a buffer/sink for:
- Incorrect technique form
- Non-Arnis poses
- Transition states
- Failed detections

All other fixes (validation, 3D angles, NaN fallback, coordinate alignment) remain in place.
