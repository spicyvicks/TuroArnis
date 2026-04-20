# Quick Tasks

## Active
- training-pipeline-fixes: Fix 5 incompatibilities between training and app
  - [x] Issue #1: Class name mismatch (FIXED - removed 'neutral')
  - [x] Issue #3: Coordinate space mismatch (FIXED - full frame normalization)
  - [x] Issue #2: Quality validation gates (FIXED - ML repo commit 00a0005)
  - [x] Issue #4: 3D angle calculation (FIXED - ML repo commit 00a0005)
  - [x] Issue #5: Stick fallback handling (FIXED - ML repo commit 00a0005)

## Completed
- training-pipeline-fixes: All 5 incompatibilities fixed across both repos!
  - TuroArnis app: 2 commits (30bf1165, c41d2f47)
  - TuroArnis-ML repo: 1 commit (00a0005)

## Next Steps
1. Regenerate templates using updated 1_extract_reference_features.py
2. Retrain models with clean data
3. Copy deployment_package to app/models/gcn/ for consistency
4. Test classification accuracy
