---
phase: 05-5-classification-optimization
plan: 03
status: complete
commit: 991a78b9
date: 2026-04-15
---

## Summary: MediaPipe Mode Audit

### What Was Built

Audited and documented all MediaPipe mode usage in the codebase to prevent temporal bleed between lesson frames. Added inline comments and runtime assertions to ensure lesson mode always uses static MediaPipe (no temporal smoothing).

### Changes Made

**Task 1: Document all process_frame call sites**

Added `# MODE:` comments to 9 call sites across 6 files:

| File | Line | Mode | Context |
|------|------|------|---------|
| app.py | 1053 | video | Live camera preview (temporal smoothing OK) |
| app.py | 1133 | snapshot | GCN warm-up (single image) |
| app.py | 1767 | snapshot | Lesson pose classification (no temporal smoothing) |
| app.py | 1997 | countdown | Countdown visualization (video mode) |
| eval_app.py | 381, 951 | snapshot | Evaluation/classification (accuracy) |
| main_video.py | 475 | snapshot | Video file frame-by-frame (no temporal bleed) |
| main_app.py | 486 | snapshot | Main app processing (lessons need accuracy) |
| main_image.py | 309 | snapshot | Single image analysis (no temporal smoothing) |
| test_image_app.py | 252, 616 | snapshot | Test image classification |

**Task 2: Add runtime mode assertions**

Added mode parameter validation:
- Raises `ValueError` if mode not in `('snapshot', 'video', 'countdown')`
- Added at start of `process_frame()` method

Added pose instance assertion:
- Verifies `pose_instance is self.pose_static` when mode == 'snapshot'
- Added after mode selection logic
- Catches refactoring errors during development

**Task 3: Verify pose_static configuration**

Enhanced documentation for pose_static initialization:
- Added `FIX #6` reference comment
- Documented `static_image_mode=True` importance
- Explains why this prevents temporal bleed

Enhanced mode selection documentation:
- Added `# MODE SELECTION:` comment explaining the mapping
- Documents: `snapshot=pose_static` (lessons/accuracy)
- Documents: `other=pose` (live/video with smoothing)

### Technical Details

**Mode Definitions:**
- **snapshot**: Uses `pose_static` (static_image_mode=True) - no temporal smoothing
- **video**: Uses `pose` (static_image_mode=False) - temporal smoothing enabled
- **countdown**: Uses `pose` - live visualization with tracking

**Temporal Bleed Prevention:**
- Video mode carries ghost keypoints between unrelated frames
- Static mode treats each frame independently
- Lessons require snapshot mode for accurate per-frame classification

### Key Implementation

```python
# Mode validation at process_frame entry
if mode not in ('snapshot', 'video', 'countdown'):
    raise ValueError(f"Invalid mode: {mode}")

# Mode selection with documentation
pose_instance = self.pose_static if mode == 'snapshot' else self.pose

# Runtime assertion for development safety
if mode == 'snapshot':
    assert pose_instance is self.pose_static, "Must use pose_static for snapshot mode"
```

### Success Criteria

- [x] All process_frame() calls have # MODE: inline comments (9 call sites)
- [x] Mode selection logic has explanatory comment at line 463-466
- [x] Runtime validation rejects invalid mode strings
- [x] Assertions verify pose_static vs pose selection at line 468
- [x] pose_static initialization documented with FIX #6 and static_image_mode explanation

### Impact

**Before:**
- No visibility into which mode was used where
- Temporal bleed risk if video mode accidentally used for lessons
- No runtime guards against incorrect mode strings

**After:**
- Every call site documented with intent and expected behavior
- Runtime validation catches typos and invalid modes
- Assertions ensure pose_static is used for snapshots
- Clear documentation explains why temporal smoothing must be disabled for lessons

### Future Maintenance

When adding new `process_frame()` calls:
1. Add `# MODE: [snapshot|video] - Brief rationale` comment
2. Use 'snapshot' for lessons, classification, single images
3. Use 'video' for live camera streams
4. Mode validation will catch invalid values automatically
