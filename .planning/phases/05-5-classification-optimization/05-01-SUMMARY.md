---
phase: 05-5-classification-optimization
plan: 01
status: complete
commit: 0d0cb233
date: 2026-04-15
---

## Summary: 3D Angle Calculation

### What Was Built

Modified `calculate_angle()` function in `feature_extraction.py` to use 3D coordinates [x, y, z] instead of 2D [x, y].

### Changes Made

**Task 1: Update calculate_angle() to 3D**
- Changed vector construction from 2D to 3D (added z component)
- Added backward compatibility for 2D inputs (converts to 3D with z=0)
- Updated docstring to mention "three-dimensional coordinates"
- All 6 joint angle features now compute true 3D angles

### Technical Details

- **Function:** `calculate_angle(p1, p2, p3)` in `app/models/gcn/feature_extraction.py`
- **Before:** Used only x, y coordinates (2D projection)
- **After:** Uses x, y, z coordinates (full 3D space)
- **Impact:** Thrusts extending forward (z variation) can now be distinguished from blocks moving sideways

### Verification Results

- ? 90-degree 3D angle test passes (returns 90.0°)
- ? Feature extraction works with 3D angles (30 features extracted)
- ? Backward compatibility maintained (2D inputs converted to 3D)

### Key Implementation

```python
# 3D vector construction with backward compatibility
if len(p1) == 2:
    p1 = [p1[0], p1[1], 0.0]
v1 = np.array([p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]])
```

### Success Criteria

- [x] calculate_angle() computes 3D angles using [x, y, z] coordinates
- [x] All 6 joint angle features (elbow, shoulder, knee - left/right) use 3D calculation
- [x] Feature extraction continues to work with existing code
- [x] No breaking changes to function signatures

### Next Steps

This enables better discrimination between:
- **Thrusts:** Large z-depth variation (arm extending toward camera)
- **Blocks:** Minimal z-depth variation (arm moving sideways)

Previously identical in 2D projection from front view, now distinguishable in 3D angle space.
