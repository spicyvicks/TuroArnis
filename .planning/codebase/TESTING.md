# Testing Patterns

**Analysis Date:** 2025-04-13

## Overview

This codebase uses a hybrid testing approach combining automated unit/integration tests with manual validation tools. The testing strategy focuses heavily on computer vision and machine learning model validation, with specialized test scripts for different components.

---

## Test File Organization

### Test Files by Category

**Integration & Model Tests:**
- `test_gcn_integration.py` - Comprehensive GCN integration test suite (5 test areas)
- `test_gcn_model_loading.py` - GCN model loading verification with 12-class check

**Batch & Validation Tests:**
- `test_images_batch.py` - CLI batch tester for image analysis pipeline (424 lines)
- `lesson_module_test.py` - UI flow test for lesson module without ML stack (607 lines)
- `scripts/batch_test_trio.py` - Batch test for 12 poses × 3 viewpoints (601 lines)

**Script-Level Tests:**
- `scripts/test_yolo_pose_video.py` - YOLO-Pose video testing
- `scripts/test_stick_on_video.py` - Stick detection video validation
- `scripts/test_stick_on_image.py` - Stick detection image validation
- `scripts/test_stick_raw.py` - Raw stick detection test
- `scripts/test_user_tracking.py` - User tracking validation

**App-Level Tests:**
- `app/test_image_app.py` - Image analysis GUI test
- `app/database/test_results_window.py` - Results window UI test

---

## Testing Framework

**No Formal Test Framework Detected**

The project does not use pytest, unittest, or other formal testing frameworks. Instead, it uses:

- **Print-based assertions** with visual pass/fail indicators
- **Exit codes** (0 for success, 1 for failure)
- **Manual verification** through visual output and saved results

---

## Test Structure Patterns

### 1. Integration Test Pattern (from `test_gcn_integration.py`)

```python
def test_gcn_loading():
    """Test 1: Verify GCN engine can be initialized"""
    print("\n=== Test 1: GCN Engine Loading ===")
    try:
        from app.computer_vision.gcn_inference import get_gcn_engine
        engine = get_gcn_engine(device='cpu')
        print("✓ GCN engine initialized successfully")
        print(f"  - Available models: {list(engine.models.keys())}")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize GCN engine: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests and report summary"""
    print("="*60)
    print("GCN Integration Test Suite")
    print("="*60)
    
    results = {
        "File Verification": test_config_files(),
        "GCN Engine Loading": test_gcn_loading(),
        "Feature Extraction": test_feature_extraction(),
        "Model Inference": test_model_inference(),
        "PoseAnalyzer Integration": test_pose_analyzer(),
    }
    
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)
```

### 2. Model Loading Test Pattern (from `test_gcn_model_loading.py`)

```python
def main():
    print("=" * 60)
    print("Testing GCN Model Loading (12 Classes)")
    print("=" * 60)
    
    try:
        # Test 1: Imports
        print("\n1. Importing GCN modules...")
        from app.computer_vision.gcn_inference import GCNInferenceEngine
        from app.models.gcn.model_architecture import CLASS_NAMES
        print(f"   ✓ Imports successful")
        
        # Test 2: Class name verification
        print(f"\n2. Checking CLASS_NAMES...")
        print(f"   Number of classes: {len(CLASS_NAMES)}")
        if len(CLASS_NAMES) != 12:
            print(f"   ✗ ERROR: Expected 12 classes, got {len(CLASS_NAMES)}")
            sys.exit(1)
        
        # Test 3: Model initialization
        print(f"\n3. Initializing GCN Engine...")
        engine = GCNInferenceEngine(device='cpu')
        print(f"   ✓ GCN Engine initialized successfully")
        
        # Test 4: Model output dimension verification
        print(f"\n5. Checking model output dimensions...")
        model = engine.models.get('front')
        output_dim = model.fc.out_features
        if output_dim == 12:
            print(f"   ✓ Model output matches 12 classes")
        else:
            print(f"   ✗ ERROR: Expected 12 output classes, got {output_dim}")
            sys.exit(1)
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED - GCN models loaded successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
```

---

## Test Coverage Areas

### 1. File/Resource Verification
Tests verify required files exist before running:
```python
def test_config_files():
    """Test 5: Verify all required config and model files exist"""
    required_files = [
        'app/models/gcn_model_config.json',
        'app/models/gcn/feature_templates.json',
        'deployment_package/models/hybrid_gcn_v2_front.pth',
        'deployment_package/models/hybrid_gcn_v2_left.pth',
        'deployment_package/models/hybrid_gcn_v2_right.pth',
        'deployment_package/weights/best.pt'
    ]
    
    all_exist = True
    for filepath in required_files:
        exists = os.path.exists(filepath)
        status = "✓" if exists else "✗"
        print(f"  {status} {filepath}")
        if not exists:
            all_exist = False
    
    return all_exist
```

### 2. Feature Extraction Testing
```python
def test_feature_extraction():
    """Test 3: Verify feature extraction functions work"""
    # Create dummy keypoints
    pose_kpts = np.random.rand(33, 4).astype(np.float32)
    stick_kpts = np.random.rand(2, 4).astype(np.float32)
    
    # Test node features
    node_feats = extract_node_features(pose_kpts, stick_kpts)
    assert node_feats.shape == (35, 6), f"Expected (35, 6), got {node_feats.shape}"
    
    # Test global features
    global_feats = compute_global_features_from_kpts(pose_kpts, stick_kpts)
    print(f"✓ Global features extracted: {len(global_feats)} features")
```

### 3. Model Inference Testing
```python
def test_model_inference():
    """Test 4: Verify GCN model can perform inference"""
    engine = get_gcn_engine(device='cpu')
    
    # Create dummy keypoints
    pose_kpts = np.random.rand(33, 4).astype(np.float32)
    stick_kpts = np.random.rand(2, 4).astype(np.float32)
    
    # Test inference for each viewpoint
    for viewpoint in ['front', 'left', 'right']:
        engine.set_viewpoint(viewpoint)
        pred_class, confidence, probs = engine.predict(
            pose_kpts, stick_kpts, global_feats
        )
        print(f"✓ {viewpoint.capitalize()} inference successful:")
        print(f"    - Predicted: {pred_class}")
        print(f"    - Confidence: {confidence:.2%}")
```

---

## Batch Testing Pattern

### CLI Batch Tester (`test_images_batch.py`)

**Features:**
- Command-line interface with argparse
- Supports glob patterns for multiple images
- Viewpoint and target pose overrides
- JSON export of results
- Quiet mode for suppressing debug output
- Terminal output with color coding

**Usage Examples:**
```bash
# Single image
python test_images_batch.py path/to/image.jpg

# Multiple images with viewpoint override
python test_images_batch.py img.jpg --viewpoint right --target crown_thrust_correct

# Save results to JSON
python test_images_batch.py *.jpg --json results.json

# Suppress verbose output
python test_images_batch.py img.jpg --quiet
```

**Result Structure:**
```python
return {
    "image": image_path,
    "viewpoint": viewpoint,
    "elapsed_ms": round(elapsed_ms, 1),
    "predicted_class": predicted_class,
    "confidence": round(confidence, 4),
    "threshold": round(threshold, 4),
    "band": _band_label(confidence, threshold),
    "target_pose": effective_target,
    "is_correct": analysis.get("is_correct", False),
    "stick_detected": stick_detected,
    "severity": analysis.get("severity", "n/a"),
    "errors": analysis.get("errors", []),
    "warnings": analysis.get("warnings", []),
    "suggestions": analysis.get("suggestions", []),
    "feedback_messages": feedback_messages,
    "live_angles": live_angles,
    "all_probs": all_probs,
}
```

---

## Validation Methods

### 1. Per-Viewpoint Validation (`scripts/batch_test_trio.py`)

Tests all 12 Arnis poses across 3 viewpoints (Left, Front, Right):

```python
# Zone layout per image:
# Zone 1 (Left)  = left viewpoint   — correct form
# Zone 2 (Center)= front viewpoint  — correct form  
# Zone 3 (Right) = right viewpoint  — INTENTIONALLY INCORRECT form

FILENAME_TO_CLASS = {
    "crown":         "crown_thrust_correct",
    "left_chest":    "left_chest_thrust_correct",
    "left_elbow":    "left_elbow_block_correct",
    # ... 12 total poses
}

ZONE_VIEWPOINTS = ["left", "front", "right"]
```

**Summary Table Output:**
```
================================================================================
BATCH TEST RESULTS — 12 Arnis Poses × 3 Viewpoints
================================================================================
Pose                   | Left View (Zone 1)           | Front View (Zone 2)          | Right View (Zone 3 - Error)
--------------------------------------------------------------------------------
crown                  | OK Crown (85%) [S]           | OK Crown (92%) [S]             | ERR No Technique (0%) [-]
left_chest             | OK Left Chest (78%) [S]      | OK Left Chest (81%) [S]        | ERR No Technique (0%) [-]
...
================================================================================
Left accuracy  (Zone 1): 12/12 (100%)
Front accuracy (Zone 2): 12/12 (100%)
Right (Zone 3): intentionally incorrect — mismatch expected
================================================================================
```

### 2. Softmax Probability Analysis
```python
# Per-class probability table in batch tester
if show_probs and r["all_probs"]:
    print(f"  {BOLD}Softmax Probabilities (all classes):{RESET}")
    sorted_probs = sorted(r["all_probs"].items(), key=lambda x: -x[1])
    for cls, p in sorted_probs:
        bar = "█" * int(p * 30)
        marker = " ← predicted" if cls == pred else ""
        color = GREEN if cls == pred else RESET
        print(f"    {color}{cls:<35}{RESET} {p:6.1%}  {DIM}{bar}{RESET}{marker}")
```

---

## UI Testing Pattern

### Module Test Without ML Stack (`lesson_module_test.py`)

Tests UI flow without requiring:
- Camera feed
- ML models loaded
- Full inference pipeline

```python
"""
lesson_module_test.py
---------------------
TEST VERSION of the lesson module for TuroArnis.
Covers only the NEW screens:
  SPLASH → MODE_SELECT → LESSON_SELECT → LESSON_INSTRUCTION → (practice loop)
The practice loop stubs out camera/CV with a simple countdown + result screen
so the full UI flow can be validated without needing the ML stack running.
"""

# Stub countdown instead of real camera/GCN
self.countdown_val = 5
self.after(1000, self._tick_practice)

def _tick_practice(self):
    if self.countdown_val > 0:
        self.countdown_val -= 1
        self.canvas.itemconfig(self._count_id, text=str(self.countdown_val))
        self.after(1000, self._tick_practice)
    else:
        self.canvas.itemconfig(self._count_id, text="SNAP!")
        self.after(900, self.show_lesson_result)
```

---

## Manual vs Automated Testing

### Automated Tests
- **Unit tests:** Feature extraction, model loading, file verification
- **Integration tests:** GCN engine initialization, inference pipeline
- **Batch tests:** Image processing with result export

### Manual/Visual Tests
- **GUI flow validation:** `lesson_module_test.py` requires visual confirmation
- **Pose detection accuracy:** Requires human verification of skeleton overlay
- **Stick detection validation:** Visual confirmation of grip/tip placement
- **Feedback quality:** Human assessment of correction suggestions

---

## Test Data & Fixtures

### Dummy Data Generation
```python
# Create dummy keypoints for testing
pose_kpts = np.random.rand(33, 4).astype(np.float32)
stick_kpts = np.random.rand(2, 4).astype(np.float32)
```

### Composite Test Images
The `trio/` directory contains composite images with 3 zones:
- Zone 1: Left viewpoint, correct form
- Zone 2: Front viewpoint, correct form
- Zone 3: Right viewpoint, incorrect form (for error testing)

### Expected Results Mapping
```python
FILENAME_TO_CLASS = {
    "crown":         "crown_thrust_correct",
    "left_chest":    "left_chest_thrust_correct",
    "left_elbow":    "left_elbow_block_correct",
    # ... maps filename stems to expected GCN class names
}
```

---

## Testing Commands Summary

| Test | Command | Purpose |
|------|---------|---------|
| GCN Integration | `python test_gcn_integration.py` | Full integration suite |
| Model Loading | `python test_gcn_model_loading.py` | 12-class verification |
| Batch Images | `python test_images_batch.py *.jpg` | Batch image analysis |
| Trio Batch | `python scripts/batch_test_trio.py` | 12 poses × 3 viewpoints |
| UI Flow | `python lesson_module_test.py` | Lesson module UI test |
| Eval App | `python -m app.eval_app` | GUI evaluation tool |

---

## Coverage Gaps

**Not Currently Tested:**
- Database operations (SQLite) - no automated tests
- User management dialogs - manual testing only
- Camera feed processing - requires hardware
- Real-time performance benchmarks - no automated latency tests
- Cross-platform packaging (PyInstaller) - manual build verification

---

*Testing analysis: 2025-04-13*
