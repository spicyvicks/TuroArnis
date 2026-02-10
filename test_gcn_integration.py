"""
Test GCN Integration
This script verifies that the GCN models load correctly and can perform inference.
"""

import os
import sys
import numpy as np
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_gcn_loading():
    """Test 1: Verify GCN engine can be initialized"""
    print("\n=== Test 1: GCN Engine Loading ===")
    try:
        from app.computer_vision.gcn_inference import get_gcn_engine
        engine = get_gcn_engine(device='cpu')
        print("✓ GCN engine initialized successfully")
        print(f"  - Available models: {list(engine.models.keys())}")
        print(f"  - Current viewpoint: {engine.current_viewpoint}")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize GCN engine: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pose_analyzer():
    """Test 2: Verify PoseAnalyzer can be initialized with GCN"""
    print("\n=== Test 2: PoseAnalyzer with GCN ===")
    try:
        from app.computer_vision.pose_analyzer import PoseAnalyzer
        from app.utils.resource_path import get_resource_path
        
        stick_model_path = get_resource_path('deployment_package/weights/best.pt')
        print(f"  - Stick model path: {stick_model_path}")
        print(f"  - Stick model exists: {os.path.exists(stick_model_path)}")
        
        analyzer = PoseAnalyzer(
            detection_interval=3,
            stick_model_path=stick_model_path,
            debug_stick=False
        )
        
        if analyzer.is_gcn:
            print("✓ PoseAnalyzer initialized with GCN successfully")
            print(f"  - GCN engine active: {analyzer.gcn_engine is not None}")
            return True
        else:
            print("✗ PoseAnalyzer did not use GCN (fallback to legacy)")
            return False
            
    except Exception as e:
        print(f"✗ Failed to initialize PoseAnalyzer: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_extraction():
    """Test 3: Verify feature extraction functions work"""
    print("\n=== Test 3: Feature Extraction ===")
    try:
        from app.models.gcn.feature_extraction import (
            extract_node_features,
            compute_global_features_from_kpts,
            compute_hybrid_features
        )
        
        # Create dummy keypoints
        pose_kpts = np.random.rand(33, 4).astype(np.float32)
        stick_kpts = np.random.rand(2, 4).astype(np.float32)
        
        # Test node features
        node_feats = extract_node_features(pose_kpts, stick_kpts)
        print(f"✓ Node features extracted: shape {node_feats.shape}")
        assert node_feats.shape == (35, 6), f"Expected (35, 6), got {node_feats.shape}"
        
        # Test global features
        global_feats = compute_global_features_from_kpts(pose_kpts, stick_kpts)
        print(f"✓ Global features extracted: {len(global_feats)} features")
        
        # Test hybrid features (requires templates)
        import json
        templates_path = 'app/models/gcn/feature_templates.json'
        if os.path.exists(templates_path):
            with open(templates_path, 'r') as f:
                templates = json.load(f)
            
            hybrid_feats = compute_hybrid_features(
                global_feats, templates, 'front', 'neutral_stance'
            )
            print(f"✓ Hybrid features computed: shape {hybrid_feats.shape}")
            assert len(hybrid_feats) == len(global_feats), "Hybrid features length mismatch"
        else:
            print("⚠ Templates file not found, skipping hybrid feature test")
        
        return True
        
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_inference():
    """Test 4: Verify GCN model can perform inference"""
    print("\n=== Test 4: GCN Model Inference ===")
    try:
        from app.computer_vision.gcn_inference import get_gcn_engine
        
        engine = get_gcn_engine(device='cpu')
        
        # Create dummy keypoints
        pose_kpts = np.random.rand(33, 4).astype(np.float32)
        stick_kpts = np.random.rand(2, 4).astype(np.float32)
        
        # Create dummy global features
        from app.models.gcn.feature_extraction import compute_global_features_from_kpts
        global_feats = compute_global_features_from_kpts(pose_kpts, stick_kpts)
        
        # Test inference for each viewpoint
        for viewpoint in ['front', 'left', 'right']:
            engine.set_viewpoint(viewpoint)
            pred_class, confidence, probs = engine.predict(
                pose_kpts, stick_kpts, global_feats
            )
            
            print(f"✓ {viewpoint.capitalize()} inference successful:")
            print(f"    - Predicted: {pred_class}")
            print(f"    - Confidence: {confidence:.2%}")
            print(f"    - Probabilities shape: {probs.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_files():
    """Test 5: Verify all required config and model files exist"""
    print("\n=== Test 5: File Verification ===")
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


def main():
    """Run all tests"""
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
    print("="*60)
    if all_passed:
        print("✓ All tests passed! GCN integration is ready.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
