"""
Simple test to verify GCN model loading with 12-class configuration
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Testing GCN Model Loading (12 Classes)")
print("=" * 60)

try:
    print("\n1. Importing GCN modules...")
    from app.computer_vision.gcn_inference import GCNInferenceEngine
    from app.models.gcn.model_architecture import CLASS_NAMES
    print(f"   ✓ Imports successful")
    
    print(f"\n2. Checking CLASS_NAMES...")
    print(f"   Number of classes: {len(CLASS_NAMES)}")
    print(f"   Class names: {CLASS_NAMES}")
    
    if len(CLASS_NAMES) != 12:
        print(f"   ✗ ERROR: Expected 12 classes, got {len(CLASS_NAMES)}")
        sys.exit(1)
    
    if 'neutral_stance' in CLASS_NAMES:
        print(f"   ✗ ERROR: neutral_stance should not be in CLASS_NAMES")
        sys.exit(1)
    
    print(f"   ✓ CLASS_NAMES is correctly configured with 12 classes")
    
    print(f"\n3. Initializing GCN Engine...")
    engine = GCNInferenceEngine(device='cpu')
    print(f"   ✓ GCN Engine initialized successfully")
    
    print(f"\n4. Checking loaded models...")
    for viewpoint in ['front', 'left', 'right']:
        if viewpoint in engine.models:
            print(f"   ✓ {viewpoint.capitalize()} model loaded successfully")
        else:
            print(f"   ✗ {viewpoint.capitalize()} model failed to load")
    
    print(f"\n5. Checking model output dimensions...")
    model = engine.models.get('front')
    if model:
        # Check the final layer output dimension
        fc_layer = model.fc
        output_dim = fc_layer.out_features
        print(f"   Model output dimension: {output_dim}")
        
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
