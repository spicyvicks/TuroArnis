"""
Example: Manual Device Configuration

This example shows how to use the device manager manually
in custom scripts or notebooks.
"""

import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.utils.device_manager import configure_device, get_yolo_device, set_tensorflow_device
import tensorflow as tf
from ultralytics import YOLO
import numpy as np

def example_tensorflow():
    """Example: TensorFlow model with GPU/CPU auto-configuration"""
    print("\n" + "="*70)
    print("TENSORFLOW EXAMPLE")
    print("="*70)
    
    # Configure device
    device_info = configure_device(verbose=False)
    
    print(f"\nTensorFlow will use: {device_info['tf_device']}")
    
    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    # TensorFlow automatically uses the configured device
    # But you can be explicit with context manager:
    with set_tensorflow_device(device_info['tf_device']):
        # Generate dummy data
        x = np.random.rand(100, 10).astype(np.float32)
        
        # Prediction automatically runs on configured device
        predictions = model.predict(x, verbose=0)
        
        print(f"Predictions shape: {predictions.shape}")
        print(f"Sample prediction: {predictions[0]}")
    
    print("✓ TensorFlow example completed successfully")


def example_yolo():
    """Example: YOLO model with GPU/CPU auto-configuration"""
    print("\n" + "="*70)
    print("YOLO EXAMPLE")
    print("="*70)
    
    # Configure device
    device_info = configure_device(verbose=False)
    yolo_device = get_yolo_device(device_info)
    
    print(f"\nYOLO will use device: {yolo_device}")
    
    # Load YOLO model
    try:
        from app.utils.resource_path import get_resource_path
        model_path = get_resource_path('yolov8n.pt')
        
        model = YOLO(model_path)
        model.to(yolo_device)  # Move to GPU or CPU
        
        print(f"YOLO model loaded on device: {yolo_device}")
        print("✓ YOLO example completed successfully")
    except Exception as e:
        print(f"Note: Could not load YOLO model - {e}")
        print("(This is expected if model files are not present)")


def example_device_info():
    """Example: Accessing device information"""
    print("\n" + "="*70)
    print("DEVICE INFO EXAMPLE")
    print("="*70)
    
    device_info = configure_device(verbose=False)
    
    print(f"\nDevice Information:")
    print(f"  Has GPU: {device_info['has_gpu']}")
    print(f"  Device Name: {device_info['device_name']}")
    print(f"  TensorFlow Device: {device_info['tf_device']}")
    print(f"  PyTorch Device: {device_info['torch_device']}")
    
    print(f"\nDetailed Information:")
    for detail in device_info['details']:
        print(f"  - {detail}")
    
    # Use in conditional logic
    if device_info['has_gpu']:
        print("\n[INFO] GPU acceleration is ENABLED")
        print("       Models will run faster!")
    else:
        print("\n[INFO] Running in CPU mode")
        print("       Consider using GPU for better performance")


def main():
    print("\n" + "="*70)
    print("DEVICE MANAGER USAGE EXAMPLES")
    print("="*70)
    
    # Run examples
    example_device_info()
    example_tensorflow()
    example_yolo()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
