"""
test script to verify device detection and GPU/CPU configuration
run this to check if your system has GPU support
"""
import sys
import os

#add project root to path
if not getattr(sys, 'frozen', False):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from app.utils.device_manager import configure_device, get_yolo_device

def main():
    print("\n" + "="*70)
    print("DEVICE DETECTION TEST")
    print("="*70)
    
    #configure and display device info
    device_info = configure_device(verbose=True)
    
    #get YOLO device
    yolo_device = get_yolo_device(device_info)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"GPU Available: {device_info['has_gpu']}")
    print(f"Device Name: {device_info['device_name']}")
    print(f"TensorFlow will use: {device_info['tf_device']}")
    print(f"PyTorch/YOLO will use: {device_info['torch_device']}")
    print(f"YOLO device parameter: {yolo_device}")
    print("="*70 + "\n")
    
    if device_info['has_gpu']:
        print("[OK] Your system has GPU support! The app will use GPU acceleration.")
    else:
        print("[WARNING] No GPU detected. The app will use CPU (still works, but slower).")
    
    print("\n")

if __name__ == "__main__":
    main()
