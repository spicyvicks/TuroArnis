"""
Simple Free Practice Mode Image Classification Test
Tests the new algorithm improvements (3D angles, STD clamping, dynamic thresholds, confidence penalty)

Usage:
    python test_classification.py <image_path> [viewpoint]
    
Examples:
    python test_classification.py demo_images/thrust_test.jpg front
    python test_classification.py demo_images/block_test.jpg left
    python test_classification.py my_test_image.jpg

No forced feedback - just pure classification with detailed logging
"""

import cv2
import numpy as np
import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.computer_vision.pose_analyzer import PoseAnalyzer
from app.utils.resource_path import get_resource_path


def test_classification(image_path, viewpoint='front'):
    """
    Test classification on a single image in free practice mode.
    
    Args:
        image_path: Path to image file
        viewpoint: 'front', 'left', or 'right' (default: 'front')
    """
    print(f"\n{'='*70}")
    print(f"FREE PRACTICE MODE - Classification Test")
    print(f"{'='*70}")
    print(f"Image: {image_path}")
    print(f"Viewpoint: {viewpoint}")
    print(f"Mode: Free Practice (no forced feedback)")
    print(f"{'='*70}\n")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"[✗] Image not found: {image_path}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Available images in demo_images/:")
        demo_dir = "demo_images"
        if os.path.exists(demo_dir):
            for f in os.listdir(demo_dir):
                if f.endswith(('.jpg', '.png', '.jpeg')):
                    print(f"  - {demo_dir}/{f}")
        return
    
    # Initialize Pose Analyzer
    print("[1] Initializing Pose Analyzer with new algorithm improvements...")
    print("    - 3D angle calculation")
    print("    - STD clamping (templates tightened)")
    print("    - Dynamic thresholds")
    print("    - Confidence penalty for missing stick")
    print()
    
    try:
        stick_model_path = get_resource_path('deployment_package/weights/best.pt')
        pose_analyzer = PoseAnalyzer(
            detection_interval=3,
            stick_model_path=stick_model_path,
            debug_stick=False  # Less verbose
        )
        
        # Set viewpoint
        if pose_analyzer.gcn_engine:
            pose_analyzer.gcn_engine.set_viewpoint(viewpoint)
            print(f"[✓] GCN Engine loaded - Viewpoint set to: {viewpoint}")
        else:
            print("[✗] GCN Engine not available")
            return
            
    except Exception as e:
        print(f"[✗] Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load image
    print(f"\n[2] Loading image...")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[✗] Failed to load image: {image_path}")
        return
    
    print(f"[✓] Image loaded: {frame.shape[1]}x{frame.shape[0]} pixels")
    
    # Run classification
    print(f"\n[3] Running classification (free practice mode)...")
    print(f"{'-'*70}")
    
    try:
        # MODE: snapshot - Free practice (no temporal smoothing)
        results = pose_analyzer.process_frame(
            frame, 
            skip_ml_inference=False, 
            mode='snapshot'  # Critical for accurate classification
        )
        
        if not results or len(results) == 0:
            print("[✗] No person detected in image")
            return
        
        # Get first detected person
        person_data = results[0]
        predicted_class = person_data.get('predicted_class', 'N/A')
        confidence = person_data.get('confidence', 0.0)
        
        print(f"\n{'='*70}")
        print(f"CLASSIFICATION RESULT")
        print(f"{'='*70}")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        print(f"{'='*70}")
        
        # Show additional details if available
        if 'global_features' in person_data and person_data['global_features']:
            features = person_data['global_features']
            print(f"\n[4] Key Features Detected:")
            if 'left_elbow_angle' in features:
                print(f"    Left Elbow Angle (3D): {features['left_elbow_angle']:.1f}°")
            if 'right_elbow_angle' in features:
                print(f"    Right Elbow Angle (3D): {features['right_elbow_angle']:.1f}°")
            if 'stick_angle' in features and features['stick_angle'] != 0.0:
                print(f"    Stick Angle: {features['stick_angle']:.1f}°")
            else:
                print(f"    Stick: Not detected (confidence penalty applied)")
        
        # Summary
        print(f"\n{'='*70}")
        if predicted_class != 'No Technique Detected':
            print(f"✓ Classification successful!")
            print(f"  The new algorithm correctly identified: {predicted_class}")
            if confidence > 0.6:
                print(f"  High confidence - good pose match")
            elif confidence > 0.4:
                print(f"  Medium confidence - acceptable match")
            else:
                print(f"  Low confidence - consider retaking photo")
        else:
            print(f"✗ No technique detected")
            print(f"  Possible reasons:")
            print(f"  - Pose not recognized as any trained technique")
            print(f"  - Stick not visible (if required)")
            print(f"  - Low confidence below dynamic threshold")
        print(f"{'='*70}")
        
        # Save visualization if desired
        save_output = input(f"\nSave visualization? (y/n): ").lower().strip() == 'y'
        if save_output:
            output_path = f"test_result_{os.path.basename(image_path)}"
            
            # Draw result on image
            vis_frame = frame.copy()
            h, w = vis_frame.shape[:2]
            
            # Add text overlay
            text = f"{predicted_class}: {confidence*100:.1f}%"
            cv2.putText(vis_frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.5, (0, 255, 0) if confidence > 0.5 else (0, 0, 255), 2)
            
            cv2.imwrite(output_path, vis_frame)
            print(f"[✓] Visualization saved: {output_path}")
        
    except Exception as e:
        print(f"[✗] Classification failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Parse arguments and run test"""
    if len(sys.argv) < 2:
        print("Usage: python test_classification.py <image_path> [viewpoint]")
        print()
        print("Examples:")
        print("  python test_classification.py demo_images/ashly_left_chest.jpg front")
        print("  python test_classification.py demo_images/indira_crown_thrust.jpg front")
        print("  python test_classification.py my_image.jpg left")
        print()
        print("Arguments:")
        print("  image_path   - Path to image file (jpg, png, etc.)")
        print("  viewpoint    - 'front', 'left', or 'right' (default: front)")
        print()
        
        # List available demo images
        demo_dir = "demo_images"
        if os.path.exists(demo_dir):
            print("Available demo images:")
            for f in sorted(os.listdir(demo_dir)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    print(f"  - {demo_dir}/{f}")
        return
    
    image_path = sys.argv[1]
    viewpoint = sys.argv[2] if len(sys.argv) > 2 else 'front'
    
    # Validate viewpoint
    if viewpoint not in ['front', 'left', 'right']:
        print(f"[✗] Invalid viewpoint: {viewpoint}")
        print(f"    Must be one of: front, left, right")
        return
    
    test_classification(image_path, viewpoint)


if __name__ == "__main__":
    main()
