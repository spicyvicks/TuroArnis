"""
Image-based testing version of the Kiosk app
Loads a static image instead of camera feed for faster testing
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

def test_image_classification(image_path, viewpoint='left', target_pose='crown_thrust_correct'):
    """
    Test GCN classification on a static image
    
    Args:
        image_path: Path to the test image
        viewpoint: 'front', 'left', or 'right'
        target_pose: Expected pose class name
    """
    print(f"\n{'='*60}")
    print(f"Testing Image Classification")
    print(f"{'='*60}")
    print(f"Image: {image_path}")
    print(f"Viewpoint: {viewpoint}")
    print(f"Target Pose: {target_pose}")
    print(f"{'='*60}\n")
    
    # Initialize Pose Analyzer
    try:
        stick_model_path = get_resource_path('deployment_package/weights/best.pt')
        pose_analyzer = PoseAnalyzer(
            detection_interval=3,
            stick_model_path=stick_model_path,
            debug_stick=True
        )
        print("[✓] Pose Analyzer initialized successfully\n")
    except Exception as e:
        print(f"[✗] Failed to initialize Pose Analyzer: {e}")
        return
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[✗] Failed to load image: {image_path}")
        return
    
    print(f"[✓] Image loaded: {frame.shape}\n")
    
    # Set viewpoint
    if pose_analyzer.gcn_engine:
        pose_analyzer.gcn_engine.set_viewpoint(viewpoint)
        print(f"[✓] Viewpoint set to: {viewpoint}\n")
    
    # NOTE: No flipping for static images - training images are already in correct orientation
    # (Only the live camera feed needs flipping because it's mirrored for user UX)
    
    # Run pose analysis
    print("Running pose analysis...")
    try:
        results = pose_analyzer.process_frame(frame, skip_ml_inference=False, mode='snapshot')
        
        if results and len(results) > 0:
            person_data = results[0]
            predicted_class = person_data.get('predicted_class', 'N/A')
            confidence = person_data.get('confidence', 0.0)
            landmarks = person_data.get('landmarks_absolute')
            stick_endpoints = person_data.get('stick_endpoints')
            
            print(f"\n{'='*60}")
            print(f"RESULTS")
            print(f"{'='*60}")
            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.2%}")
            print(f"Landmarks Detected: {len(landmarks) if landmarks else 0}")
            print(f"Stick Detected: {'Yes' if stick_endpoints else 'No'}")
            
            # TESTING MODE: Show green for ANY pose with high confidence (not just matching target)
            # This allows verification that the system is working
            confidence_threshold = 0.55
            is_correct = (confidence > confidence_threshold) and (predicted_class != 'N/A') and (predicted_class.lower() != 'no technique detected')
            
            print(f"\nMatch Target: {'✓ YES' if (predicted_class == target_pose) else '✗ NO'}")
            print(f"High Confidence: {'✓ YES' if is_correct else '✗ NO'} (threshold: {confidence_threshold:.0%})")
            print(f"{'='*60}\n")
            
            # Visualize results
            visualize_results(frame, person_data, predicted_class, confidence, is_correct)
            
        else:
            print("[✗] No person detected in image")
            
    except Exception as e:
        print(f"[✗] Error during analysis: {e}")
        import traceback
        traceback.print_exc()

def visualize_results(frame, person_data, predicted_class, confidence, is_correct):
    """Create visualization of results"""
    
    # Get landmarks
    landmarks = person_data.get('landmarks_absolute')
    stick_endpoints = person_data.get('stick_endpoints')
    
    # Create visualization on original frame
    vis_frame = frame.copy()
    
    # Draw skeleton
    if landmarks:
        # Color based on correctness
        color = (0, 255, 0) if is_correct else (0, 0, 255)
        
        # Draw connections (simplified - just major limbs)
        connections = [
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            (11, 12),            # Shoulders
            (11, 23), (12, 24),  # Torso
            (23, 25), (25, 27),  # Left leg
            (24, 26), (26, 28),  # Right leg
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_pt = landmarks[start_idx][:2]
                end_pt = landmarks[end_idx][:2]
                cv2.line(vis_frame, tuple(map(int, start_pt)), tuple(map(int, end_pt)), color, 3)
        
        # Draw keypoints
        for lm in landmarks:
            x, y = int(lm[0]), int(lm[1])
            cv2.circle(vis_frame, (x, y), 5, color, -1)
            cv2.circle(vis_frame, (x, y), 7, (255, 255, 255), 2)
    
    # Draw stick if detected
    if stick_endpoints:
        grip_pt, tip_pt = stick_endpoints
        cv2.line(vis_frame, grip_pt, tip_pt, (255, 255, 0), 4)
        cv2.circle(vis_frame, grip_pt, 8, (0, 0, 255), -1)
        cv2.circle(vis_frame, tip_pt, 8, (255, 0, 0), -1)
    
    # Add text overlay
    text = f"{predicted_class} ({confidence:.1%})"
    text_color = (0, 255, 0) if is_correct else (0, 0, 255)
    cv2.putText(vis_frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 3)
    
    # Save result
    output_path = "test_result.jpg"
    cv2.imwrite(output_path, vis_frame)
    print(f"[✓] Visualization saved to: {output_path}")
    
    # Display with smart resizing (fit to screen height, maintain aspect ratio)
    try:
        # Get screen dimensions (approximate - works on most systems)
        screen_height = 1080  # Default, adjust if needed
        
        # Calculate resize to fit screen height
        img_height, img_width = vis_frame.shape[:2]
        aspect_ratio = img_width / img_height
        
        # New dimensions: height = screen height, width maintains aspect ratio
        new_height = min(screen_height, img_height)  # Don't upscale
        new_width = int(new_height * aspect_ratio)
        
        # Resize for display
        display_frame = cv2.resize(vis_frame, (new_width, new_height))
        
        cv2.imshow("Test Result", display_frame)
        print(f"\nDisplay size: {new_width}x{new_height} (original: {img_width}x{img_height})")
        print("Press any key to close visualization...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"[!] Could not display image: {e}")


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python test_image_app.py <image_path> [viewpoint] [target_pose]")
        print("\nExample:")
        print("  python test_image_app.py training_image.jpg left crown_thrust_correct")
        print("\nViewpoints: front, left, right")
        print("Target poses: crown_thrust_correct, left_knee_block_correct, etc.")
        sys.exit(1)
    
    image_path = sys.argv[1]
    viewpoint = sys.argv[2] if len(sys.argv) > 2 else 'left'
    target_pose = sys.argv[3] if len(sys.argv) > 3 else 'crown_thrust_correct'
    
    test_image_classification(image_path, viewpoint, target_pose)
