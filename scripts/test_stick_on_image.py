import sys
import os
import cv2
import argparse
import numpy as np

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.computer_vision.pose_analyzer import PoseAnalyzer
from app.utils.resource_path import get_resource_path

def main():
    parser = argparse.ArgumentParser(description="Test stick detection on a single image.")
    # Make image_path optional
    parser.add_argument("image_path", nargs='?', help="Path to the input image")
    parser.add_argument("--output", default="debug_stick_output.jpg", help="Path to save the output image")
    args = parser.parse_args()

    # Determine image path
    target_path = args.image_path
    if not target_path:
        # Try to find a default image in the current directory
        potential_defaults = [
            "1 (1).png",
            "test_image.jpg",
            "image.png"
        ]
        for p in potential_defaults:
            if os.path.exists(p):
                target_path = p
                print(f"No image argument provided. Using default found: {target_path}")
                break
    
    if not target_path:
        print("Error: No image path provided and no default test images found.")
        print("Usage: python scripts/test_stick_on_image.py <path_to_image>")
        return

    # Load image
    if not os.path.exists(target_path):
        print(f"Error: Image not found at {target_path}")
        return

    frame = cv2.imread(target_path)
    if frame is None:
        print(f"Error: Could not read image {target_path}")
        return

    print(f"Loaded image: {target_path} ({frame.shape})")

    # Initialize PoseAnalyzer
    try:
        # Use absolute path relative to project root for reliability
        stick_model_path = os.path.join(project_root, 'runs', 'pose', 'arnis_stick_detector', 'weights', 'best.pt')
        
        print(f"Loading stick model from: {stick_model_path}")
        if not os.path.exists(stick_model_path):
             print(f"WARNING: Stick model not found at {stick_model_path}")
             # Try alternatives
             alt_paths = [
                 os.path.join(project_root, 'deployment_package', 'weights', 'best.pt'),
                 os.path.join(project_root, 'app', 'models', 'best.pt'),
                 os.path.join(project_root, 'runs', 'pose', 'arnis_stick_detector', 'weights', 'best.pt')
             ]
             
             for p in alt_paths:
                 if os.path.exists(p):
                     print(f"Found alternative at: {p}")
                     stick_model_path = p
                     break
        
        analyzer = PoseAnalyzer(
            detection_interval=1, # Process every frame
            stick_model_path=stick_model_path,
            debug_stick=True
        )
        
        print(f"PoseAnalyzer initialized. Stick model loaded: {analyzer.stick_detector is not None}")

        print("Running inference...")
        
        results = analyzer.process_frame(frame, skip_ml_inference=True, skip_stick_detection=False)
        
        if not results:
            print("No person detected.")
        else:
            print(f"Detected {len(results)} person(s).")
            for i, data in enumerate(results):
                person_id = data.get('id', i)
                print(f"\nPerson {person_id}:")
                
                # Check stick
                if 'stick_endpoints' in data and data['stick_endpoints']:
                    grip, tip = data['stick_endpoints']
                    print(f"  Stick DETECTED!")
                    print(f"  Grip: {grip}, Tip: {tip}")
                    analyzer.draw_stick_debug(frame, (grip, tip))
                else:
                    print(f"  Stick NOT detected.")
                    # Force try to detect stick manually to see what's happening (bypass cache/logic)
                    if analyzer.stick_detector:
                        x1, y1, x2, y2 = data['bbox']
                        print(f"  Result BBox: {x1, y1, x2, y2}")
                        # Force detection on this bbox
                        stick_res, _ = analyzer._detect_stick_with_yolo(frame, (x1, y1, x2, y2))
                        if stick_res:
                            print(f"  [FORCE DEBUG] Stick found on retry: {stick_res}")
                            analyzer.draw_stick_debug(frame, stick_res)
                        else:
                            print(f"  [FORCE DEBUG] Stick still not found on retry.")

        # Save output
        cv2.imwrite(args.output, frame)
        print(f"\nSaved debug output to: {args.output}")

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
