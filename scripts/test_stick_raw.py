import cv2
import sys
import os
import numpy as np
from ultralytics import YOLO

# Add project root to path
# Assuming script is in tests/, root is up one level
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_stick_detection():
    # Path to stick model
    # Try different potential locations relative to the script or CWD
    potential_paths = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), '../deployment_package/weights/best.pt')),
        'deployment_package/weights/best.pt',
        'app/models/weights/best.pt',
        'weights/best.pt'
    ]
    
    model_path = None
    for p in potential_paths:
        if os.path.exists(p):
            model_path = p
            break
            
    if not model_path:
        print("Error: Could not find stick detection model (best.pt)")
        print(f"Searched in: {potential_paths}")
        return

    print(f"Loading stick model from: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Starting video loop. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Run inference
        # conf=0.25 to match the recent app update
        # verbose=False to keep console clean
        results = model(frame, verbose=False, conf=0.25)
        
        # Draw results
        if results and len(results) > 0:
            result = results[0]
            
            # Draw all detections
            if result.boxes:
                for i, box in enumerate(result.boxes):
                    # Bounding Box
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf.item()
                    
                    # Draw Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Stick {conf:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Draw Keypoints
                    if result.keypoints is not None and len(result.keypoints) > i:
                        # Extract keypoints for this specific box index
                        kpts = result.keypoints[i].data[0]
                        
                        if kpts.shape[0] >= 2:
                            # Grip (Index 0)
                            grip_x, grip_y = int(kpts[0][0]), int(kpts[0][1])
                            grip_conf = kpts[0][2].item()
                            
                            # Tip (Index 1)
                            tip_x, tip_y = int(kpts[1][0]), int(kpts[1][1]) 
                            tip_conf = kpts[1][2].item()
                            
                            # Draw Grip (Cyan)
                            if grip_conf > 0.3:
                                cv2.circle(frame, (grip_x, grip_y), 8, (255, 255, 0), -1)
                                cv2.putText(frame, "Grip", (grip_x + 10, grip_y), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                            
                            # Draw Tip (Magenta)
                            if tip_conf > 0.3:
                                cv2.circle(frame, (tip_x, tip_y), 8, (255, 0, 255), -1)
                                cv2.putText(frame, "Tip", (tip_x + 10, tip_y), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                            
                            # Draw Stick Line
                            if grip_conf > 0.3 and tip_conf > 0.3:
                                cv2.line(frame, (grip_x, grip_y), (tip_x, tip_y), (0, 165, 255), 3)

        cv2.imshow("Stick Detection Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_stick_detection()
