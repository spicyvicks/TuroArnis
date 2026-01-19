"""python tools/test_stick_detection.py"""
import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# ============================================
# PUT YOUR IMAGE PATH HERE
# ============================================
IMAGE_PATH = "Left Temple Block.jpg"  # <-- CHANGE THIS
# ============================================

# paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
STICK_MODEL_PATH = os.path.join(project_root, 'runs', 'pose', 'arnis_stick_detector', 'weights', 'best.pt')

def calculate_stick_angle(grip_point, tip_point):
    """calculate stick angle relative to vertical (0° = pointing up)"""
    dx = tip_point[0] - grip_point[0]
    dy = tip_point[1] - grip_point[1]
    
    # angle from vertical (0° = up, 90° = right, 180° = down, -90° = left)
    angle = np.degrees(np.arctan2(dx, -dy))  # -dy because y increases downward
    return angle

def detect_stick(image_path, show_result=True):
    """detect stick in image and return angle"""
    
    # load model
    if not os.path.exists(STICK_MODEL_PATH):
        print(f"[ERROR] Stick model not found: {STICK_MODEL_PATH}")
        return None
    
    stick_detector = YOLO(STICK_MODEL_PATH)
    
    # load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return None
    
    print(f"\n[INFO] Processing: {os.path.basename(image_path)}")
    print(f"[INFO] Image size: {image.shape[1]}x{image.shape[0]}")
    
    # run detection
    results = stick_detector(image, verbose=False, conf=0.3)
    
    if len(results) == 0 or results[0].keypoints is None:
        print("[WARN] No stick detected")
        return None
    
    result = results[0]
    
    if len(result.boxes) == 0:
        print("[WARN] No bounding boxes")
        return None
    
    # get keypoints (grip and tip)
    if result.keypoints is not None and len(result.keypoints) > 0:
        kpts = result.keypoints[0].data[0]
        
        grip_point = (int(kpts[0][0]), int(kpts[0][1]))
        tip_point = (int(kpts[1][0]), int(kpts[1][1]))
        grip_conf = kpts[0][2].item()
        tip_conf = kpts[1][2].item()
        
        # calculate angle
        stick_angle = calculate_stick_angle(grip_point, tip_point)
        
        print(f"\n[RESULTS]")
        print(f"  Grip: {grip_point} (conf: {grip_conf:.2f})")
        print(f"  Tip:  {tip_point} (conf: {tip_conf:.2f})")
        print(f"  Stick Angle: {stick_angle:.1f}°")
        print(f"  (0° = up, 90° = right, -90° = left, 180° = down)")
        
        if show_result:
            # draw on image
            cv2.circle(image, grip_point, 10, (0, 255, 0), -1)  # green = grip
            cv2.circle(image, tip_point, 10, (0, 0, 255), -1)   # red = tip
            cv2.line(image, grip_point, tip_point, (255, 255, 0), 3)
            
            cv2.putText(image, "GRIP", (grip_point[0]-25, grip_point[1]-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(image, "TIP", (tip_point[0]-15, tip_point[1]-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(image, f"Angle: {stick_angle:.1f} deg", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # resize image to fit screen while keeping aspect ratio
            screen_height = 900  # approximate screen height
            h, w = image.shape[:2]
            if h > screen_height:
                scale = screen_height / h
                new_w = int(w * scale)
                image = cv2.resize(image, (new_w, screen_height))
            
            # show image
            cv2.namedWindow("Stick Detection", cv2.WINDOW_NORMAL)
            cv2.imshow("Stick Detection", image)
            print("\n[INFO] Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return {
            'grip': grip_point,
            'tip': tip_point,
            'grip_conf': grip_conf,
            'tip_conf': tip_conf,
            'angle': stick_angle
        }
    
    print("[WARN] No keypoints found")
    return None


if __name__ == "__main__":
    # use IMAGE_PATH from top of file
    image_path = IMAGE_PATH
    
    # handle relative paths
    if not os.path.isabs(image_path):
        image_path = os.path.join(project_root, image_path)
    
    detect_stick(image_path)
