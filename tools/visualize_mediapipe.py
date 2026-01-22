"""
Visualize MediaPipe pose detection on dataset images
Run: python tools/visualize_mediapipe.py
"""
import os
import sys
import cv2
import numpy as np
import mediapipe as mp

# paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# ============================================
# CONFIGURATION
# ============================================
DATASET_FOLDER = "archive"  # change this to your dataset folder
IMAGES_PER_CLASS = 3        # how many images to check per class
# ============================================

def visualize_pose_detection():
    """visualize mediapipe detection on sample images"""
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5
    )
    
    dataset_path = os.path.join(project_root, DATASET_FOLDER)
    
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found: {dataset_path}")
        return
    
    classes = [d for d in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, d))]
    
    print(f"\n[INFO] Found {len(classes)} classes in {DATASET_FOLDER}")
    print(f"[INFO] Showing {IMAGES_PER_CLASS} images per class")
    print("\n[CONTROLS]")
    print("  SPACE = Next image")
    print("  Q = Quit")
    print("  S = Save current image")
    print()
    
    detection_stats = {'success': 0, 'fail': 0}
    
    for class_name in sorted(classes):
        class_path = os.path.join(dataset_path, class_name)
        images = [f for f in os.listdir(class_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            print(f"[WARN] No images in {class_name}")
            continue
        
        # sample a few images
        sample_images = images[:IMAGES_PER_CLASS]
        
        for img_name in sample_images:
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"[ERROR] Cannot read: {img_path}")
                continue
            
            # convert to RGB for mediapipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            # create annotated image
            annotated = image.copy()
            
            if results.pose_landmarks:
                detection_stats['success'] += 1
                
                # draw landmarks
                mp_drawing.draw_landmarks(
                    annotated,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                status = "DETECTED"
                status_color = (0, 255, 0)
                
                # count visible landmarks
                visible = sum(1 for lm in results.pose_landmarks.landmark if lm.visibility > 0.5)
                details = f"Visible landmarks: {visible}/33"
            else:
                detection_stats['fail'] += 1
                status = "NO POSE DETECTED"
                status_color = (0, 0, 255)
                details = "MediaPipe could not find a pose"
            
            # add text overlay
            h, w = annotated.shape[:2]
            cv2.rectangle(annotated, (0, 0), (w, 80), (0, 0, 0), -1)
            cv2.putText(annotated, f"Class: {class_name}", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated, f"File: {img_name}", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(annotated, f"Status: {status}", (10, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(annotated, details, (w - 250, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # resize for display
            scale = min(900 / h, 1200 / w, 1.0)
            display = cv2.resize(annotated, None, fx=scale, fy=scale)
            
            # show image
            cv2.imshow("MediaPipe Pose Detection", display)
            
            key = cv2.waitKey(0)
            if key == ord('q'):
                cv2.destroyAllWindows()
                pose.close()
                print_stats(detection_stats)
                return
            elif key == ord('s'):
                save_path = os.path.join(project_root, f"debug_{class_name}_{img_name}")
                cv2.imwrite(save_path, annotated)
                print(f"[SAVED] {save_path}")
    
    cv2.destroyAllWindows()
    pose.close()
    print_stats(detection_stats)


def print_stats(stats):
    """print detection statistics"""
    total = stats['success'] + stats['fail']
    if total > 0:
        success_rate = stats['success'] / total * 100
        print(f"\n[SUMMARY]")
        print(f"  Total images checked: {total}")
        print(f"  Successful detections: {stats['success']}")
        print(f"  Failed detections: {stats['fail']}")
        print(f"  Success rate: {success_rate:.1f}%")


if __name__ == "__main__":
    visualize_pose_detection()
