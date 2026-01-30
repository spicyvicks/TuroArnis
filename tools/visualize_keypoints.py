"""
Visualize extracted landmark keypoints from dataset images
Uses same extraction method as training.py
Run: python tools/visualize_keypoints.py
"""
import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
output_dir = os.path.join(current_dir, 'output')

#============================================
#configuration
#============================================
DATASET_FOLDER = "dataset"   #dataset folder
MAX_SAMPLES = 100            #limit for clarity
#============================================

def extract_keypoints(image_path, pose):
    """extract 33 landmarks (same as training.py)"""
    image = cv2.imread(image_path)
    if image is None:
        return None, None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if not results.pose_world_landmarks:
        return None, None
    
    landmarks = []
    for lm in results.pose_world_landmarks.landmark:
        landmarks.append([lm.x, lm.y, lm.z])
    
    return np.array(landmarks), results.pose_landmarks


def visualize_3d_keypoints(all_landmarks, class_names):
    """visualize 3D keypoints distribution"""
    fig = plt.figure(figsize=(15, 10))
    
    #3d scatter plot
    ax1 = fig.add_subplot(121, projection='3d')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(set(class_names))))
    unique_classes = sorted(set(class_names))
    color_map = {cls: colors[i] for i, cls in enumerate(unique_classes)}
    
    for landmarks, cls in zip(all_landmarks, class_names):
        color = color_map[cls]
        #plot only key joints (shoulders, elbows, wrists, hips)
        key_joints = [11, 12, 13, 14, 15, 16, 23, 24]  #shoulder, elbow, wrist, hip
        for j in key_joints:
            ax1.scatter(landmarks[j, 0], landmarks[j, 1], landmarks[j, 2], 
                       c=[color], s=10, alpha=0.5)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'3D Keypoint Distribution ({len(all_landmarks)} samples)')
    
    #2d plot (x vs y)
    ax2 = fig.add_subplot(122)
    
    for landmarks, cls in zip(all_landmarks, class_names):
        color = color_map[cls]
        for j in key_joints:
            ax2.scatter(landmarks[j, 0], landmarks[j, 1], 
                       c=[color], s=10, alpha=0.5)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('2D Keypoint Distribution (X vs Y)')
    ax2.invert_yaxis()  #flip y to match image coordinates
    
    #legend
    patches = [plt.Line2D([0], [0], marker='o', color='w', 
               markerfacecolor=color_map[cls], markersize=10, label=cls) 
               for cls in unique_classes]
    ax2.legend(handles=patches, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'keypoint_visualization.png'), dpi=150)
    print(f"\n[SAVED] tools/output/keypoint_visualization.png")
    plt.show()


def visualize_on_images(sample_images, pose, mp_drawing, mp_pose):
    """visualize keypoints overlaid on sample images"""
    n_cols = 4
    n_rows = min(5, (len(sample_images) + n_cols - 1) // n_cols)
    n_show = min(n_cols * n_rows, len(sample_images))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, (img_path, class_name) in enumerate(sample_images[:n_show]):
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
        
        axes[i].imshow(image_rgb)
        axes[i].set_title(f'{class_name}', fontsize=10)
        axes[i].axis('off')
    
    # hide empty subplots
    for i in range(n_show, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'keypoint_overlay.png'), dpi=150)
    print(f"[SAVED] tools/output/keypoint_overlay.png")
    plt.show()


def main():
    print("\n" + "="*50)
    print("  KEYPOINT VISUALIZATION")
    print("="*50)
    
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
    print(f"[INFO] Extracting up to {MAX_SAMPLES} samples...")
    
    all_landmarks = []
    class_names = []
    sample_images = []
    
    samples_per_class = max(1, MAX_SAMPLES // len(classes))
    
    for class_name in sorted(classes):
        class_path = os.path.join(dataset_path, class_name)
        images = [f for f in os.listdir(class_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        count = 0
        for img_name in images[:samples_per_class]:
            if len(all_landmarks) >= MAX_SAMPLES:
                break
                
            img_path = os.path.join(class_path, img_name)
            landmarks, _ = extract_keypoints(img_path, pose)
            
            if landmarks is not None:
                all_landmarks.append(landmarks)
                class_names.append(class_name)
                sample_images.append((img_path, class_name))
                count += 1
        
        print(f"  {class_name}: {count} samples extracted")
    
    pose.close()
    
    print(f"\n[INFO] Total samples: {len(all_landmarks)}")
    
    if len(all_landmarks) == 0:
        print("[ERROR] No keypoints extracted!")
        return
    
    # print sample keypoint values
    print("\n[SAMPLE KEYPOINTS]")
    print(f"  Sample 1 ({class_names[0]}):")
    print(f"    Left shoulder (11):  x={all_landmarks[0][11,0]:.4f}, y={all_landmarks[0][11,1]:.4f}, z={all_landmarks[0][11,2]:.4f}")
    print(f"    Right shoulder (12): x={all_landmarks[0][12,0]:.4f}, y={all_landmarks[0][12,1]:.4f}, z={all_landmarks[0][12,2]:.4f}")
    print(f"    Left wrist (15):     x={all_landmarks[0][15,0]:.4f}, y={all_landmarks[0][15,1]:.4f}, z={all_landmarks[0][15,2]:.4f}")
    print(f"    Right wrist (16):    x={all_landmarks[0][16,0]:.4f}, y={all_landmarks[0][16,1]:.4f}, z={all_landmarks[0][16,2]:.4f}")
    
    # visualizations
    print("\n[INFO] Generating visualizations...")
    
    # reinitialize pose for image overlay
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)
    
    # 1. Keypoints on images
    visualize_on_images(sample_images[:20], pose, mp_drawing, mp_pose)
    
    # 2. 3D distribution
    visualize_3d_keypoints(all_landmarks, class_names)
    
    pose.close()
    print("\n[DONE]")


if __name__ == "__main__":
    main()
