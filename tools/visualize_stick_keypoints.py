"""
Visualize stick detection keypoints from dataset images
Uses YOLO stick detection model (grip and tip keypoints)
Run: python tools/visualize_stick_keypoints.py
"""
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

#paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
output_dir = os.path.join(current_dir, 'output')

DATASET_FOLDER = "dataset"   #dataset folder
MAX_SAMPLES = 100            #limit for clarity
STICK_MODEL_PATH = os.path.join(project_root, 'runs', 'pose', 'arnis_stick_detector', 'weights', 'best.pt')


def calculate_stick_angle(grip_point, tip_point):
    """calculate stick angle relative to vertical (0° = pointing up)"""
    dx = tip_point[0] - grip_point[0]
    dy = tip_point[1] - grip_point[1]
    angle = np.degrees(np.arctan2(dx, -dy))
    return angle

def detect_stick(image, model):
    """detect stick grip and tip keypoints"""
    results = model(image, verbose=False, conf=0.3)
    
    if len(results) == 0 or results[0].keypoints is None:
        return None
    
    result = results[0]
    
    if len(result.boxes) == 0:
        return None
    
    if result.keypoints is not None and len(result.keypoints) > 0:
        kpts = result.keypoints[0].data[0]
        
        grip_point = (int(kpts[0][0]), int(kpts[0][1]))
        tip_point = (int(kpts[1][0]), int(kpts[1][1]))
        grip_conf = float(kpts[0][2])
        tip_conf = float(kpts[1][2])
        
        angle = calculate_stick_angle(grip_point, tip_point)
        
        return {
            'grip': grip_point,
            'tip': tip_point,
            'grip_conf': grip_conf,
            'tip_conf': tip_conf,
            'angle': angle
        }
    
    return None


def visualize_on_images(sample_images, model):
    """visualize stick keypoints overlaid on sample images"""
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
        
        # detect stick
        stick_data = detect_stick(image, model)
        
        if stick_data:
            grip = stick_data['grip']
            tip = stick_data['tip']
            angle = stick_data['angle']
            
            # draw on image
            cv2.circle(image_rgb, grip, 8, (0, 255, 0), -1)  # green = grip
            cv2.circle(image_rgb, tip, 8, (255, 0, 0), -1)   # red = tip
            cv2.line(image_rgb, grip, tip, (255, 255, 0), 3)
            
            # add angle text
            cv2.putText(image_rgb, f"{angle:.1f}°", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            title = f'{class_name}\nAngle: {angle:.1f}°'
        else:
            title = f'{class_name}\n(No stick detected)'
        
        axes[i].imshow(image_rgb)
        axes[i].set_title(title, fontsize=9)
        axes[i].axis('off')
    
    # hide empty subplots
    for i in range(n_show, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'stick_keypoint_overlay.png'), dpi=150)
    print(f"[SAVED] tools/output/stick_keypoint_overlay.png")
    plt.show()


def visualize_angle_distribution(all_angles, class_names):
    """visualize stick angle distribution by class"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    unique_classes = sorted(set(class_names))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    color_map = {cls: colors[i] for i, cls in enumerate(unique_classes)}
    
    #scatter plot: angle by class
    for i, (angle, cls) in enumerate(zip(all_angles, class_names)):
        ax1.scatter(i, angle, c=[color_map[cls]], s=30, alpha=0.7)
    
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Stick Angle (degrees)')
    ax1.set_title('Stick Angle by Sample')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=90, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=-90, color='gray', linestyle='--', alpha=0.5)
    
    #box plot: angle distribution per class
    class_angles = {cls: [] for cls in unique_classes}
    for angle, cls in zip(all_angles, class_names):
        class_angles[cls].append(angle)
    
    box_data = [class_angles[cls] for cls in unique_classes]
    bp = ax2.boxplot(box_data, labels=unique_classes, patch_artist=True)
    
    for patch, cls in zip(bp['boxes'], unique_classes):
        patch.set_facecolor(color_map[cls])
        patch.set_alpha(0.7)
    
    ax2.set_xlabel('Pose Class')
    ax2.set_ylabel('Stick Angle (degrees)')
    ax2.set_title('Stick Angle Distribution by Class')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'stick_angle_distribution.png'), dpi=150)
    print(f"[SAVED] tools/output/stick_angle_distribution.png")
    plt.show()


def main():
    print("\n" + "="*50)
    print("  STICK KEYPOINT VISUALIZATION")
    print("="*50)
    
    #check model exists
    if not os.path.exists(STICK_MODEL_PATH):
        print(f"[ERROR] Stick model not found: {STICK_MODEL_PATH}")
        return
    
    print(f"\n[INFO] Loading stick detection model...")
    model = YOLO(STICK_MODEL_PATH)
    
    dataset_path = os.path.join(project_root, DATASET_FOLDER)
    
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found: {dataset_path}")
        return
    
    classes = [d for d in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, d))]
    
    print(f"[INFO] Found {len(classes)} classes in {DATASET_FOLDER}")
    print(f"[INFO] Extracting up to {MAX_SAMPLES} samples...")
    
    all_angles = []
    class_names = []
    sample_images = []
    detection_stats = {'success': 0, 'fail': 0}
    
    samples_per_class = max(1, MAX_SAMPLES // len(classes))
    
    for class_name in sorted(classes):
        class_path = os.path.join(dataset_path, class_name)
        images = [f for f in os.listdir(class_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        count = 0
        for img_name in images[:samples_per_class]:
            if len(all_angles) >= MAX_SAMPLES:
                break
                
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)
            
            if image is None:
                continue
            
            stick_data = detect_stick(image, model)
            
            if stick_data:
                all_angles.append(stick_data['angle'])
                class_names.append(class_name)
                sample_images.append((img_path, class_name))
                detection_stats['success'] += 1
                count += 1
            else:
                detection_stats['fail'] += 1
        
        print(f"  {class_name}: {count} sticks detected")
    
    print(f"\n[SUMMARY]")
    print(f"  Total images processed: {detection_stats['success'] + detection_stats['fail']}")
    print(f"  Sticks detected: {detection_stats['success']}")
    print(f"  Detection failures: {detection_stats['fail']}")
    
    if len(all_angles) == 0:
        print("[ERROR] No sticks detected!")
        return
    
    success_rate = detection_stats['success'] / (detection_stats['success'] + detection_stats['fail']) * 100
    print(f"  Success rate: {success_rate:.1f}%")
    
    #print sample values
    print(f"\n[SAMPLE ANGLES]")
    for i in range(min(5, len(all_angles))):
        print(f"  {class_names[i]}: {all_angles[i]:.1f}°")
    
    #visualizations
    print("\n[INFO] Generating visualizations...")
    
    #1. keypoints on images
    visualize_on_images(sample_images[:20], model)
    
    #2. angle distribution
    visualize_angle_distribution(all_angles, class_names)
    
    print("\n[DONE]")


if __name__ == "__main__":
    main()
