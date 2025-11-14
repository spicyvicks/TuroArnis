import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)


def calculate_angle(point1, point2, point3):
    vector1 = np.array([point1.x - point2.x, point1.y - point2.y])
    vector2 = np.array([point3.x - point2.x, point3.y - point2.y])
    
    unit_vector1 = vector1 / (np.linalg.norm(vector1) + 1e-6)
    unit_vector2 = vector2 / (np.linalg.norm(vector2) + 1e-6)
    
    dot_product = np.clip(np.dot(unit_vector1, unit_vector2), -1.0, 1.0)
    angle = np.degrees(np.arccos(dot_product))
    
    return angle


def detect_stick_line(image):
    
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 50, 150)
    
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, 
                            minLineLength=80, maxLineGap=15)
    
    if lines is None:
        return None
    
    max_length = 0
    best_line = None
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        if length > max_length and length > 50: 
            max_length = length
            best_line = ((x1, y1), (x2, y2))
    
    return best_line


def analyze_single_image(image_path):
    """
    Analyze a single image to extract:
    - Joint angles for all major joints
    - Stick position and orientation
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    h, w = image.shape[:2]
    
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None
    
    landmarks = results.pose_landmarks.landmark
    
    angles = {}
    
    # Right arm
    if all(landmarks[i].visibility > 0.5 for i in [12, 14, 16]):  # shoulder, elbow, wrist
        angles['right_elbow'] = calculate_angle(
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        )
    
    if all(landmarks[i].visibility > 0.5 for i in [11, 12, 14]):  # hip, shoulder, elbow
        angles['right_shoulder'] = calculate_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        )
    
    # Left arm
    if all(landmarks[i].visibility > 0.5 for i in [11, 13, 15]):
        angles['left_elbow'] = calculate_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        )
    
    if all(landmarks[i].visibility > 0.5 for i in [12, 11, 13]):
        angles['left_shoulder'] = calculate_angle(
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        )
    
    # Right leg
    if all(landmarks[i].visibility > 0.5 for i in [24, 26, 28]):
        angles['right_knee'] = calculate_angle(
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE],
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
        )
    
    if all(landmarks[i].visibility > 0.5 for i in [12, 24, 26]):
        angles['right_hip'] = calculate_angle(
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        )
    
    # Left leg
    if all(landmarks[i].visibility > 0.5 for i in [23, 25, 27]):
        angles['left_knee'] = calculate_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_HIP],
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE],
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        )
    
    if all(landmarks[i].visibility > 0.5 for i in [11, 23, 25]):
        angles['left_hip'] = calculate_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
            landmarks[mp_pose.PoseLandmark.LEFT_HIP],
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        )
    
    # Ankles
    if all(landmarks[i].visibility > 0.5 for i in [26, 28, 32]):
        angles['right_ankle'] = calculate_angle(
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE],
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE],
            landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
        )
    
    if all(landmarks[i].visibility > 0.5 for i in [25, 27, 31]):
        angles['left_ankle'] = calculate_angle(
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE],
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE],
            landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
        )
    
    # Detect stick
    stick_data = None
    stick_line = detect_stick_line(image)
    
    if stick_line:
        stick_start, stick_end = stick_line
        
        # Get both hands
        r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        r_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        l_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        
        # Determine which hand is holding the stick (closest to stick midpoint)
        stick_mid = np.array([(stick_start[0] + stick_end[0])/2,
                              (stick_start[1] + stick_end[1])/2])
        
        r_wrist_pos = np.array([r_wrist.x * w, r_wrist.y * h])
        l_wrist_pos = np.array([l_wrist.x * w, l_wrist.y * h])
        
        r_dist = np.linalg.norm(r_wrist_pos - stick_mid)
        l_dist = np.linalg.norm(l_wrist_pos - stick_mid)
        
        if r_dist < l_dist:
            hand = 'right'
            wrist_pos = r_wrist_pos
            elbow_pos = np.array([r_elbow.x * w, r_elbow.y * h])
            shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        else:
            hand = 'left'
            wrist_pos = l_wrist_pos
            elbow_pos = np.array([l_elbow.x * w, l_elbow.y * h])
            shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        
        # Calculate stick vector
        stick_vector = np.array([stick_end[0] - stick_start[0],
                                stick_end[1] - stick_start[1]])
        stick_length = np.linalg.norm(stick_vector)
        
        # Calculate arm vector
        arm_vector = wrist_pos - elbow_pos
        arm_length = np.linalg.norm(arm_vector)
        
        if stick_length > 10 and arm_length > 10:
            # Normalize vectors
            stick_unit = stick_vector / stick_length
            arm_unit = arm_vector / arm_length
            
            # Calculate angle between stick and arm
            dot_product = np.dot(stick_unit, arm_unit)
            cross_product = stick_unit[0] * arm_unit[1] - stick_unit[1] * arm_unit[0]
            angle_radians = np.arctan2(cross_product, dot_product)
            angle_degrees = np.degrees(angle_radians)
            
            # Calculate body reference length
            body_length = np.sqrt((shoulder.x*w - knee.x*w)**2 + 
                                 (shoulder.y*h - knee.y*h)**2)
            
            stick_length_ratio = stick_length / body_length if body_length > 0 else 1.0
            
            stick_data = {
                'hand': hand,
                'stick_arm_angle_degrees': float(angle_degrees),
                'stick_length_ratio': float(stick_length_ratio),
            }
    
    return {
        'angles': angles,
        'stick': stick_data
    }


def analyze_pose_folder(pose_folder):
    """Analyze all images in a pose folder and aggregate results."""
    print(f"Analyzing {pose_folder.name}...")
    
    angle_data = defaultdict(list)
    stick_data_list = []
    
    image_count = 0
    for img_path in pose_folder.glob('*.jpg'):
        result = analyze_single_image(img_path)
        if result:
            image_count += 1
            
            for joint, angle in result['angles'].items():
                angle_data[joint].append(angle)
            
            if result['stick']:
                stick_data_list.append(result['stick'])
    
    if image_count == 0:
        return None
    
    angle_ranges = {}
    for joint, angles in angle_data.items():
        mean_angle = np.mean(angles)
        angle_ranges[joint] = [float(mean_angle - 10), float(mean_angle + 10)]
    
    stick_pattern = None
    if stick_data_list:
        avg_angle = np.mean([s['stick_arm_angle_degrees'] for s in stick_data_list])
        avg_length_ratio = np.mean([s['stick_length_ratio'] for s in stick_data_list])
        
        hand_counts = {'right': 0, 'left': 0}
        for s in stick_data_list:
            hand_counts[s['hand']] += 1
        
        most_common_hand = 'right' if hand_counts['right'] >= hand_counts['left'] else 'left'
        
        stick_pattern = {
            'hand': most_common_hand,
            'stick_arm_angle_degrees': float(avg_angle),
            'stick_length_ratio': float(avg_length_ratio),
        }
    
    print(f"  ✓ Processed {image_count} images")
    if stick_pattern:
        print(f"    Stick: {stick_pattern['hand']} hand, {stick_pattern['stick_arm_angle_degrees']:.1f}° from arm")
    
    return {
        'joint_angles': angle_ranges,
        'stick_pattern': stick_pattern,
        'sample_count': image_count
    }


def generate_pose_library(dataset_path):
    """Generate complete pose library from dataset."""
    dataset_path = Path(dataset_path)
    pose_library = {}
    
    # Process each pose folder
    for pose_folder in sorted(dataset_path.iterdir()):
        if not pose_folder.is_dir() or pose_folder.name == 'incorrect':
            continue
        
        # Clean pose name
        pose_name = pose_folder.name.replace('_correct', '')
        
        result = analyze_pose_folder(pose_folder)
        if result:
            pose_library[pose_name] = result
    
    return pose_library


def save_pose_library(library, output_path='pose_definitions.py'):
    """Save pose library as Python file."""
    with open(output_path, 'w') as f:
        f.write('# Auto-generated pose library with joint angles and stick patterns\n\n')
        f.write('POSE_LIBRARY = ')
        f.write(json.dumps(library, indent=4))
        f.write('\n')
    
    print(f"\n✓ Pose library saved to {output_path}")
    print(f"  Total poses: {len(library)}")
    
    # Print summary
    poses_with_stick = sum(1 for p in library.values() if p.get('stick_pattern'))
    print(f"  Poses with stick patterns: {poses_with_stick}")


def test_single_image(image_path, output_path='test_output.jpg'):
    """
    Test stick detection on a single image and visualize the results.
    Shows detected landmarks, stick line, and joint angles.
    """
    print(f"Testing image: {image_path}")
    
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    h, w = image.shape[:2]
    result = analyze_single_image(image_path)
    
    if not result:
        print("Error: Could not detect pose in image")
        return
    
    output_image = image.copy()
    
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            output_image, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
        )
        
        if result['stick']:
            stick_info = result['stick']
            hand = stick_info['hand']
            
            if hand == 'right':
                wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
                shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            else:
                wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
                shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            
            wrist_pos = np.array([wrist.x * w, wrist.y * h])
            elbow_pos = np.array([elbow.x * w, elbow.y * h])
            
            arm_vector = wrist_pos - elbow_pos
            arm_angle = np.arctan2(arm_vector[1], arm_vector[0])
            
            stick_angle = arm_angle + np.radians(stick_info['stick_arm_angle_degrees'])
            
            body_length = np.sqrt((shoulder.x*w - knee.x*w)**2 + (shoulder.y*h - knee.y*h)**2)
            stick_length = int(body_length * stick_info['stick_length_ratio'])
            
            wrist_x, wrist_y = int(wrist_pos[0]), int(wrist_pos[1])
            endpoint1 = (int(wrist_x - np.cos(stick_angle) * 30),
                        int(wrist_y - np.sin(stick_angle) * 30))
            endpoint2 = (int(wrist_x + np.cos(stick_angle) * stick_length),
                        int(wrist_y + np.sin(stick_angle) * stick_length))
            
            cv2.line(output_image, endpoint1, endpoint2, (0, 255, 255), 4)
            
            info_text = f"Stick: {hand} hand, {stick_info['stick_arm_angle_degrees']:.1f}° from arm"
            cv2.putText(output_image, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    y_offset = 60
    for joint, angle in result['angles'].items():
        text = f"{joint}: {angle:.1f}°"
        cv2.putText(output_image, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
    
    cv2.imwrite(output_path, output_image)
    print(f"\n✓ Results saved to {output_path}")
    print(f"\nDetected angles:")
    for joint, angle in result['angles'].items():
        print(f"  {joint}: {angle:.1f}°")
    
    if result['stick']:
        print(f"\nStick pattern:")
        print(f"  Hand: {result['stick']['hand']}")
        print(f"  Angle from arm: {result['stick']['stick_arm_angle_degrees']:.1f}°")
        print(f"  Length ratio: {result['stick']['stick_length_ratio']:.2f}")
    else:
        print("\nNo stick detected in image")
    
    try:
        cv2.imshow('Test Results', output_image)
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("(Could not display image - saved to file only)")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else 'test_output.jpg'
        test_single_image(test_image_path, output_path)
    else:
        print("Generating complete pose library from training data...\n")
        
        library = generate_pose_library('dataset_multiclass_2')
        
        save_pose_library(library, 'pose_definitions.py')
        
        print("\nDone! The new pose_definitions.py includes:")
        print("  - Joint angle ranges for each pose")
        print("  - Stick position patterns (hand, angle, length)")
        print("\nYou can now use this in pose_analyzer.py for accurate stick drawing!")
        print("\nTo test a single image, run:")
        print("  python tools/generate_pose_library.py path/to/image.jpg")
