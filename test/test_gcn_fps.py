"""
GCN FPS Performance Test Script
Tests the inference speed of GCN models on MP4 video files

Usage:
    python test_gcn_fps.py --video path/to/video.mp4 --viewpoint front
"""

import cv2
import numpy as np
import torch
import json
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, List
from collections import deque

# Import YOLO and MediaPipe
from ultralytics import YOLO
import mediapipe as mp

# Import GCN components
import sys
sys.path.append(str(Path(__file__).parent.parent))  # Add parent directory to path


from deployment_package.src.model_architecture import HybridGCN, SKELETON_EDGES, CLASS_NAMES
from deployment_package.src.feature_extraction import (
    extract_node_features,
    compute_hybrid_features,
    calculate_angle
)


class GCNPerformanceTester:
    """Test GCN model performance on video files"""
    
    def __init__(self, model_path: str, viewpoint: str = 'front', device: str = 'cpu'):
        self.device = torch.device(device)
        self.viewpoint = viewpoint
        
        print(f"[INFO] Initializing GCN Performance Tester")
        print(f"[INFO] Device: {self.device}")
        print(f"[INFO] Viewpoint: {viewpoint}")
        
        # Load GCN model
        self._load_gcn_model(model_path)
        
        # Initialize YOLO for person detection
        print("[INFO] Loading YOLO person detector...")
        self.person_detector = YOLO('../yolov8n.pt')
        
        # Initialize MediaPipe Pose
        print("[INFO] Loading MediaPipe Pose...")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load stick detector if available
        stick_path = Path('../deployment_package/weights/best.pt')
        if stick_path.exists():
            print("[INFO] Loading stick detector...")
            self.stick_detector = YOLO(str(stick_path))
        else:
            print("[WARN] Stick detector not found, using default positions")
            self.stick_detector = None
        
        # Load feature templates
        template_path = Path('../deployment_package/src/feature_templates.json')
        if template_path.exists():
            with open(template_path, 'r') as f:
                self.templates = json.load(f)
        else:
            print("[WARN] Feature templates not found")
            self.templates = {}
        
        # Prepare graph structure
        self.edge_index = torch.tensor(SKELETON_EDGES, dtype=torch.long).t().to(self.device)
        
        # Performance metrics
        self.reset_metrics()
        
    def _load_gcn_model(self, model_path: str):
        """Load GCN model from checkpoint"""
        print(f"[INFO] Loading GCN model from {model_path}...")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = HybridGCN(
            node_in_channels=checkpoint['node_feat_dim'],
            hybrid_in_channels=checkpoint['hybrid_feat_dim'],
            hidden_channels=checkpoint['hidden_dim'],
            num_classes=len(CLASS_NAMES),
            num_layers=checkpoint['num_layers'],
            dropout=checkpoint['dropout']
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"[INFO] Model loaded successfully")
        print(f"[INFO] Test Accuracy: {checkpoint.get('test_accuracy', 0):.2%}")
        
    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics = {
            'total_frames': 0,
            'processed_frames': 0,
            'yolo_times': [],
            'mediapipe_times': [],
            'stick_times': [],
            'gcn_times': [],
            'total_times': [],
            'predictions': [],
            'confidences': []
        }
        
    def detect_person(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect person bounding box using YOLO"""
        results = self.person_detector(frame, verbose=False, classes=[0])  # class 0 = person
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            # Get first person detection
            box = results[0].boxes[0].xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            return frame[y1:y2, x1:x2]
        
        return None
    
    def extract_pose(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract pose landmarks using MediaPipe"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            pose_kpts = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks])
            return pose_kpts
        
        return None
    
    def detect_stick(self, frame: np.ndarray) -> np.ndarray:
        """Detect stick keypoints using YOLO"""
        if self.stick_detector is None:
            # Return default stick positions
            return np.array([[0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]])
        
        results = self.stick_detector(frame, verbose=False)
        
        if len(results) > 0 and hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
            if results[0].keypoints.xy.shape[1] == 0:  # Check if any keypoints detected
                 return np.array([[0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]])
                 
            try:
                kpts = results[0].keypoints.xy[0].cpu().numpy()
            except IndexError:
                # Handle case where xy exists but is empty
                return np.array([[0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]])
            if len(kpts) >= 2:
                h, w = frame.shape[:2]
                stick_kpts = np.array([
                    [kpts[0][0]/w, kpts[0][1]/h, 0.0, 1.0],
                    [kpts[1][0]/w, kpts[1][1]/h, 0.0, 1.0]
                ])
                return stick_kpts
        
        # Default positions
        return np.array([[0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]])
    
    def compute_global_features(self, kpts: np.ndarray, stick_kpts: np.ndarray) -> Dict:
        """Compute global geometric features"""
        features = {}
        
        # Helper for Euclidian distance
        def calculate_distance(p1, p2):
             return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        # Joint angles
        features['left_elbow_angle'] = calculate_angle(kpts[11], kpts[13], kpts[15])
        features['right_elbow_angle'] = calculate_angle(kpts[12], kpts[14], kpts[16])
        features['left_shoulder_angle'] = calculate_angle(kpts[13], kpts[11], kpts[23])
        features['right_shoulder_angle'] = calculate_angle(kpts[14], kpts[12], kpts[24])
        features['left_knee_angle'] = calculate_angle(kpts[23], kpts[25], kpts[27])
        features['right_knee_angle'] = calculate_angle(kpts[24], kpts[26], kpts[28])
        
        # Stick endpoints
        stick_grip = stick_kpts[0]
        stick_tip = stick_kpts[1]
        
        # Heights relative to hip center
        hip_center_y = (kpts[23][1] + kpts[24][1]) / 2
        features['left_wrist_height'] = hip_center_y - kpts[15][1]
        features['right_wrist_height'] = hip_center_y - kpts[16][1]
        features['left_elbow_height'] = hip_center_y - kpts[13][1]
        features['right_elbow_height'] = hip_center_y - kpts[14][1]
        features['stick_tip_height'] = hip_center_y - stick_tip[1]
        features['stick_grip_height'] = hip_center_y - stick_grip[1]
        
        # Horizontal positions relative to hip center
        hip_center_x = (kpts[23][0] + kpts[24][0]) / 2
        features['left_wrist_x'] = kpts[15][0] - hip_center_x
        features['right_wrist_x'] = kpts[16][0] - hip_center_x
        features['stick_tip_x'] = stick_tip[0] - hip_center_x
        features['stick_grip_x'] = stick_grip[0] - hip_center_x
        
        # Stick orientation
        stick_vector = np.array([stick_tip[0] - stick_grip[0], stick_tip[1] - stick_grip[1]])
        features['stick_angle'] = np.degrees(np.arctan2(stick_vector[1], stick_vector[0]))
        
        stick_len = np.linalg.norm(stick_vector) + 1e-6
        features['stick_dx'] = stick_vector[0] / stick_len
        features['stick_dy'] = stick_vector[1] / stick_len
        
        # Expert features (relative to body landmarks)
        root_x = (kpts[23][0] + kpts[24][0]) / 2
        root_y = (kpts[23][1] + kpts[24][1]) / 2
        shoulder_y = (kpts[11][1] + kpts[12][1]) / 2
        nose_y = kpts[0][1]
        
        features['tip_vs_nose'] = stick_tip[1] - nose_y
        features['tip_vs_shoulder'] = stick_tip[1] - shoulder_y
        features['tip_vs_hip'] = stick_tip[1] - root_y
        
        features['r_hand_vs_nose'] = kpts[16][1] - nose_y
        features['r_hand_vs_shoulder'] = kpts[16][1] - shoulder_y
        features['r_hand_vs_hip'] = kpts[16][1] - root_y
        
        features['tip_side'] = stick_tip[0] - root_x
        features['grip_side'] = stick_grip[0] - root_x
        
        features['foot_stagger'] = kpts[27][1] - kpts[28][1]
        
        # Distances
        features['hands_distance'] = calculate_distance(kpts[15], kpts[16])
        features['stick_length'] = calculate_distance(stick_grip, stick_tip)
        
        return features
    
    def predict(self, pose_kpts: np.ndarray, stick_kpts: np.ndarray, 
                global_features: Dict) -> tuple:
        """Run GCN inference"""
        # Extract node features
        node_features = extract_node_features(pose_kpts, stick_kpts)
        
        # Compute hybrid features
        hybrid_features = compute_hybrid_features(
            global_features,
            self.templates,
            viewpoint=self.viewpoint,
            class_name='neutral_stance'  # Use neutral as reference
        )
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float32).to(self.device)
        hybrid = torch.tensor(hybrid_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        batch = torch.zeros(35, dtype=torch.long).to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(x, self.edge_index, batch, hybrid)
            probabilities = torch.softmax(logits, dim=1)[0]
        
        # Get prediction
        pred_idx = probabilities.argmax().item()
        confidence = probabilities[pred_idx].item()
        predicted_class = CLASS_NAMES[pred_idx]
        
        return predicted_class, confidence
    
    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """Process a single frame through the entire pipeline"""
        t_start = time.perf_counter()
        
        # 1. Person detection
        t0 = time.perf_counter()
        person_crop = self.detect_person(frame)
        t1 = time.perf_counter()
        self.metrics['yolo_times'].append((t1 - t0) * 1000)
        
        if person_crop is None:
            return None
        
        # 2. Pose estimation
        t0 = time.perf_counter()
        pose_kpts = self.extract_pose(person_crop)
        t1 = time.perf_counter()
        self.metrics['mediapipe_times'].append((t1 - t0) * 1000)
        
        if pose_kpts is None:
            return None
        
        # 3. Stick detection
        t0 = time.perf_counter()
        stick_kpts = self.detect_stick(person_crop)
        t1 = time.perf_counter()
        self.metrics['stick_times'].append((t1 - t0) * 1000)
        
        # 4. Global features
        global_features = self.compute_global_features(pose_kpts, stick_kpts)
        
        # 5. GCN inference
        t0 = time.perf_counter()
        predicted_class, confidence = self.predict(pose_kpts, stick_kpts, global_features)
        t1 = time.perf_counter()
        self.metrics['gcn_times'].append((t1 - t0) * 1000)
        
        t_end = time.perf_counter()
        self.metrics['total_times'].append((t_end - t_start) * 1000)
        
        self.metrics['predictions'].append(predicted_class)
        self.metrics['confidences'].append(confidence)
        self.metrics['processed_frames'] += 1
        
        return {
            'prediction': predicted_class,
            'confidence': confidence,
            'processing_time': (t_end - t_start) * 1000
        }
    
    def test_video(self, video_path: str, max_frames: Optional[int] = None,
                   display: bool = False) -> Dict:
        """Test GCN performance on a video file"""
        print(f"\n[INFO] Testing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"[INFO] Video FPS: {fps:.2f}")
        print(f"[INFO] Total frames: {total_frames}")
        print(f"[INFO] Resolution: {width}x{height}")
        
        if max_frames:
            print(f"[INFO] Processing first {max_frames} frames")
        
        self.reset_metrics()
        
        # FPS calculation
        fps_window = deque(maxlen=30)
        
        frame_idx = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                self.metrics['total_frames'] += 1
                frame_idx += 1
                
                # Process frame
                frame_start = time.perf_counter()
                result = self.process_frame(frame)
                frame_end = time.perf_counter()
                
                # Calculate FPS
                frame_time = frame_end - frame_start
                if frame_time > 0:
                    fps_window.append(1.0 / frame_time)
                
                current_fps = np.mean(fps_window) if fps_window else 0
                
                # Display progress
                if frame_idx % 10 == 0:
                    if result:
                        print(f"[FRAME {frame_idx:4d}] FPS: {current_fps:5.2f} | "
                              f"Prediction: {result['prediction']:30s} | "
                              f"Confidence: {result['confidence']:.2%} | "
                              f"Time: {result['processing_time']:6.2f}ms")
                    else:
                        print(f"[FRAME {frame_idx:4d}] FPS: {current_fps:5.2f} | No person detected")
                
                # Display frame if requested
                if display and result:
                    display_frame = frame.copy()
                    cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{result['prediction']}", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Conf: {result['confidence']:.2%}", (10, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow('GCN FPS Test', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Check max frames
                if max_frames and frame_idx >= max_frames:
                    break
                    
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
        
        elapsed_time = time.time() - start_time
        
        # Calculate statistics
        stats = self._calculate_statistics(elapsed_time)
        
        return stats
    
    def _calculate_statistics(self, elapsed_time: float) -> Dict:
        """Calculate performance statistics"""
        m = self.metrics
        
        stats = {
            'total_frames': m['total_frames'],
            'processed_frames': m['processed_frames'],
            'success_rate': m['processed_frames'] / m['total_frames'] if m['total_frames'] > 0 else 0,
            'elapsed_time': elapsed_time,
            'overall_fps': m['total_frames'] / elapsed_time if elapsed_time > 0 else 0,
            'processing_fps': m['processed_frames'] / elapsed_time if elapsed_time > 0 else 0,
        }
        
        # Timing statistics
        for key in ['yolo_times', 'mediapipe_times', 'stick_times', 'gcn_times', 'total_times']:
            times = m[key]
            if times:
                stats[f'{key}_mean'] = np.mean(times)
                stats[f'{key}_std'] = np.std(times)
                stats[f'{key}_min'] = np.min(times)
                stats[f'{key}_max'] = np.max(times)
        
        # Prediction statistics
        if m['predictions']:
            unique, counts = np.unique(m['predictions'], return_counts=True)
            stats['prediction_distribution'] = dict(zip(unique, counts))
            stats['avg_confidence'] = np.mean(m['confidences'])
            stats['min_confidence'] = np.min(m['confidences'])
            stats['max_confidence'] = np.max(m['confidences'])
        
        return stats
    
    def print_report(self, stats: Dict):
        """Print detailed performance report"""
        print("\n" + "="*80)
        print("GCN PERFORMANCE TEST REPORT")
        print("="*80)
        
        print(f"\n📊 OVERALL STATISTICS")
        print(f"  Total Frames:      {stats['total_frames']}")
        print(f"  Processed Frames:  {stats['processed_frames']}")
        print(f"  Success Rate:      {stats['success_rate']:.2%}")
        print(f"  Elapsed Time:      {stats['elapsed_time']:.2f}s")
        print(f"  Overall FPS:       {stats['overall_fps']:.2f}")
        print(f"  Processing FPS:    {stats['processing_fps']:.2f}")
        
        print(f"\n⏱️  TIMING BREAKDOWN (milliseconds)")
        print(f"  {'Component':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        
        components = [
            ('YOLO Person Detect', 'yolo_times'),
            ('MediaPipe Pose', 'mediapipe_times'),
            ('Stick Detection', 'stick_times'),
            ('GCN Inference', 'gcn_times'),
            ('Total Pipeline', 'total_times')
        ]
        
        for name, key in components:
            mean_key = f'{key}_mean'
            if mean_key in stats:
                print(f"  {name:<20} {stats[f'{key}_mean']:8.2f} "
                      f"{stats[f'{key}_std']:8.2f} "
                      f"{stats[f'{key}_min']:8.2f} "
                      f"{stats[f'{key}_max']:8.2f}")
        
        if 'avg_confidence' in stats:
            print(f"\n🎯 PREDICTION STATISTICS")
            print(f"  Average Confidence: {stats['avg_confidence']:.2%}")
            print(f"  Min Confidence:     {stats['min_confidence']:.2%}")
            print(f"  Max Confidence:     {stats['max_confidence']:.2%}")
            
            print(f"\n📈 PREDICTION DISTRIBUTION")
            for pred, count in sorted(stats['prediction_distribution'].items(), 
                                     key=lambda x: x[1], reverse=True):
                percentage = count / stats['processed_frames'] * 100
                print(f"  {pred:<35} {count:4d} ({percentage:5.1f}%)")
        
        print("\n" + "="*80)
        
        # Performance assessment
        gcn_fps = 1000 / stats['gcn_times_mean'] if 'gcn_times_mean' in stats else 0
        total_fps = 1000 / stats['total_times_mean'] if 'total_times_mean' in stats else 0
        
        print(f"\n💡 PERFORMANCE ASSESSMENT")
        print(f"  GCN Inference Speed: {gcn_fps:.1f} FPS ({stats.get('gcn_times_mean', 0):.2f}ms per frame)")
        print(f"  Full Pipeline Speed: {total_fps:.1f} FPS ({stats.get('total_times_mean', 0):.2f}ms per frame)")
        
        if stats.get('gcn_times_mean', 0) < 10:
            print(f"  ✅ GCN inference is FAST (<10ms)")
        elif stats.get('gcn_times_mean', 0) < 20:
            print(f"  ⚠️  GCN inference is ACCEPTABLE (10-20ms)")
        else:
            print(f"  ❌ GCN inference is SLOW (>20ms)")
        
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Test GCN FPS performance on MP4 video')
    parser.add_argument('--video', type=str, required=True, help='Path to MP4 video file')
    parser.add_argument('--viewpoint', type=str, default='front', 
                       choices=['front', 'left', 'right'],
                       help='Viewpoint model to use')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to process')
    parser.add_argument('--display', action='store_true',
                       help='Display video with predictions')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Determine model path
    model_path = f'../deployment_package/models/hybrid_gcn_v2_{args.viewpoint}.pth'
    
    if not Path(model_path).exists():
        print(f"[ERROR] Model not found: {model_path}")
        return
    
    if not Path(args.video).exists():
        print(f"[ERROR] Video not found: {args.video}")
        return
    
    # Create tester
    tester = GCNPerformanceTester(
        model_path=model_path,
        viewpoint=args.viewpoint,
        device=args.device
    )
    
    # Run test
    stats = tester.test_video(
        video_path=args.video,
        max_frames=args.max_frames,
        display=args.display
    )
    
    # Print report
    tester.print_report(stats)


if __name__ == '__main__':
    main()
