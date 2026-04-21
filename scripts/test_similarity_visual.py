#!/usr/bin/env python3
"""
Similarity Scoring Test Script - Compare 4 Methods on Single Image

Usage:
    python scripts/test_similarity_visual.py --image path/to/image.jpg --technique left_chest_thrust --viewpoint front --output results/

Arguments:
    --image: Path to test image
    --technique: Target technique (e.g., left_chest_thrust, right_chest_thrust, etc.)
    --viewpoint: Camera viewpoint (front, left, right)
    --output: Output directory for result images (default: similarity_results/)

Outputs 5 files:
    1. method1_simple.png - Simple average approach
    2. method2_variance.png - Variance-weighted approach
    3. method3_multifactor.png - Multi-factor category approach
    4. method4_critical.png - Critical features weighted approach
    5. comparison_grid.png - 2x2 comparison of all methods
    6. report.json - Detailed scoring report
"""

import argparse
import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.computer_vision.pose_analyzer import PoseAnalyzer
from app.computer_vision.gcn_inference import get_gcn_engine
from app.computer_vision.lesson_feedback import SimilarityCalculator
from app.computer_vision.feedback_mapper import generate_lesson_tips
from app.models.gcn.feature_extraction import compute_global_features_from_kpts


class VisualSimilarityTester:
    """Test similarity scoring with visual output for all 4 methods."""
    
    CRITICAL_FEATURES = {
        'stick_angle': 2.0,
        'stick_tip_height': 1.8,
        'stick_grip_height': 1.8,
        'left_elbow_angle': 1.5,
        'right_elbow_angle': 1.5,
        'left_shoulder_angle': 1.3,
        'right_shoulder_angle': 1.3,
        'left_wrist_height': 1.2,
        'right_wrist_height': 1.2,
    }
    
    def __init__(self, output_dir: str = "similarity_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("[INIT] Loading models...")
        self.pose_analyzer = PoseAnalyzer(
            detection_interval=1,
            stick_model_path='app/models/best_stick_v1_18.pt'
        )
        self.gcn_engine = get_gcn_engine(device='cpu')
        self.calculator = SimilarityCalculator(self.gcn_engine.templates)
        print("[INIT] Ready!")
    
    def approach4_critical_weighted(self, hybrid_scores: np.ndarray, feature_names: List[str]) -> float:
        """Approach 4: Weight by criticality for Arnis techniques."""
        weights = []
        for fname in feature_names:
            weight = self.CRITICAL_FEATURES.get(fname, 1.0)
            weights.append(weight)
        
        weights = np.array(weights)
        weighted_scores = hybrid_scores * weights
        total_weight = np.sum(weights)
        
        if total_weight > 0:
            weighted_mean = np.sum(weighted_scores) / total_weight
        else:
            weighted_mean = np.mean(hybrid_scores)
        
        return float(weighted_mean * 100)
    
    def process_image(self, image_path: str, target_technique: str, viewpoint: str) -> Dict:
        """Process image and compute similarity scores for all 4 methods."""
        print(f"\n[PROCESS] Loading image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print(f"[PROCESS] Running pose detection...")
        results = self.pose_analyzer.yolo_model(frame_rgb, verbose=False)
        
        if len(results[0].boxes) == 0:
            raise ValueError("No person detected in image")
        
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        person_crop = frame_rgb[y1:y2, x1:x2]
        h, w = person_crop.shape[:2]
        
        pose_results = self.pose_analyzer.pose_static.process(person_crop)
        if pose_results.pose_landmarks is None:
            raise ValueError("No pose landmarks detected")
        
        pose_keypoints = np.zeros((33, 4))
        for i, lm in enumerate(pose_results.pose_landmarks.landmark):
            pose_keypoints[i] = [lm.x, lm.y, lm.z, lm.visibility]
        
        print(f"[PROCESS] Detecting stick...")
        stick_endpoints, _ = self.pose_analyzer._detect_stick_with_yolo(
            frame_rgb,
            person_bbox=[x1, y1, x2, y2],
            skip_smoothing=True
        )

        stick_keypoints = np.zeros((2, 4))
        if stick_endpoints is not None:
            grip_point, tip_point = stick_endpoints
            gx_norm = (grip_point[0] - x1) / w if w > 0 else 0
            gy_norm = (grip_point[1] - y1) / h if h > 0 else 0
            tx_norm = (tip_point[0] - x1) / w if w > 0 else 0
            ty_norm = (tip_point[1] - y1) / h if h > 0 else 0
            stick_keypoints[0] = [gx_norm, gy_norm, 1.0, 0]
            stick_keypoints[1] = [tx_norm, ty_norm, 1.0, 0]
            print(f"    Stick detected: grip=({grip_point}), tip=({tip_point})")
        
        print(f"[PROCESS] Computing features...")
        global_features = compute_global_features_from_kpts(pose_keypoints, stick_keypoints)
        
        self.gcn_engine.set_viewpoint(viewpoint)
        corrections = self.gcn_engine.get_feature_corrections(global_features, target_technique)
        
        if corrections is None:
            raise ValueError(f"No template found for {viewpoint}_{target_technique}")
        
        hybrid_scores = corrections['hybrid_scores']
        feature_names = corrections['feature_names']
        template_key = f"{viewpoint}_{target_technique}"
        template = self.gcn_engine.templates.get(template_key, {})
        
        print(f"[PROCESS] Computing 4 similarity methods...")
        
        results = {}
        
        score1 = self.calculator.approach1_simple_average(hybrid_scores)
        results['simple'] = {
            'actual_score': score1,
            'display_score': self.calculator.apply_psychological_buffer(score1),
            'passed': score1 >= 65.0,
        }
        
        score2 = self.calculator.approach2_variance_weighted(hybrid_scores, feature_names, template)
        results['variance'] = {
            'actual_score': score2,
            'display_score': self.calculator.apply_psychological_buffer(score2),
            'passed': score2 >= 65.0,
        }
        
        score3, categories = self.calculator.approach3_multi_factor(hybrid_scores, feature_names)
        results['multifactor'] = {
            'actual_score': score3,
            'display_score': self.calculator.apply_psychological_buffer(score3),
            'passed': score3 >= 65.0,
            'categories': categories,
        }
        
        score4 = self.approach4_critical_weighted(hybrid_scores, feature_names)
        results['critical'] = {
            'actual_score': score4,
            'display_score': self.calculator.apply_psychological_buffer(score4),
            'passed': score4 >= 65.0,
        }
        
        low_features = self.calculator.identify_low_features(hybrid_scores, feature_names, threshold=0.70)
        
        # Get raw features and template means from corrections for tip generation
        raw_features = corrections['raw_values']
        template_means = corrections['template_means']
        tips = generate_lesson_tips(low_features, raw_features, template_means, max_tips=3)
        
        print(f"\n[RESULTS] Scores:")
        for method, data in results.items():
            print(f"  {method:12s}: {data['actual_score']:5.1f}% (display: {data['display_score']:5.1f}%) {'PASS' if data['passed'] else 'FAIL'}")
        
        return {
            'image': image,
            'pose_keypoints': pose_keypoints,
            'stick_keypoints': stick_keypoints,
            'person_bbox': [x1, y1, x2, y2],
            'results': results,
            'low_features': low_features,
            'tips': tips,
            'feature_names': feature_names,
            'hybrid_scores': hybrid_scores,
        }
    
    def draw_similarity_overlay(
        self, image: np.ndarray, result_data: Dict,
        method: str, method_name: str, color_scheme: Tuple[int, int, int]
    ) -> np.ndarray:
        """Draw similarity feedback overlay on image."""
        output = image.copy()
        h, w = output.shape[:2]
        
        data = result_data['results'][method]
        display_score = data['display_score']
        actual_score = data['actual_score']
        passed = data['passed']
        
        if display_score >= 90:
            score_color = (0, 255, 0)
            bar_color = (0, 200, 0)
        elif display_score >= 70:
            score_color = (0, 255, 255)
            bar_color = (0, 200, 200)
        else:
            score_color = (0, 0, 255)
            bar_color = (0, 0, 200)
        
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (w, 180), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.8, output, 0.2, 0, output)
        
        cv2.putText(output, f"Method: {method_name}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color_scheme, 2)
        
        score_text = f"{display_score:.0f}%"
        cv2.putText(output, score_text, (w//2 - 80, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.5, score_color, 4)
        
        actual_text = f"Actual: {actual_score:.1f}%"
        cv2.putText(output, actual_text, (w//2 - 70, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        bar_x, bar_y = 20, 155
        bar_w = w - 40
        bar_h = 15
        filled_w = int((display_score / 100.0) * bar_w)
        
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), bar_color, -1)
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (150, 150, 150), 1)
        
        pass_text = "PASS" if passed else "NEEDS WORK"
        pass_color = (0, 255, 0) if passed else (0, 0, 255)
        cv2.putText(output, pass_text, (w - 150, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, pass_color, 2)
        
        if method == 'multifactor' and 'categories' in data:
            cat_y = 200
            cv2.putText(output, "Categories:", (20, cat_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            cat_colors = {
                'joint_angles': (255, 150, 100),
                'stick_position': (100, 255, 150),
                'body_posture': (150, 100, 255)
            }
            
            for i, (cat, score) in enumerate(data['categories'].items()):
                cat_name = cat.replace('_', ' ').title()
                text = f"  {cat_name}: {score:.0f}%"
                color = cat_colors.get(cat, (200, 200, 200))
                cv2.putText(output, text, (20, cat_y + 30 + (i * 25)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        if method == 'critical':
            crit_y = 200
            cv2.putText(output, "Critical Features Weighted:", (20, crit_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(output, "  Stick: 2.0x  Joints: 1.5x  Other: 1.0x", (20, crit_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        if result_data['tips']:
            tips_y = h - 100
            overlay = output.copy()
            cv2.rectangle(overlay, (0, tips_y - 30), (w, h), (40, 30, 30), -1)
            cv2.addWeighted(overlay, 0.85, output, 0.15, 0, output)
            
            cv2.putText(output, "Tips:", (20, tips_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 1)
            
            for i, tip in enumerate(result_data['tips'][:3]):
                cv2.putText(output, f"  * {tip}", (20, tips_y + 30 + (i * 25)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        
        self._draw_pose_skeleton(output, result_data['pose_keypoints'], result_data['person_bbox'])
        
        return output
    
    def _draw_pose_skeleton(self, image: np.ndarray, keypoints: np.ndarray, bbox: List[int]):
        """Draw simplified pose skeleton on image."""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        
        connections = [
            (11, 13), (13, 15),
            (12, 14), (14, 16),
            (11, 12),
            (11, 23), (12, 24),
            (23, 25), (25, 27),
            (24, 26), (26, 28),
        ]
        
        for start_idx, end_idx in connections:
            if keypoints[start_idx][3] > 0.5 and keypoints[end_idx][3] > 0.5:
                x_start = int(keypoints[start_idx][0] * w + x1)
                y_start = int(keypoints[start_idx][1] * h + y1)
                x_end = int(keypoints[end_idx][0] * w + x1)
                y_end = int(keypoints[end_idx][1] * h + y1)
                cv2.line(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        
        for i in range(33):
            if keypoints[i][3] > 0.5:
                x = int(keypoints[i][0] * w + x1)
                y = int(keypoints[i][1] * h + y1)
                cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
    
    def create_comparison_grid(self, result_data: Dict, target_technique: str, viewpoint: str) -> np.ndarray:
        """Create a 2x2 grid comparing all 4 methods."""
        methods = [
            ('simple', 'Simple Average', (100, 150, 255)),
            ('variance', 'Variance-Weighted', (150, 255, 100)),
            ('multifactor', 'Multi-Factor', (255, 100, 150)),
            ('critical', 'Critical-Weighted', (255, 200, 100)),
        ]
        
        images = []
        for method, name, color in methods:
            img = self.draw_similarity_overlay(
                result_data['image'].copy(), result_data, method, name, color
            )
            images.append(img)
        
        h, w = images[0].shape[:2]
        target_h, target_w = 600, 800
        images = [cv2.resize(img, (target_w, target_h)) for img in images]
        
        top_row = np.hstack([images[0], images[1]])
        bottom_row = np.hstack([images[2], images[3]])
        grid = np.vstack([top_row, bottom_row])
        
        title_h = 50
        title_bar = np.zeros((title_h, grid.shape[1], 3), dtype=np.uint8)
        title_bar[:] = (50, 50, 50)
        
        title_text = f"Similarity Comparison: {target_technique.replace('_', ' ').title()} ({viewpoint.title()} view)"
        cv2.putText(title_bar, title_text, (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        grid = np.vstack([title_bar, grid])
        
        return grid
    
    def run(self, image_path: str, target_technique: str, viewpoint: str):
        """Run complete test and save outputs."""
        result_data = self.process_image(image_path, target_technique, viewpoint)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(image_path).stem
        
        methods = [
            ('simple', 'Simple Average', (100, 150, 255)),
            ('variance', 'Variance-Weighted', (150, 255, 100)),
            ('multifactor', 'Multi-Factor', (255, 100, 150)),
            ('critical', 'Critical-Weighted', (255, 200, 100)),
        ]
        
        print(f"\n[OUTPUT] Saving result images to: {self.output_dir}")
        
        saved_files = []
        for method, name, color in methods:
            output = self.draw_similarity_overlay(
                result_data['image'].copy(), result_data, method, name, color
            )
            
            filename = f"{timestamp}_{base_name}_{method}.png"
            filepath = self.output_dir / filename
            cv2.imwrite(str(filepath), output)
            saved_files.append(filepath)
            print(f"  OK {filepath.name}")
        
        grid = self.create_comparison_grid(result_data, target_technique, viewpoint)
        grid_filename = f"{timestamp}_{base_name}_comparison.png"
        grid_filepath = self.output_dir / grid_filename
        cv2.imwrite(str(grid_filepath), grid)
        saved_files.append(grid_filepath)
        print(f"  OK {grid_filepath.name} (comparison grid)")
        
        report = {
            'timestamp': timestamp,
            'image': image_path,
            'technique': target_technique,
            'viewpoint': viewpoint,
            'results': {
                method: {
                    'actual_score': data['actual_score'],
                    'display_score': data['display_score'],
                    'passed': data['passed'],
                    'categories': data.get('categories', {})
                }
                for method, data in result_data['results'].items()
            },
            'tips': result_data['tips'],
            'low_features': [
                {'name': f['name'], 'score': f['score']}
                for f in result_data['low_features']
            ]
        }
        
        json_filename = f"{timestamp}_{base_name}_report.json"
        json_filepath = self.output_dir / json_filename
        with open(json_filepath, 'w') as f:
            json.dump(report, f, indent=2)
        saved_files.append(json_filepath)
        print(f"  OK {json_filepath.name} (JSON report)")
        
        print(f"\n[COMPLETE] Generated {len(saved_files)} files")
        
        return saved_files


def main():
    parser = argparse.ArgumentParser(
        description='Test similarity scoring with visual output for 4 methods'
    )
    parser.add_argument('--image', '-i', required=True, help='Path to test image')
    parser.add_argument('--technique', '-t', required=True, help='Target technique (e.g., left_chest_thrust)')
    parser.add_argument('--viewpoint', '-v', default='front',
                       choices=['front', 'left', 'right'], help='Camera viewpoint')
    parser.add_argument('--output', '-o', default='similarity_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)
    
    tester = VisualSimilarityTester(output_dir=args.output)
    
    try:
        files = tester.run(args.image, args.technique, args.viewpoint)
        print(f"\n[SUCCESS] All outputs saved to: {args.output}/")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
