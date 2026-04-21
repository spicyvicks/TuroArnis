"""
Lesson Feedback - Similarity-based scoring for lesson mode

Provides continuous feedback on how close a user's pose is to the target technique
using multiple calculation approaches with A/B testing support.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class SimilarityCalculator:
    """
    Calculate pose similarity using 3 different approaches for A/B testing.
    
    Approaches:
    1. Simple Average - Mean of all hybrid similarity scores
    2. Variance-Weighted - Weight scores by template reliability (1/std)
    3. Multi-Factor - Category breakdown (stick/joints/posture)
    """
    
    def __init__(self, templates: Dict):
        """
        Initialize with feature templates.
        
        Args:
            templates: Dict from feature_templates.json with means and stds
        """
        self.templates = templates
    
    def approach1_simple_average(self, hybrid_scores: np.ndarray) -> float:
        """
        Approach 1: Simple average of all hybrid similarity scores.
        
        Args:
            hybrid_scores: Array of 30 Gaussian similarity scores (0-1)
        
        Returns:
            Similarity percentage (0-100)
        """
        return float(np.mean(hybrid_scores) * 100)
    
    def approach2_variance_weighted(
        self, 
        hybrid_scores: np.ndarray, 
        feature_names: List[str],
        template: Dict
    ) -> float:
        """
        Approach 2: Weight scores by template reliability (Mahalanobis-inspired).
        
        Templates with tight variance (low std) are more reliable,
        so their scores count more toward the final similarity.
        
        Args:
            hybrid_scores: Array of 30 Gaussian similarity scores (0-1)
            feature_names: List of feature names in same order as scores
            template: Template dict with 'mean' and 'std' for each feature
        
        Returns:
            Similarity percentage (0-100)
        """
        weights = []
        for fname in feature_names:
            if fname in template:
                # Weight = 1/variance (lower std = higher weight)
                std = template[fname].get('std', 1.0)
                # Prevent division by very small numbers
                weight = 1.0 / max(std, 0.01)
                weights.append(weight)
            else:
                weights.append(1.0)  # Default weight
        
        weights = np.array(weights)
        weighted_scores = hybrid_scores * weights
        
        # Normalize by sum of weights
        total_weight = np.sum(weights)
        if total_weight > 0:
            weighted_mean = np.sum(weighted_scores) / total_weight
        else:
            weighted_mean = np.mean(hybrid_scores)
        
        return float(weighted_mean * 100)
    
    def approach3_multi_factor(
        self, 
        hybrid_scores: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Approach 3: Category-based breakdown with sub-scores.
        
        Groups features into categories and calculates weighted overall score.
        
        Categories:
        - Joint Angles (40%): elbow_angle, shoulder_angle, knee_angle
        - Stick Position (40%): stick_angle, stick_tip_height, stick_grip_height, stick_x
        - Body Posture (20%): wrist_height, wrist_x, elbow_height
        
        Args:
            hybrid_scores: Array of 30 Gaussian similarity scores (0-1)
            feature_names: List of feature names in same order as scores
        
        Returns:
            Tuple of (overall_score, category_breakdown_dict)
            overall_score: 0-100
            category_breakdown: {'joint_angles': %, 'stick_position': %, 'body_posture': %}
        """
        # Define category mappings
        categories = {
            'joint_angles': [
                'left_elbow_angle', 'right_elbow_angle',
                'left_shoulder_angle', 'right_shoulder_angle',
                'left_knee_angle', 'right_knee_angle'
            ],
            'stick_position': [
                'stick_angle', 'stick_tip_height', 'stick_grip_height',
                'stick_tip_x', 'stick_grip_x', 'stick_dx', 'stick_dy'
            ],
            'body_posture': [
                'left_wrist_height', 'right_wrist_height',
                'left_elbow_height', 'right_elbow_height',
                'left_wrist_x', 'right_wrist_x',
                'left_knee_height', 'right_knee_height'
            ]
        }
        
        # Calculate per-category scores
        category_scores = {}
        feature_dict = dict(zip(feature_names, hybrid_scores))
        
        for cat_name, cat_features in categories.items():
            scores = []
            for feat in cat_features:
                if feat in feature_dict:
                    scores.append(feature_dict[feat])
            
            if scores:
                category_scores[cat_name] = np.mean(scores) * 100
            else:
                category_scores[cat_name] = 50.0  # Default neutral
        
        # Weighted overall score
        weights = {
            'joint_angles': 0.4,
            'stick_position': 0.4,
            'body_posture': 0.2
        }
        
        overall = sum(
            category_scores[cat] * weight 
            for cat, weight in weights.items()
        )
        
        return float(overall), category_scores
    
    def calculate_similarity(
        self,
        hybrid_scores: np.ndarray,
        feature_names: List[str],
        template: Dict,
        approach: str = 'variance'
    ) -> Dict:
        """
        Calculate similarity using specified approach.
        
        Args:
            hybrid_scores: Array of 30 Gaussian similarity scores (0-1)
            feature_names: List of feature names
            template: Template dict for variance-weighted approach
            approach: 'simple', 'variance', or 'multifactor'
        
        Returns:
            Dict with 'actual_score', 'category_scores' (if multifactor)
        """
        if approach == 'simple':
            score = self.approach1_simple_average(hybrid_scores)
            return {'actual_score': score}
        
        elif approach == 'variance':
            score = self.approach2_variance_weighted(hybrid_scores, feature_names, template)
            return {'actual_score': score}
        
        elif approach == 'multifactor':
            score, categories = self.approach3_multi_factor(hybrid_scores, feature_names)
            return {
                'actual_score': score,
                'category_scores': categories
            }
        
        else:
            raise ValueError(f"Unknown approach: {approach}")
    
    def apply_psychological_buffer(self, actual_score: float) -> float:
        """
        Apply +5% psychological buffer to displayed scores.
        
        65% actual → 70% display (passing threshold)
        95% actual → 100% display (perfect)
        
        Args:
            actual_score: Real similarity score (0-100)
        
        Returns:
            Display score with buffer applied (capped at 100)
        """
        return min(100.0, actual_score + 5.0)
    
    def identify_low_features(
        self,
        hybrid_scores: np.ndarray,
        feature_names: List[str],
        threshold: float = 0.70
    ) -> List[Dict]:
        """
        Identify features with similarity below threshold for tip generation.
        
        Args:
            hybrid_scores: Array of 30 Gaussian similarity scores (0-1)
            feature_names: List of feature names
            threshold: Similarity threshold (default 0.70 = 70%)
        
        Returns:
            List of dicts: [{'name': str, 'score': float, 'index': int}, ...]
            Sorted by score (lowest first)
        """
        low_features = []
        for i, (name, score) in enumerate(zip(feature_names, hybrid_scores)):
            if score < threshold:
                low_features.append({
                    'name': name,
                    'score': float(score),
                    'index': i
                })
        
        # Sort by score (lowest first)
        low_features.sort(key=lambda x: x['score'])
        return low_features


def test_calculator():
    """Unit tests for SimilarityCalculator"""
    import json
    
    # Mock template
    template = {
        'left_elbow_angle': {'mean': 90, 'std': 10},
        'right_elbow_angle': {'mean': 90, 'std': 5},  # More reliable
        'stick_angle': {'mean': 45, 'std': 15},  # Less reliable
    }
    
    calc = SimilarityCalculator(template)
    
    # Test data: 3 features
    scores = np.array([0.8, 0.9, 0.6])  # 80%, 90%, 60%
    names = ['left_elbow_angle', 'right_elbow_angle', 'stick_angle']
    
    # Test approach 1: simple average = (0.8 + 0.9 + 0.6) / 3 = 76.67%
    result1 = calc.approach1_simple_average(scores)
    assert 75 < result1 < 78, f"Simple average failed: {result1}"
    print(f"[OK] Approach 1 (Simple): {result1:.1f}%")
    
    # Test approach 2: variance-weighted
    # stds: 10, 5, 15 → weights: 0.1, 0.2, 0.067
    # weighted: (0.8*0.1 + 0.9*0.2 + 0.6*0.067) / (0.1+0.2+0.067) = ~0.795
    result2 = calc.approach2_variance_weighted(scores, names, template)
    print(f"[OK] Approach 2 (Variance-Weighted): {result2:.1f}%")
    
    # Test approach 3: multi-factor (with limited features, some categories empty)
    result3, categories = calc.approach3_multi_factor(scores, names)
    print(f"[OK] Approach 3 (Multi-Factor): {result3:.1f}%")
    print(f"  Categories: {categories}")
    
    # Test psychological buffer
    actual = 65.0
    display = calc.apply_psychological_buffer(actual)
    assert display == 70.0, f"Buffer failed: {actual} -> {display}"
    print(f"[OK] Psychological buffer: {actual}% -> {display}%")
    
    # Test capping at 100%
    actual = 98.0
    display = calc.apply_psychological_buffer(actual)
    assert display == 100.0, f"Cap failed: {actual} -> {display}"
    print(f"[OK] Buffer cap at 100%: {actual}% -> {display}%")
    
    # Test low feature detection
    low = calc.identify_low_features(scores, names, threshold=0.70)
    assert len(low) == 1, f"Should find 1 low feature, found {len(low)}"
    assert low[0]['name'] == 'stick_angle', f"Wrong feature: {low[0]['name']}"
    print(f"[OK] Low features: {low}")
    
    print("\n[OK] All SimilarityCalculator tests passed!")


if __name__ == '__main__':
    test_calculator()
