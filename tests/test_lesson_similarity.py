"""
Unit tests for lesson similarity feedback system

Tests all 3 similarity approaches, psychological buffer, and A/B logger
"""

import unittest
import numpy as np
import json
import tempfile
import os
import shutil
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from app.computer_vision.lesson_feedback import SimilarityCalculator
from app.computer_vision.ab_test_logger import ABTestLogger
from app.computer_vision.feedback_mapper import generate_lesson_tips, FEATURE_MESSAGES


class TestSimilarityCalculator(unittest.TestCase):
    """Test SimilarityCalculator class with all 3 approaches"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.template = {
            'left_elbow_angle': {'mean': 90, 'std': 10},
            'right_elbow_angle': {'mean': 90, 'std': 5},
            'stick_angle': {'mean': 45, 'std': 15},
            'left_wrist_height': {'mean': 0.5, 'std': 0.1},
            'right_wrist_height': {'mean': 0.5, 'std': 0.1},
        }
        self.calculator = SimilarityCalculator(self.template)
    
    def test_approach1_simple_average(self):
        """Test simple average approach"""
        scores = np.array([0.8, 0.9, 0.6, 0.7, 0.75])
        result = self.calculator.approach1_simple_average(scores)
        
        expected = np.mean(scores) * 100
        self.assertAlmostEqual(result, expected, places=1)
        self.assertTrue(0 <= result <= 100)
    
    def test_approach2_variance_weighted(self):
        """Test variance-weighted approach weights low std more"""
        scores = np.array([0.8, 0.9, 0.6])
        names = ['left_elbow_angle', 'right_elbow_angle', 'stick_angle']
        
        result = self.calculator.approach2_variance_weighted(scores, names, self.template)
        
        # Right elbow has lower std (5) vs left elbow (10), so it should weight more
        # Result should be different from simple average
        simple_avg = np.mean(scores) * 100
        self.assertNotEqual(result, simple_avg)
        self.assertTrue(0 <= result <= 100)
    
    def test_approach3_multi_factor(self):
        """Test multi-factor category breakdown"""
        # Create scores for features in all 3 categories
        scores = np.array([
            0.8, 0.9,  # joint angles (left/right elbow)
            0.6,       # stick position (stick_angle)
            0.7, 0.75  # body posture (wrist heights)
        ])
        names = [
            'left_elbow_angle', 'right_elbow_angle',
            'stick_angle',
            'left_wrist_height', 'right_wrist_height'
        ]
        
        result, categories = self.calculator.approach3_multi_factor(scores, names)
        
        self.assertTrue(0 <= result <= 100)
        self.assertIn('joint_angles', categories)
        self.assertIn('stick_position', categories)
        self.assertIn('body_posture', categories)
        
        # Check that stick_position is lowest (0.6 score)
        self.assertLess(categories['stick_position'], categories['joint_angles'])
    
    def test_psychological_buffer(self):
        """Test +5% buffer with capping at 100%"""
        # 65% actual should show as 70%
        actual = 65.0
        display = self.calculator.apply_psychological_buffer(actual)
        self.assertEqual(display, 70.0)
        
        # 98% should cap at 100%
        actual = 98.0
        display = self.calculator.apply_psychological_buffer(actual)
        self.assertEqual(display, 100.0)
        
        # 60% should show as 65%
        actual = 60.0
        display = self.calculator.apply_psychological_buffer(actual)
        self.assertEqual(display, 65.0)
    
    def test_identify_low_features(self):
        """Test identifying features below threshold"""
        scores = np.array([0.8, 0.9, 0.6, 0.7, 0.55])
        names = ['feat1', 'feat2', 'feat3', 'feat4', 'feat5']
        
        low = self.calculator.identify_low_features(scores, names, threshold=0.70)
        
        # Should find feat3 (0.6) and feat5 (0.55)
        self.assertEqual(len(low), 2)
        self.assertEqual(low[0]['name'], 'feat5')  # Lowest first
        self.assertEqual(low[1]['name'], 'feat3')
    
    def test_calculate_similarity_all_approaches(self):
        """Test calculate_similarity wrapper for all approaches"""
        scores = np.array([0.8, 0.9, 0.6])
        names = ['left_elbow_angle', 'right_elbow_angle', 'stick_angle']
        
        # Test simple approach
        result = self.calculator.calculate_similarity(scores, names, self.template, 'simple')
        self.assertIn('actual_score', result)
        self.assertNotIn('category_scores', result)
        
        # Test variance approach
        result = self.calculator.calculate_similarity(scores, names, self.template, 'variance')
        self.assertIn('actual_score', result)
        
        # Test multifactor approach
        result = self.calculator.calculate_similarity(scores, names, self.template, 'multifactor')
        self.assertIn('actual_score', result)
        self.assertIn('category_scores', result)


class TestABTestLogger(unittest.TestCase):
    """Test A/B test logger functionality"""
    
    def setUp(self):
        """Create temporary directory for test logs"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, 'test_ab.json')
        self.logger = ABTestLogger(self.log_file)
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_approach_rotation(self):
        """Test that approaches rotate correctly"""
        approaches = []
        for i in range(6):  # Should cycle through all 3 twice
            approach = self.logger.get_approach_for_session()
            approaches.append(approach)
        
        expected = ['simple', 'variance', 'multifactor', 'simple', 'variance', 'multifactor']
        self.assertEqual(approaches, expected)
    
    def test_log_result(self):
        """Test logging a result"""
        self.logger.log_result(
            approach='simple',
            actual_score=75.0,
            display_score=80.0,
            technique='left_chest_thrust_correct',
            viewpoint='front',
            user_rating=4,
            low_features_count=2
        )
        
        # Check file was created
        self.assertTrue(os.path.exists(self.log_file))
        
        # Check data was logged
        with open(self.log_file, 'r') as f:
            results = json.load(f)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['approach'], 'simple')
        self.assertEqual(results[0]['actual_score'], 75.0)
        self.assertEqual(results[0]['user_rating'], 4)
    
    def test_summary_stats(self):
        """Test summary statistics calculation"""
        # Add data for each approach
        for i in range(10):
            self.logger.log_result('simple', 70.0 + i, 75.0 + i, 'test', 'front')
            self.logger.log_result('variance', 75.0 + i, 80.0 + i, 'test', 'front')
            self.logger.log_result('multifactor', 72.0 + i, 77.0 + i, 'test', 'front')
        
        stats = self.logger.get_summary_stats()
        
        self.assertEqual(stats['simple']['count'], 10)
        self.assertEqual(stats['variance']['count'], 10)
        self.assertEqual(stats['multifactor']['count'], 10)
        
        # Check averages
        self.assertAlmostEqual(stats['simple']['avg_actual_score'], 74.5, places=1)
        self.assertAlmostEqual(stats['variance']['avg_actual_score'], 79.5, places=1)
    
    def test_winner_selection_not_enough_data(self):
        """Test that winner selection requires minimum sessions"""
        # Add only 10 sessions per approach
        for i in range(10):
            self.logger.log_result('simple', 70.0, 75.0, 'test', 'front')
            self.logger.log_result('variance', 75.0, 80.0, 'test', 'front')
            self.logger.log_result('multifactor', 72.0, 77.0, 'test', 'front')
        
        winner = self.logger.should_select_winner(min_sessions_per_approach=50)
        self.assertIsNone(winner)
    
    def test_winner_selection_with_enough_data(self):
        """Test winner selection when enough data collected"""
        # Add 50 sessions per approach
        for i in range(50):
            self.logger.log_result('simple', 70.0, 75.0, 'test', 'front')
            self.logger.log_result('variance', 82.0, 87.0, 'test', 'front')  # Highest
            self.logger.log_result('multifactor', 72.0, 77.0, 'test', 'front')
        
        winner = self.logger.should_select_winner(min_sessions_per_approach=50)
        self.assertEqual(winner, 'variance')  # Should pick highest avg


class TestGenerateLessonTips(unittest.TestCase):
    """Test generate_lesson_tips function"""
    
    def test_basic_tip_generation(self):
        """Test generating tips from low features"""
        low_features = [
            {'name': 'left_elbow_angle', 'score': 0.6, 'index': 0},
            {'name': 'stick_angle', 'score': 0.55, 'index': 1},
        ]
        raw_features = {'left_elbow_angle': 80, 'stick_angle': 40}
        template_means = {'left_elbow_angle': 90, 'stick_angle': 45}
        
        tips = generate_lesson_tips(low_features, raw_features, template_means, max_tips=3)
        
        self.assertEqual(len(tips), 2)
        # Left elbow: raw (80) < mean (90) → should get "low" message (extend arm)
        self.assertIn("Extend your left arm", tips[0])
    
    def test_max_tips_limit(self):
        """Test that max_tips is respected"""
        low_features = [
            {'name': 'left_elbow_angle', 'score': 0.6, 'index': 0},
            {'name': 'right_elbow_angle', 'score': 0.55, 'index': 1},
            {'name': 'stick_angle', 'score': 0.5, 'index': 2},
            {'name': 'left_wrist_height', 'score': 0.45, 'index': 3},
        ]
        raw_features = {
            'left_elbow_angle': 80, 'right_elbow_angle': 80,
            'stick_angle': 40, 'left_wrist_height': 0.4
        }
        template_means = {
            'left_elbow_angle': 90, 'right_elbow_angle': 90,
            'stick_angle': 45, 'left_wrist_height': 0.5
        }
        
        tips = generate_lesson_tips(low_features, raw_features, template_means, max_tips=2)
        
        self.assertEqual(len(tips), 2)  # Should be limited to 2
    
    def test_threshold_filtering(self):
        """Test that features above threshold are filtered out"""
        low_features = [
            {'name': 'left_elbow_angle', 'score': 0.75, 'index': 0},  # Above 70%
            {'name': 'stick_angle', 'score': 0.6, 'index': 1},  # Below 70%
        ]
        raw_features = {'left_elbow_angle': 85, 'stick_angle': 40}
        template_means = {'left_elbow_angle': 90, 'stick_angle': 45}
        
        tips = generate_lesson_tips(low_features, raw_features, template_means, 
                                   max_tips=3, similarity_threshold=0.70)
        
        # Only stick_angle is below 70%
        self.assertEqual(len(tips), 1)
    
    def test_unknown_feature_fallback(self):
        """Test fallback message for unknown features"""
        low_features = [
            {'name': 'unknown_feature', 'score': 0.6, 'index': 0},
        ]
        raw_features = {'unknown_feature': 100}
        template_means = {'unknown_feature': 50}
        
        tips = generate_lesson_tips(low_features, raw_features, template_means)
        
        # Should skip unknown feature, returning empty list
        self.assertEqual(len(tips), 0)
    
    def test_missing_data_fallback(self):
        """Test fallback message when raw/mean data missing"""
        low_features = [
            {'name': 'left_elbow_angle', 'score': 0.4, 'index': 0},  # Very low
        ]
        raw_features = {}  # Missing data
        template_means = {}
        
        tips = generate_lesson_tips(low_features, raw_features, template_means)
        
        # Should generate generic tip due to low score
        self.assertEqual(len(tips), 1)
        self.assertIn("left elbow angle", tips[0].lower())


def run_all_tests():
    """Run all tests and report results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSimilarityCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestABTestLogger))
    suite.addTests(loader.loadTestsFromTestCase(TestGenerateLessonTips))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
