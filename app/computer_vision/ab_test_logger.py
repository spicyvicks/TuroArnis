"""
A/B Test Logger - Track similarity approach effectiveness

Rotates through 3 similarity calculation approaches per session
and collects data for comparison.
"""

import json
import os
from datetime import datetime
from typing import Optional, Dict, List


class ABTestLogger:
    """
    Rotate through similarity approaches and collect comparison data.
    
    Approaches:
    - 'simple': Simple average of hybrid scores
    - 'variance': Variance-weighted by template reliability  
    - 'multifactor': Category breakdown (stick/joints/posture)
    
    Rotation: simple → variance → multifactor → simple (session-based)
    """
    
    APPROACHES = ['simple', 'variance', 'multifactor']
    
    def __init__(self, log_file: str = None):
        """
        Initialize A/B test logger.
        
        Args:
            log_file: Path to JSON log file. Defaults to .planning/ab_test_results.json
        """
        if log_file is None:
            # Store in .planning directory relative to project root
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            log_file = os.path.join(base_dir, '.planning', 'ab_test_results.json')
        
        self.results_file = log_file
        self.index_file = log_file.replace('.json', '_current_index.txt')
        self._ensure_files_exist()
    
    def _ensure_files_exist(self):
        """Create log files if they don't exist."""
        os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
        
        if not os.path.exists(self.results_file):
            with open(self.results_file, 'w') as f:
                json.dump([], f)
        
        if not os.path.exists(self.index_file):
            self._save_current_index(0)
    
    def _load_current_index(self) -> int:
        """Load current approach rotation index."""
        try:
            with open(self.index_file, 'r') as f:
                return int(f.read().strip())
        except (FileNotFoundError, ValueError):
            return 0
    
    def _save_current_index(self, index: int):
        """Save current approach rotation index."""
        with open(self.index_file, 'w') as f:
            f.write(str(index % len(self.APPROACHES)))
    
    def get_approach_for_session(self) -> str:
        """
        Get the similarity approach for the current session.
        
        Rotates through approaches: simple → variance → multifactor → simple
        
        Returns:
            Approach name: 'simple', 'variance', or 'multifactor'
        """
        current_idx = self._load_current_index()
        approach = self.APPROACHES[current_idx]
        
        # Advance index for next session
        self._save_current_index((current_idx + 1) % len(self.APPROACHES))
        
        return approach
    
    def log_result(
        self,
        approach: str,
        actual_score: float,
        display_score: float,
        technique: str,
        viewpoint: str,
        user_rating: Optional[int] = None,
        low_features_count: int = 0
    ):
        """
        Log a similarity result for A/B testing analysis.
        
        Args:
            approach: Which approach was used ('simple', 'variance', 'multifactor')
            actual_score: Real similarity score (0-100)
            display_score: Score with +5% buffer applied
            technique: Name of the technique being practiced
            viewpoint: Camera viewpoint ('front', 'left', 'right')
            user_rating: Optional user satisfaction rating (1-5)
            low_features_count: Number of features below 70% threshold
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'approach': approach,
            'actual_score': actual_score,
            'display_score': display_score,
            'technique': technique,
            'viewpoint': viewpoint,
            'user_rating': user_rating,
            'low_features_count': low_features_count,
            'passed': actual_score >= 65.0
        }
        
        # Load existing results
        try:
            with open(self.results_file, 'r') as f:
                results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            results = []
        
        # Append new entry
        results.append(entry)
        
        # Save updated results
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics for A/B test analysis.
        
        Returns:
            Dict with per-approach statistics:
            {
                'simple': {'count': N, 'avg_score': X, 'pass_rate': Y},
                'variance': {...},
                'multifactor': {...}
            }
        """
        try:
            with open(self.results_file, 'r') as f:
                results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        
        stats = {}
        for approach in self.APPROACHES:
            approach_results = [r for r in results if r.get('approach') == approach]
            
            if approach_results:
                avg_score = sum(r['actual_score'] for r in approach_results) / len(approach_results)
                pass_count = sum(1 for r in approach_results if r.get('passed', False))
                pass_rate = pass_count / len(approach_results)
                avg_rating = sum(r['user_rating'] for r in approach_results if r.get('user_rating')) / max(1, sum(1 for r in approach_results if r.get('user_rating')))
                
                stats[approach] = {
                    'count': len(approach_results),
                    'avg_actual_score': round(avg_score, 2),
                    'avg_display_score': round(avg_score + 5, 2),  # With buffer
                    'pass_rate': round(pass_rate * 100, 1),
                    'avg_user_rating': round(avg_rating, 2) if avg_rating > 0 else None
                }
            else:
                stats[approach] = {
                    'count': 0,
                    'avg_actual_score': None,
                    'avg_display_score': None,
                    'pass_rate': None,
                    'avg_user_rating': None
                }
        
        return stats
    
    def should_select_winner(self, min_sessions_per_approach: int = 50) -> Optional[str]:
        """
        Check if we have enough data to select a winning approach.
        
        Args:
            min_sessions_per_approach: Minimum sessions needed per approach
        
        Returns:
            Winning approach name if enough data, None otherwise
        """
        stats = self.get_summary_stats()
        
        # Check if all approaches have enough data
        for approach in self.APPROACHES:
            if stats.get(approach, {}).get('count', 0) < min_sessions_per_approach:
                return None
        
        # Select winner based on highest average actual score
        best_approach = max(
            self.APPROACHES,
            key=lambda a: stats.get(a, {}).get('avg_actual_score', 0) or 0
        )
        
        return best_approach


def test_ab_logger():
    """Unit tests for ABTestLogger"""
    import tempfile
    import shutil
    
    # Create temporary directory for tests
    temp_dir = tempfile.mkdtemp()
    log_file = os.path.join(temp_dir, 'test_ab_results.json')
    
    try:
        logger = ABTestLogger(log_file)
        
        # Test rotation
        approaches_seen = []
        for i in range(6):  # Should cycle through all 3 twice
            approach = logger.get_approach_for_session()
            approaches_seen.append(approach)
        
        assert approaches_seen == ['simple', 'variance', 'multifactor', 'simple', 'variance', 'multifactor'], \
            f"Rotation failed: {approaches_seen}"
        print("[OK] Approach rotation cycles correctly")
        
        # Test logging
        logger.log_result(
            approach='simple',
            actual_score=75.0,
            display_score=80.0,
            technique='left_chest_thrust_correct',
            viewpoint='front',
            user_rating=4,
            low_features_count=2
        )
        
        logger.log_result(
            approach='variance',
            actual_score=82.0,
            display_score=87.0,
            technique='left_chest_thrust_correct',
            viewpoint='front',
            user_rating=5,
            low_features_count=1
        )
        print("[OK] Results logged successfully")
        
        # Test stats
        stats = logger.get_summary_stats()
        assert 'simple' in stats
        assert 'variance' in stats
        assert stats['simple']['count'] == 1
        assert stats['variance']['count'] == 1
        assert stats['simple']['avg_actual_score'] == 75.0
        assert stats['variance']['avg_actual_score'] == 82.0
        print("[OK] Summary statistics calculated correctly")
        
        # Test winner selection (not enough data yet)
        winner = logger.should_select_winner(min_sessions_per_approach=10)
        assert winner is None, "Should not have winner yet"
        print("[OK] Winner selection requires minimum sessions")
        
        # Add more data to trigger winner (need 50 total per approach)
        # Note: rotation test already cycled through all 3 once, but didn't log
        # So we start from 0 for each
        for i in range(49):  # Add 49 more 'simple' results (total 50 with initial)
            logger.log_result('simple', 70.0 + i, 75.0 + i, 'test', 'front')
        for i in range(49):  # Add 49 more 'variance' results (total 50 with initial)
            logger.log_result('variance', 75.0 + i, 80.0 + i, 'test', 'front')
        for i in range(50):  # Add 50 'multifactor' results (none logged yet)
            logger.log_result('multifactor', 72.0 + i, 77.0 + i, 'test', 'front')
        
        winner = logger.should_select_winner(min_sessions_per_approach=50)
        assert winner is not None, "Should have winner now"
        assert winner in logger.APPROACHES, f"Invalid winner: {winner}"
        print(f"[OK] Winner selected: {winner}")
        
        print("\n[OK] All ABTestLogger tests passed!")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    test_ab_logger()
