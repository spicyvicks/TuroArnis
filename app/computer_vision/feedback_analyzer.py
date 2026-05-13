"""
Feedback Analyzer - Provides detailed form correction feedback
Analyzes pose data against target form requirements and generates actionable corrections
"""

import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple

class FeedbackAnalyzer:
    """Analyzes pose results and generates detailed correction feedback"""
    
    # Per-viewpoint confidence thresholds (defaults if config not found)
    DEFAULT_CONFIDENCE_THRESHOLDS = {
        'front': 0.60,
        'left': 0.55,
        'right': 0.65,
    }
    
    def __init__(self):
        # Load per-viewpoint confidence thresholds from config
        self.confidence_thresholds = self._load_confidence_thresholds()
        #grip angle targets for different forms
        self.grip_angle_ranges = {
            'default': (80, 120),
            'thrust': (85, 115),  #more strict for thrusts
            'block': (75, 125),   #more lenient for blocks
        }
        
        #joint angle targets for specific forms
        #format: {form_name: {joint_name: (min, max, importance)}}
        self.joint_angle_targets = self._initialize_joint_targets()
        
        #posture checking landmarks
        self.SHOULDER_LEFT = 11
        self.SHOULDER_RIGHT = 12
        self.HIP_LEFT = 23
        self.HIP_RIGHT = 24
        self.ELBOW_LEFT = 13
        self.ELBOW_RIGHT = 14
        self.WRIST_LEFT = 15
        self.WRIST_RIGHT = 16
        self.KNEE_LEFT = 25
        self.KNEE_RIGHT = 26
    
    def _load_confidence_thresholds(self) -> Dict[str, float]:
        """Load per-viewpoint confidence thresholds from gcn_model_config.json"""
        thresholds = dict(self.DEFAULT_CONFIDENCE_THRESHOLDS)
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'gcn_model_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                for vp, model_info in config.get('models', {}).items():
                    if 'confidence_threshold' in model_info:
                        thresholds[vp] = model_info['confidence_threshold']
        except Exception as e:
            print(f"[FeedbackAnalyzer] Could not load config thresholds: {e}")
        return thresholds
    
    def get_confidence_threshold(self, viewpoint: str = 'front') -> float:
        """Get the confidence threshold for a specific viewpoint"""
        return self.confidence_thresholds.get(viewpoint, 0.60)
    
    def _initialize_joint_targets(self) -> Dict:
        """Initialize joint angle targets for each form"""
        return {
            #thrust forms - emphasis on arm extension
            'crown_thrust_correct': {
                'right_elbow': (160, 180, 'high'),  #should be extended
                'left_elbow': (90, 120, 'medium'),
                'right_shoulder': (80, 110, 'high'),
            },
            'left_chest_thrust_correct': {
                'left_elbow': (160, 180, 'high'),
                'right_elbow': (90, 120, 'medium'),
                'left_shoulder': (70, 100, 'high'),
            },
            'right_chest_thrust_correct': {
                'right_elbow': (160, 180, 'high'),
                'left_elbow': (90, 120, 'medium'),
                'right_shoulder': (70, 100, 'high'),
            },
            'left_eye_thrust_correct': {
                'left_elbow': (160, 180, 'high'),
                'left_shoulder': (85, 115, 'high'),
            },
            'right_eye_thrust_correct': {
                'right_elbow': (160, 180, 'high'),
                'right_shoulder': (85, 115, 'high'),
            },
            'solar_plexus_thrust_correct': {
                'right_elbow': (150, 180, 'high'),
                'right_shoulder': (60, 90, 'high'),
            },
            
            #block forms - emphasis on defensive positioning
            'left_elbow_block_correct': {
                'left_elbow': (80, 110, 'high'),
                'right_elbow': (140, 170, 'medium'),
            },
            'right_elbow_block_correct': {
                'right_elbow': (80, 110, 'high'),
                'left_elbow': (140, 170, 'medium'),
            },
            'left_temple_block_correct': {
                'left_elbow': (100, 130, 'high'),
                'left_shoulder': (110, 140, 'high'),
            },
            'right_temple_block_correct': {
                'right_elbow': (100, 130, 'high'),
                'right_shoulder': (110, 140, 'high'),
            },
            'left_knee_block_correct': {
                'left_elbow': (140, 170, 'high'),
                'right_knee': (160, 180, 'medium'),
            },
            'right_knee_block_correct': {
                'right_elbow': (140, 170, 'high'),
                'left_knee': (160, 180, 'medium'),
            },
        }
    
    
    def analyze(self, result: Dict, target_form: str,
                confidence_threshold: float = None, viewpoint: str = 'front',
                gcn_engine=None) -> Dict:
        """
        Analyze pose result and generate feedback with structured corrections.

        Returns:
            {
                'is_correct': bool,
                'confidence': float,
                'errors': List[str],
                'corrections': List[Dict],
                'warnings': List[str],
                'suggestions': List[str],
                'error_count': int,
                'severity': str
            }
        """
        from app.computer_vision.feedback_mapper import get_corrections

        if confidence_threshold is None:
            confidence_threshold = self.get_confidence_threshold(viewpoint)

        predicted_class = result.get('predicted_class', '').strip()
        confidence = result.get('confidence', 0.0)

        feedback = {
            'is_correct': False,
            'confidence': confidence,
            'errors': [],
            'corrections': [],
            'warnings': [],
            'suggestions': [],
            'error_count': 0,
            'severity': 'ok'
        }

        # ── Body visibility gate (checked before is_correct) ─────────────────
        # Run this early so a partial-body view never gets marked as correct.
        landmarks_abs = result.get('landmarks_absolute')
        frame_w = result.get('frame_w', 640)
        frame_h = result.get('frame_h', 480)
        body_visible = True
        body_warning = None
        if landmarks_abs and len(landmarks_abs) >= 27:
            is_vis, vis_count, total, missing = self.is_full_body_visible(
                landmarks_abs, frame_w, frame_h
            )
            if not is_vis:
                body_visible = False
                missing_str = ', '.join(missing[:3])
                body_warning = f"Step back — body not fully in frame ({missing_str} cut off)"
        print(f"[ANALYZE] target={target_form} | predicted={predicted_class} | conf={confidence:.2f} vs thresh={confidence_threshold:.2f} | body_visible={body_visible} | gcn={'yes' if gcn_engine else 'no'} | global_feats={'yes' if result.get('global_features') else 'no'}")

        # ── Correct pose ──────────────────────────────────────────────────────
        if body_visible and predicted_class == target_form and confidence > confidence_threshold:
            feedback['is_correct'] = True
            feedback['suggestions'].append('Maintain this position')
            print(f"[ANALYZE] → EXCELLENT (is_correct=True, returning early)")
            return feedback

        errors = []
        corrections = []
        warnings = []
        suggestions = []

        # ── Model-driven corrections (hybrid feature approach) ─────────────────
        global_features = result.get('global_features')
        used_hybrid = False

        if gcn_engine is not None and global_features and target_form != 'neutral':
            correction_data = gcn_engine.get_feature_corrections(
                global_features, target_form
            )
            if correction_data:
                raw_corrections = get_corrections(
                    raw_features=correction_data['raw_values'],
                    hybrid_scores=correction_data['hybrid_scores'],
                    feature_names=correction_data['feature_names'],
                    template_means=correction_data['template_means'],
                    max_corrections=3,
                )
                for msg, score in raw_corrections:
                    errors.append(msg)
                    corrections.append({
                        'joint': 'body',
                        'action': 'adjust',
                        'value': round(1.0 - score, 2),
                        'message': msg
                    })
                used_hybrid = True
                print(f"[ANALYZE] → hybrid corrections ({len(errors)}): {errors}")
        if not used_hybrid:
            print(f"[ANALYZE] → fallback to joint-angle analysis (gcn={'yes' if gcn_engine else 'no'}, global_feats={'yes' if result.get('global_features') else 'no'})")

        # ── Fallback: original hardcoded joint-angle analysis ─────────────────
        if not used_hybrid:
            joint_errors, joint_corrections = self._analyze_joint_angles(result, target_form)
            errors.extend(joint_errors)
            corrections.extend(joint_corrections)
            grip_errors, grip_corrections = self._analyze_grip_angle(result, target_form)
            errors.extend(grip_errors)
            corrections.extend(grip_corrections)

        # ── Always-on checks ──────────────────────────────────────────────────
        # If body is cut off, prepend that warning and skip further posture analysis
        if body_warning:
            warnings.insert(0, body_warning)
        else:
            warnings.extend(self._analyze_posture(result))
        warnings.extend(self._analyze_stick_detection(result))
        suggestions.extend(self._analyze_confidence(predicted_class, target_form, confidence))

        error_count = len(errors)
        warning_count = len(warnings)
        if error_count == 0 and warning_count == 0:
            severity = 'ok'
        elif error_count == 0:
            severity = 'minor'
        elif error_count <= 2:
            severity = 'major'
        else:
            severity = 'critical'

        feedback.update({
            'errors': errors,
            'corrections': corrections,
            'warnings': warnings,
            'suggestions': suggestions,
            'error_count': error_count,
            'severity': severity
        })
        return feedback

    @staticmethod
    def _joint_direction(joint_name: str, action: str) -> str:
        """Convert joint + mechanical action into a natural directional cue.

        Examples:
            _joint_direction('right_elbow', 'extend') -> 'Straighten your right elbow'
            _joint_direction('left_shoulder', 'flex')  -> 'Lower your left arm'
        """
        parts = joint_name.split('_')
        side = parts[0].title() if parts else ''
        region = parts[1] if len(parts) > 1 else joint_name

        if region == 'elbow':
            if action == 'extend':
                return f"Straighten your {side} elbow"
            return f"Bend your {side} elbow more"
        elif region == 'shoulder':
            if action == 'extend':
                return f"Raise your {side} arm higher"
            return f"Lower your {side} arm"
        elif region == 'knee':
            if action == 'extend':
                return f"Straighten your {side} knee"
            return f"Bend your {side} knee more"
        elif region == 'hip':
            if action == 'extend':
                return f"Straighten your {side} hip"
            return f"Bend your {side} hip more"
        else:
            # Fallback for unknown joints
            verb = "Extend" if action == 'extend' else "Bend"
            return f"{verb} your {joint_name.replace('_', ' ')}"

    def _analyze_grip_angle(self, result: Dict, target_form: str) -> Tuple[List[str], List[Dict]]:
        """Analyze grip angle and return error messages and corrections"""
        errors = []
        corrections = []
        grip_angle = result.get('grip_angle')
        
        if grip_angle is None:
            return errors, corrections
        
        #determine target range based on form type
        if 'thrust' in target_form:
            target_min, target_max = self.grip_angle_ranges['thrust']
        elif 'block' in target_form:
            target_min, target_max = self.grip_angle_ranges['block']
        else:
            target_min, target_max = self.grip_angle_ranges['default']
        
        if grip_angle < target_min:
            diff = target_min - grip_angle
            msg = "Bring stick grip closer to you"
            errors.append(msg)
            corrections.append({
                'joint': 'wrist', # Generalizing to wrist/hand
                'action': 'extend_grip',
                'value': diff,
                'message': msg
            })
        elif grip_angle > target_max:
            diff = grip_angle - target_max
            msg = "Extend stick grip further out"
            errors.append(msg)
            corrections.append({
                'joint': 'wrist',
                'action': 'retract_grip',
                'value': diff,
                'message': msg
            })

        return errors, corrections
    
    def _analyze_joint_angles(self, result: Dict, target_form: str) -> Tuple[List[str], List[Dict]]:
        """Analyze joint angles against target form requirements"""
        errors = []
        corrections = []
        
        if target_form not in self.joint_angle_targets:
            return errors, corrections
        
        live_angles = result.get('live_angles')
        if not live_angles:
            return errors, corrections
        
        form_targets = self.joint_angle_targets[target_form]
        
        for joint_name, (target_min, target_max, importance) in form_targets.items():
            if joint_name not in live_angles:
                continue
            
            current_angle = live_angles[joint_name]
            
            if current_angle < target_min:
                diff = target_min - current_angle
                msg = self._joint_direction(joint_name, 'extend')

                if importance == 'high' or diff > 15:
                    errors.append(msg)
                    corrections.append({
                        'joint': joint_name,
                        'action': 'extend',
                        'value': diff,
                        'message': msg
                    })

            elif current_angle > target_max:
                diff = current_angle - target_max
                msg = self._joint_direction(joint_name, 'flex')

                if importance == 'high' or diff > 15:
                    errors.append(msg)
                    corrections.append({
                        'joint': joint_name,
                        'action': 'flex', # "bend"
                        'value': diff,
                        'message': msg
                    })
        
        return errors, corrections
    
    # Key landmark indices required for a 'full body' check
    _FULL_BODY_LANDMARKS = {
        0:  'nose',
        11: 'left shoulder',
        12: 'right shoulder',
        15: 'left wrist',
        16: 'right wrist',
        23: 'left hip',
        24: 'right hip',
        27: 'left ankle',
        28: 'right ankle',
    }

    def is_full_body_visible(
        self,
        landmarks_abs: list,
        frame_w: int,
        frame_h: int,
        min_visible: int = 7,
        margin: int = 10,
    ) -> tuple:
        """
        Check whether the key body landmarks are all within the frame bounds.

        Returns:
            (is_visible: bool, visible_count: int, total_checked: int,
             missing_names: list[str])
        """
        visible = 0
        missing = []
        for idx, name in self._FULL_BODY_LANDMARKS.items():
            if idx >= len(landmarks_abs):
                missing.append(name)
                continue
            x, y = landmarks_abs[idx][0], landmarks_abs[idx][1]
            # (-1, -1) is the sentinel pose_analyzer stores for landmarks with
            # visibility < 0.3 (cut off or heavily occluded)
            is_sentinel = (x == -1 and y == -1)
            in_bounds = (not is_sentinel) and (margin < x < frame_w - margin) and (margin < y < frame_h - margin)
            if in_bounds:
                visible += 1
            else:
                missing.append(name)
        total = len(self._FULL_BODY_LANDMARKS)
        return (visible >= min_visible, visible, total, missing)

    def _analyze_posture(self, result: Dict) -> List[str]:
        """Analyze overall posture and body alignment"""
        warnings = []

        landmarks = result.get('landmarks_absolute')
        if not landmarks or len(landmarks) < 27:
            return warnings

        # ── Full-body visibility check ────────────────────────────────────────
        frame_w = result.get('frame_w', 640)
        frame_h = result.get('frame_h', 480)
        is_visible, visible_count, total, missing = self.is_full_body_visible(
            landmarks, frame_w, frame_h
        )
        if not is_visible:
            missing_str = ', '.join(missing[:3])
            warnings.append(f"Step back — body not fully in frame ({missing_str} cut off)")
            return warnings  # Skip further posture checks if body is cut off

        try:
            #check shoulder alignment (should be relatively level)
            left_shoulder = landmarks[self.SHOULDER_LEFT]
            right_shoulder = landmarks[self.SHOULDER_RIGHT]
            shoulder_height_diff = abs(left_shoulder[1] - right_shoulder[1])
            
            #if shoulders are more than 30 pixels apart vertically
            if shoulder_height_diff > 30:
                warnings.append("Posture: Level your shoulders")
            
            #check hip alignment
            left_hip = landmarks[self.HIP_LEFT]
            right_hip = landmarks[self.HIP_RIGHT]
            hip_height_diff = abs(left_hip[1] - right_hip[1])
            
            if hip_height_diff > 25:
                warnings.append("Posture: Align your hips")
            
            #check spine alignment (shoulders should be above hips horizontally)
            shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            hip_center_x = (left_hip[0] + right_hip[0]) / 2
            lean_diff = abs(shoulder_center_x - hip_center_x)
            
            if lean_diff > 40:
                if shoulder_center_x < hip_center_x:
                    warnings.append("Posture: Leaning backward, center your stance")
                else:
                    warnings.append("Posture: Leaning forward, center your stance")
        
        except (IndexError, TypeError):
            pass
        
        return warnings
    
    def _analyze_stick_detection(self, result: Dict) -> List[str]:
        """Check stick detection status — only surface this if there is nothing
        more important to say (it's a low-priority warning)."""
        warnings = []
        stick_endpoints = result.get('stick_endpoints')
        if stick_endpoints is None:
            warnings.append("Make sure your stick is visible")
        return warnings
    
    def _analyze_confidence(self, predicted_class: str, target_form: str, confidence: float) -> List[str]:
        """Generate suggestions based on prediction confidence.
        Only emit something useful — no generic 'You did X, aim for Y' noise.
        """
        suggestions = []

        if predicted_class == target_form and confidence < 0.55:
            # Right technique but low confidence — coach to sharpen the form
            suggestions.append(f"Close — sharpen your form ({confidence:.0%} confidence)")
        elif predicted_class and predicted_class not in ('N/A', 'neutral') \
                and target_form and predicted_class != target_form:
            # Wrong technique detected in lesson mode — keep it short
            target_display = target_form.replace('_correct', '').replace('_', ' ').title()
            suggestions.append(f"Try to match {target_display} position")

        return suggestions
    
    def get_prioritized_messages(self, feedback: Dict, max_messages: int = 4) -> List[Tuple[str, str]]:
        """
        Get prioritized feedback messages for display.

        Priority order: errors > warnings (skip stick warning when errors
        are present) > suggestions.

        Returns:
            List of (message, type) tuples where type is 'error', 'warning',
            or 'suggestion'.
        """
        messages = []

        # Priority 1: actionable joint / form errors (max 2)
        for error in feedback['errors'][:2]:
            messages.append((error, 'error'))

        # Priority 2: warnings — but skip the stick warning when we already
        # have real joint corrections to show (it just adds noise).
        if len(messages) < max_messages:
            remaining = max_messages - len(messages)
            has_errors = len(feedback['errors']) > 0
            for warning in feedback['warnings'][:remaining]:
                # Suppress the low-priority stick warning when there is
                # already something more meaningful to say.
                if has_errors and 'stick' in warning.lower():
                    continue
                messages.append((warning, 'warning'))

        # Priority 3: suggestions (only if there's still room)
        if len(messages) < max_messages:
            remaining = max_messages - len(messages)
            for suggestion in feedback['suggestions'][:remaining]:
                messages.append((suggestion, 'suggestion'))

        return messages[:max_messages]
