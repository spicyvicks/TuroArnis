"""
Feedback Analyzer - Provides detailed form correction feedback
Analyzes pose data against target form requirements and generates actionable corrections
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

class FeedbackAnalyzer:
    """Analyzes pose results and generates detailed correction feedback"""
    
    def __init__(self):
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
    
    def analyze(self, result: Dict, target_form: str, confidence_threshold: float = 0.60) -> Dict:
        """
        Analyze pose result and generate feedback
        
        Args:
            result: Pose analysis result dictionary
            target_form: Target form to compare against
            confidence_threshold: Minimum confidence for correct form
            
        Returns:
            Dictionary with feedback analysis:
            {
                'is_correct': bool,
                'confidence': float,
                'errors': List[str],
                'warnings': List[str],
                'suggestions': List[str],
                'error_count': int,
                'severity': str ('ok', 'minor', 'major', 'critical')
            }
        """
        predicted_class = result.get('predicted_class', '').strip()
        confidence = result.get('confidence', 0.0)
        
        feedback = {
            'is_correct': False,
            'confidence': confidence,
            'errors': [],
            'warnings': [],
            'suggestions': [],
            'error_count': 0,
            'severity': 'ok'
        }
        
        #check if predicted class matches target
        if predicted_class == target_form and confidence > confidence_threshold:
            feedback['is_correct'] = True
            feedback['suggestions'].append('Maintain this position')
            return feedback
        
        #analyze errors
        errors = []
        warnings = []
        suggestions = []
        
        #1. grip angle analysis
        grip_errors = self._analyze_grip_angle(result, target_form)
        errors.extend(grip_errors)
        
        #2. joint angle analysis
        joint_errors = self._analyze_joint_angles(result, target_form)
        errors.extend(joint_errors)
        
        #3. posture analysis
        posture_errors = self._analyze_posture(result)
        warnings.extend(posture_errors)
        
        #4. stick detection
        stick_warnings = self._analyze_stick_detection(result)
        warnings.extend(stick_warnings)
        
        #5. confidence-based suggestions
        confidence_suggestions = self._analyze_confidence(predicted_class, target_form, confidence)
        suggestions.extend(confidence_suggestions)
        
        #categorize severity
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
            'warnings': warnings,
            'suggestions': suggestions,
            'error_count': error_count,
            'severity': severity
        })
        
        return feedback
    
    def _analyze_grip_angle(self, result: Dict, target_form: str) -> List[str]:
        """Analyze grip angle and return error messages"""
        errors = []
        grip_angle = result.get('grip_angle')
        
        if grip_angle is None:
            return errors
        
        #determine target range based on form type
        if 'thrust' in target_form:
            target_min, target_max = self.grip_angle_ranges['thrust']
        elif 'block' in target_form:
            target_min, target_max = self.grip_angle_ranges['block']
        else:
            target_min, target_max = self.grip_angle_ranges['default']
        
        if grip_angle < target_min:
            diff = target_min - grip_angle
            if diff > 20:
                errors.append(f"Grip: Extend stick significantly ({diff:.0f}° too narrow)")
            else:
                errors.append(f"Grip: Extend stick slightly ({diff:.0f}° too narrow)")
        elif grip_angle > target_max:
            diff = grip_angle - target_max
            if diff > 20:
                errors.append(f"Grip: Retract stick significantly ({diff:.0f}° too wide)")
            else:
                errors.append(f"Grip: Retract stick slightly ({diff:.0f}° too wide)")
        
        return errors
    
    def _analyze_joint_angles(self, result: Dict, target_form: str) -> List[str]:
        """Analyze joint angles against target form requirements"""
        errors = []
        
        if target_form not in self.joint_angle_targets:
            return errors
        
        live_angles = result.get('live_angles')
        if not live_angles:
            return errors
        
        form_targets = self.joint_angle_targets[target_form]
        
        for joint_name, (target_min, target_max, importance) in form_targets.items():
            if joint_name not in live_angles:
                continue
            
            current_angle = live_angles[joint_name]
            
            if current_angle < target_min:
                diff = target_min - current_angle
                if importance == 'high':
                    errors.append(f"{joint_name.replace('_', ' ').title()}: Extend more ({diff:.0f}° below target)")
                elif diff > 15:  #only report medium importance if significant
                    errors.append(f"{joint_name.replace('_', ' ').title()}: Slightly extend ({diff:.0f}° below)")
            
            elif current_angle > target_max:
                diff = current_angle - target_max
                if importance == 'high':
                    errors.append(f"{joint_name.replace('_', ' ').title()}: Bend more ({diff:.0f}° above target)")
                elif diff > 15:
                    errors.append(f"{joint_name.replace('_', ' ').title()}: Slightly bend ({diff:.0f}° above)")
        
        return errors
    
    def _analyze_posture(self, result: Dict) -> List[str]:
        """Analyze overall posture and body alignment"""
        warnings = []
        
        landmarks = result.get('landmarks_absolute')
        if not landmarks or len(landmarks) < 27:
            return warnings
        
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
        """Check stick detection status"""
        warnings = []
        
        stick_endpoints = result.get('stick_endpoints')
        if stick_endpoints is None:
            warnings.append("Stick: Not detected - ensure stick is visible")
        
        return warnings
    
    def _analyze_confidence(self, predicted_class: str, target_form: str, confidence: float) -> List[str]:
        """Generate suggestions based on prediction confidence"""
        suggestions = []
        
        if predicted_class != target_form:
            #format form names for display
            predicted_display = predicted_class.replace('_correct', '').replace('_', ' ').title()
            target_display = target_form.replace('_correct', '').replace('_', ' ').title()
            
            if confidence > 0.60:
                suggestions.append(f"Detected: {predicted_display} - Switch to {target_display}")
            else:
                suggestions.append(f"Adjust position to match {target_display}")
        elif confidence < 0.60:
            suggestions.append(f"Close to correct - refine position (confidence: {confidence:.0%})")
        
        return suggestions
    
    def get_prioritized_messages(self, feedback: Dict, max_messages: int = 4) -> List[Tuple[str, str]]:
        """
        Get prioritized feedback messages for display
        
        Returns:
            List of (message, type) tuples where type is 'error', 'warning', or 'suggestion'
        """
        messages = []
        
        #priority 1: critical errors (high importance)
        for error in feedback['errors'][:2]:  #max 2 errors
            messages.append((error, 'error'))
        
        #priority 2: warnings
        remaining = max_messages - len(messages)
        if remaining > 0:
            for warning in feedback['warnings'][:remaining]:
                messages.append((warning, 'warning'))
        
        #priority 3: suggestions
        remaining = max_messages - len(messages)
        if remaining > 0:
            for suggestion in feedback['suggestions'][:remaining]:
                messages.append((suggestion, 'suggestion'))
        
        return messages[:max_messages]
