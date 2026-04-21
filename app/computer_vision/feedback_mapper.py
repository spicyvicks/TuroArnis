"""
Feedback Mapper - Maps low-scoring hybrid features to actionable corrections.

Each hybrid feature is a Gaussian similarity score (0=bad, 1=perfect). 
When a score is low, we know the user's value deviates from the ideal template.
The direction (too_high vs too_low) is determined by comparing the raw value 
against the template mean.
"""

from typing import List, Tuple, Dict

# Threshold below which a feature is "wrong enough" to mention
CORRECTION_THRESHOLD = 0.55

# Map from feature name → (message if value too low, message if value too high)
# "low" = user's raw value < template mean
# "high" = user's raw value > template mean
FEATURE_MESSAGES: Dict[str, Tuple[str, str]] = {
    # Joint angles
    'left_elbow_angle': (
        "Extend your left arm more",
        "Bend your left elbow more"
    ),
    'right_elbow_angle': (
        "Extend your right arm more",
        "Bend your right elbow more"
    ),
    'left_shoulder_angle': (
        "Raise your left arm higher",
        "Lower your left arm"
    ),
    'right_shoulder_angle': (
        "Raise your right arm higher",
        "Lower your right arm"
    ),
    'left_knee_angle': (
        "Straighten your left leg",
        "Bend your left knee more"
    ),
    'right_knee_angle': (
        "Straighten your right leg",
        "Bend your right knee more"
    ),

    # Wrist / elbow heights (relative to hip)
    'left_wrist_height': (
        "Raise your left hand higher",
        "Lower your left hand"
    ),
    'right_wrist_height': (
        "Raise your right hand higher",
        "Lower your right hand"
    ),
    'left_elbow_height': (
        "Raise your left elbow",
        "Drop your left elbow"
    ),
    'right_elbow_height': (
        "Raise your right elbow",
        "Drop your right elbow"
    ),

    # Stick height / position
    'stick_tip_height': (
        "Raise the tip of the stick higher",
        "Lower the tip of the stick"
    ),
    'stick_grip_height': (
        "Raise the grip of the stick",
        "Lower the grip of the stick"
    ),

    # Lateral position (relative to hip center)
    'left_wrist_x': (
        "Move your left hand more to the left",
        "Move your left hand more to the right"
    ),
    'right_wrist_x': (
        "Move your right hand more to the right",
        "Move your right hand more to the left"
    ),
    'stick_tip_x': (
        "Direct the stick tip further out",
        "Bring the stick tip closer in"
    ),
    'stick_grip_x': (
        "Move the grip further from your body",
        "Keep the grip closer to your body"
    ),

    # Stick orientation
    'stick_angle': (
        "Angle the stick more upward",
        "Angle the stick more downward"
    ),
    'stick_dx': (
        "Point the stick more horizontally",
        "Point the stick more diagonally"
    ),
    'stick_dy': (
        "Point the stick more vertically",
        "Point the stick more horizontally"
    ),

    # Stick tip vs body landmarks
    'tip_vs_nose': (
        "The stick tip should be above nose level",
        "The stick tip is too high — aim lower"
    ),
    'tip_vs_shoulder': (
        "Raise the stick tip above shoulder level",
        "Lower the stick tip to shoulder level"
    ),
    'tip_vs_hip': (
        "The stick tip should be above hip level",
        "Lower the stick tip"
    ),

    # Right hand vs body landmarks
    'r_hand_vs_nose': (
        "Raise your right hand above nose level",
        "Lower your right hand"
    ),
    'r_hand_vs_shoulder': (
        "Raise your right hand to shoulder level",
        "Lower your right hand"
    ),
    'r_hand_vs_hip': (
        "Keep your right hand above hip level",
        "Lower your right hand"
    ),

    # Lateral stick metrics
    'tip_side': (
        "Extend the stick tip further to the side",
        "Bring the stick tip closer to your centre"
    ),
    'grip_side': (
        "Move the grip further from your centre",
        "Bring the grip closer to your centre"
    ),

    # Stance
    'foot_stagger': (
        "Step your rear foot further back",
        "Bring your feet closer together"
    ),

    # Misc
    'hands_distance': (
        "Spread your hands further apart",
        "Bring your hands closer together"
    ),
    'stick_length': (
        "Show more of the stick — extend your grip",
        "Shorten your grip on the stick"
    ),
}


def get_corrections(
    raw_features: Dict[str, float],
    hybrid_scores: 'np.ndarray',
    feature_names: List[str],
    template_means: Dict[str, float],
    threshold: float = CORRECTION_THRESHOLD,
    max_corrections: int = 3,
) -> List[Tuple[str, float]]:
    """
    Return the most important corrections sorted by severity.

    Args:
        raw_features:    dict of feature_name → raw value for the user
        hybrid_scores:   array of Gaussian similarity scores (0–1) in feature_names order
        feature_names:   ordered list matching hybrid_scores indices
        template_means:  dict of feature_name → template mean (from feature_templates.json)
        threshold:       scores below this trigger a correction message
        max_corrections: maximum number of messages to return

    Returns:
        List of (message, score) sorted worst-first, length ≤ max_corrections
    """
    import numpy as np

    corrections = []
    for i, fname in enumerate(feature_names):
        score = float(hybrid_scores[i])
        if score >= threshold:
            continue  # Good enough, skip

        messages = FEATURE_MESSAGES.get(fname)
        if messages is None:
            continue  # No human-readable mapping defined

        raw_val = raw_features.get(fname)
        mean_val = template_means.get(fname)
        if raw_val is None or mean_val is None:
            continue

        direction = 'low' if raw_val < mean_val else 'high'
        msg = messages[0] if direction == 'low' else messages[1]
        corrections.append((msg, score))

    # Sort by worst score first, then trim
    corrections.sort(key=lambda x: x[1])
    return corrections[:max_corrections]


def generate_lesson_tips(
    low_features: List[Dict],
    raw_features: Dict[str, float],
    template_means: Dict[str, float],
    max_tips: int = 3,
    similarity_threshold: float = 0.70
) -> List[str]:
    """
    Generate 2-3 actionable tips for features with similarity below threshold.
    
    This is used in lesson mode to provide contextual feedback based on
    similarity scores, not just correction detection.
    
    Args:
        low_features: List of feature dicts with 'name' and 'score' keys
                     (from SimilarityCalculator.identify_low_features)
        raw_features: Dict of feature_name → user's raw value
        template_means: Dict of feature_name → template mean value
        max_tips: Maximum number of tips to return (default 3)
        similarity_threshold: Similarity threshold (default 0.70 = 70%)
    
    Returns:
        List of actionable tip strings (max max_tips)
    """
    tips = []
    
    for feat in low_features:
        fname = feat['name']
        score = feat['score']
        
        # Skip if above threshold (shouldn't happen if filtered properly)
        if score >= similarity_threshold:
            continue
        
        # Get messages for this feature
        if fname not in FEATURE_MESSAGES:
            continue
        
        messages = FEATURE_MESSAGES[fname]
        
        # Determine direction based on raw vs mean
        raw_val = raw_features.get(fname)
        mean_val = template_means.get(fname)
        
        if raw_val is None or mean_val is None:
            # Fallback: use generic message based on score severity
            if score < 0.50:
                tip = f"Focus on your {fname.replace('_', ' ')}"
            else:
                tip = f"Adjust your {fname.replace('_', ' ')}"
            tips.append(tip)
            continue
        
        direction = 'low' if raw_val < mean_val else 'high'
        tip = messages[0] if direction == 'low' else messages[1]
        tips.append(tip)
        
        # Stop once we have max_tips
        if len(tips) >= max_tips:
            break
    
    return tips
