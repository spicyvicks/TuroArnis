"""
Manual stick pattern definitions for all poses.
Define stick position relative to arm direction without needing image analysis.

For each pose, specify:
- hand: 'right' or 'left' (which hand holds the stick)
- stick_arm_angle_degrees: angle offset from forearm direction
    * 0° = stick extends straight along forearm
    * Positive = stick angles away from body/upward
    * Negative = stick angles toward body/downward
- stick_length_ratio: stick length as ratio of shoulder-to-knee distance
    * Typical arnis stick ≈ 0.9 to 1.0

How to fill this out:
1. Look at 2-3 training images for each pose
2. Visually estimate the angle between the stick and forearm
3. Note which hand is used
4. Fill in the values below
"""

MANUAL_STICK_PATTERNS = {
    # BLOCKS
    "1. left_temple_block": {
        "hand": "right",  # Which hand holds the stick
        "stick_arm_angle_degrees": 45,  # Stick angles upward from forearm
        "stick_length_ratio": 0.70,
        "notes": "Stick extends upward and outward to protect left temple"
    },
    
    "2. right_temple_block": {
        "hand": "right",
        "stick_arm_angle_degrees": -45,
        "stick_length_ratio": 0.70,
        "notes": "Mirror of left temple block"
    },
    
    "3. left_elbow_block": {
        "hand": "right",
        "stick_arm_angle_degrees": 0,  # Stick follows forearm
        "stick_length_ratio": 0.70,
        "notes": "Stick extends along forearm for blocking"
    },
    
    "4. right_elbow_block": {
        "hand": "right",
        "stick_arm_angle_degrees": 0,
        "stick_length_ratio": 0.70,
        "notes": "Stick follows forearm"
    },
    
    "8. left_knee_block": {
        "hand": "right",
        "stick_arm_angle_degrees": -30,  # Stick angles downward
        "stick_length_ratio": 0.70,
        "notes": "Stick extends downward to protect knee"
    },
    
    "9. right_knee_block": {
        "hand": "right",
        "stick_arm_angle_degrees": -30,
        "stick_length_ratio": 0.70,
        "notes": "Stick angles downward"
    },
    
    # THRUSTS
    "5. solar_plexus_thrust": {
        "hand": "right",
        "stick_arm_angle_degrees": 0,  # Stick extends straight forward along arm
        "stick_length_ratio": 0.70,
        "notes": "Thrust forward - stick follows arm direction"
    },
    
    "6. left_chest_thrust": {
        "hand": "right",
        "stick_arm_angle_degrees": 0,
        "stick_length_ratio": 0.70,
        "notes": "Thrust to left chest"
    },
    
    "7. right_chest_thrust": {
        "hand": "right",
        "stick_arm_angle_degrees": 0,
        "stick_length_ratio": 0.70,
        "notes": "Thrust to right chest"
    },
    
    "10. left_eye_thrust": {
        "hand": "right",
        "stick_arm_angle_degrees": 10,  # Slightly upward
        "stick_length_ratio": 0.70,
        "notes": "Thrust upward toward eye level"
    },
    
    "11. right_eye_thrust": {
        "hand": "right",
        "stick_arm_angle_degrees": 10,
        "stick_length_ratio": 0.70,
        "notes": "Thrust to eye level"
    },
    
    "12. crown_thrust": {
        "hand": "right",
        "stick_arm_angle_degrees": 60,  # Steeply upward
        "stick_length_ratio": 0.70,
        "notes": "Thrust upward toward crown of head"
    },
}


def get_stick_pattern(pose_class):
    """
    Get stick pattern for a given pose class.
    Handles variations in class name format.
    """
    import re
    
    # Normalize the class name
    normalized = re.sub(r'^\d+\.\s*', '', pose_class)  # Remove number prefix
    normalized = normalized.replace('_correct', '')  # Remove _correct suffix
    
    # Try to find matching pattern
    for key in MANUAL_STICK_PATTERNS.keys():
        clean_key = re.sub(r'^\d+\.\s*', '', key)
        if clean_key == normalized or key == pose_class:
            return MANUAL_STICK_PATTERNS[key]
    
    return None


# Instructions for updating this file:
"""
TO UPDATE STICK PATTERNS:

1. Look at your training images for a specific pose
2. Identify which hand holds the stick (usually right for right-handed)
3. Estimate the angle:
   - Extend a line from elbow through wrist (the forearm)
   - See how much the stick deviates from this line
   - Positive = stick goes away from body
   - Negative = stick goes toward body
   
4. Examples:
   - Temple blocks: stick angled upward ~45°
   - Knee blocks: stick angled downward ~-30°
   - Thrusts: stick follows arm ~0°
   - Crown thrust: stick angled steeply up ~60°

5. Update the values above and save this file
"""
