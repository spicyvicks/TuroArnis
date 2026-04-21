---
title: Neutral Buffer Class Implementation
description: Architecture and behavior of the 13th class for handling invalid poses and transitions
phase: 5.7
date: 2026-04-21
---

# Neutral Buffer Class

This document describes the neutral class implementation, which serves as a "soft rejection" mechanism for poses that don't match any valid Arnis technique.

## Overview

The GCN classifier uses **13 classes**: 12 Arnis techniques + 1 **neutral buffer class**. The neutral class prevents false positives by providing a "none of the above" prediction option when the model is uncertain or the pose is invalid.

## Class Structure

### 12 Arnis Technique Classes

| Index | Class Name | Category | Description |
|-------|------------|----------|-------------|
| 0 | `front_left_chest_thrust` | Thrust | Front view, left hand leading chest thrust |
| 1 | `front_right_chest_thrust` | Thrust | Front view, right hand leading chest thrust |
| 2 | `front_crown_thrust` | Thrust | Front view, overhead crown thrust |
| 3 | `left_crown_thrust` | Thrust | Left side view, overhead crown thrust |
| 4 | `right_crown_thrust` | Thrust | Right side view, overhead crown thrust |
| 5 | `left_jab` | Thrust | Left side view, jabbing motion |
| 6 | `right_jab` | Thrust | Right side view, jabbing motion |
| 7 | `front_left_downward_block` | Block | Front view, left hand downward block |
| 8 | `front_right_downward_block` | Block | Front view, right hand downward block |
| 9 | `left_outward_block` | Block | Left side view, outward block |
| 10 | `right_outward_block` | Block | Right side view, outward block |
| 11 | `left_waist_block` | Block | Left side view, waist-level block |

### Neutral Buffer Class

| Index | Class Name | Purpose |
|-------|------------|---------|
| 12 | `neutral` | Sink for invalid poses, transitions, non-Arnis movements |

**Note:** Neutral is **not** a technique to be performed—it's a classification outcome meaning "no valid technique detected."

## Neutral Class Purpose

### 1. Wrong/Incorrect Poses During Practice

When a student attempts a technique but performs it incorrectly:
- Form is too far from any template to match
- Body positioning doesn't resemble any known technique
- Stick handling is improper

**Behavior:** Classify as neutral → Show "No technique detected" → No feedback (since pose is unrecognizable)

### 2. Non-Arnis Movements

Users may perform actions unrelated to Arnis:
- Walking, turning, adjusting position
- Waving hands, scratching head, adjusting clothing
- Talking with hand gestures
- Picking up or putting down the stick

**Behavior:** Classify as neutral → Continue monitoring → Wait for valid Arnis pose

### 3. Transition States

Between techniques, users pass through intermediate poses:
- Recovering from a thrust
- Preparing for the next block
- Resetting to guard position
- Switching hands on the stick

**Behavior:** Classify as neutral → Don't generate feedback during transitions → Resume classification when pose stabilizes

### 4. Low-Confidence Predictions

When model confidence is insufficient:
- Highest class probability below threshold (e.g., 0.65)
- Multiple classes have similar probabilities
- Feature pattern doesn't match any template strongly

**Behavior:** Override argmax → Predict neutral → Prevents false classification

## Inference Behavior

### Confidence Threshold Fallback

```python
# In gcn_inference.py
CLASS_NAMES = [
    'front_left_chest_thrust', 'front_right_chest_thrust',
    'front_crown_thrust', 'left_crown_thrust', 'right_crown_thrust',
    'left_jab', 'right_jab',
    'front_left_downward_block', 'front_right_downward_block',
    'left_outward_block', 'right_outward_block', 'left_waist_block',
    'neutral'  # Index 12
]

def classify_with_neutral_fallback(features, confidence_threshold=0.65):
    """Classify with neutral fallback for low confidence."""
    probs = model.predict_proba(features)  # Shape: (13,)
    max_prob = np.max(probs)
    predicted_class = np.argmax(probs)
    
    # Low confidence → neutral
    if max_prob < confidence_threshold:
        return 12, probs  # Return neutral class index
    
    return predicted_class, probs
```

### UI Display

| Classification | UI Display | Feedback |
|----------------|------------|----------|
| Technique (0-11) | Technique name (e.g., "Left Crown Thrust") | Specific corrections |
| Neutral (12) | "No technique detected" or "—" | None (or "Ready for next technique") |

### Calibration Goals

| Metric | Target | Measurement |
|--------|--------|-------------|
| False positive rate on correct poses | < 15% | Correct technique classified as neutral |
| True positive rate on invalid poses | > 80% | Invalid pose classified as neutral |
| Transition detection | < 500ms | Time to classify neutral after technique ends |

## Training Considerations

### Training Data Collection

Neutral class examples come from:

1. **Random Poses**
   - Standing with arms at sides
   - Walking, turning
   - Non-martial-arts poses
   - Sitting, bending

2. **Transition Frames**
   - Mid-movement between techniques
   - Stick transfers between hands
   - Recovery positions

3. **Incorrect Technique Attempts**
   - Student practicing with poor form
   - Deliberately wrong poses (training data augmentation)
   - Partially completed techniques

4. **Non-Arnis Activities**
   - Adjusting clothing or equipment
   - Gesturing while talking
   - Preparing/cooling down

### Training Approach

```python
# In 4c_train_hybrid_gcn_v2.py
NUM_CLASSES = 13  # 12 techniques + neutral

# Neutral class has no template (not used in template matching)
# GCN learns to associate neutral with:
# - Low-confidence feature patterns
# - Feature vectors far from any technique template
# - Incoherent body-stick configurations
```

### No Template for Neutral

Unlike technique classes, neutral has **no reference template** in `feature_templates.json`:
- Not used for template-based similarity scoring
- GCN learns neutral purely from training examples
- Neutral acts as a statistical "sink" for outliers

**Rationale:** Neutral isn't a specific pose—it's the absence of a recognizable pose. Creating a template would incorrectly suggest there's a "correct way to be neutral."

### Class Balance

| Class | Training Examples | Notes |
|-------|-------------------|-------|
| Technique classes | ~50-100 each | Balanced across viewpoints |
| Neutral | ~150-200 | Slightly more to prevent over-triggering |

**Goal:** Neutral should trigger readily but not dominate predictions.

## Model Architecture

### Output Layer

```python
# In model_architecture.py
class HybridGCN(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes=13):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)  # 13 outputs
```

### Class Name Configuration

```python
# Model architecture defines class names
CLASS_NAMES = [
    'front_left_chest_thrust', 'front_right_chest_thrust',
    'front_crown_thrust', 'left_crown_thrust', 'right_crown_thrust',
    'left_jab', 'right_jab',
    'front_left_downward_block', 'front_right_downward_block',
    'left_outward_block', 'right_outward_block', 'left_waist_block',
    'neutral'
]
```

## Validation Checklist

- [ ] Model outputs 13 probabilities (not 12)
- [ ] `CLASS_NAMES` includes 'neutral' at index 12
- [ ] Neutral false positive rate < 15% on correct poses
- [ ] Neutral triggers appropriately during transitions
- [ ] UI handles neutral gracefully (no crashes, appropriate message)
- [ ] No feedback generated for neutral classification

## Integration with Similarity Scoring

In lesson mode, similarity scores are computed only for the **target technique**:

```python
# Similarity calculation excludes neutral
def compute_similarity(features, target_technique):
    """Compare user pose to target technique template."""
    template = templates[target_technique]  # Only technique classes have templates
    similarity = template_match(features, template)
    return similarity  # 0-100% score
```

If user pose is classified as neutral during a lesson:
- Similarity score will be very low (< 30%)
- UI shows "Try the technique"
- No specific corrections (pose not recognized)

## Historical Context

The neutral class was added after observing:
- 28% false positive rate on random poses (without neutral)
- Users confused by incorrect technique labels during transitions
- Need for "soft failure" instead of forced classification

The 13-class architecture was restored after a temporary regression to 12 classes (see `.continue-here.md` for class alignment history).

## References

- Class alignment: `.continue-here.md` (Blocking issue: Class alignment gap)
- Quick fixes: `.planning/todos/pending/training-quality-gates.md`
- Inference implementation: `app/computer_vision/gcn_inference.py`
- Model architecture: `app/models/gcn/model_architecture.py`
