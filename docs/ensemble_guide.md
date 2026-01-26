# Ensemble Model Documentation

## What is an Ensemble Model?

An **ensemble model** combines predictions from multiple machine learning models to produce better results than any single model. Think of it like getting a second (and third) opinion from doctors - the collective wisdom is often more accurate than any individual.

## Why Use Ensemble Models?

### The Problem with Single Models
- Each model architecture has **biases** and **blind spots**
- DNN: Great at complex patterns but can overfit
- Random Forest: Robust but may miss subtle patterns
- XGBoost: Powerful but sensitive to hyperparameters

### The Ensemble Solution
By combining different model types, we:
- ✅ **Reduce overfitting** - averaging smooths out individual model biases
- ✅ **Increase robustness** - less sensitive to data variations  
- ✅ **Capture diverse patterns** - each model learns differently
- ✅ **Improve accuracy** - typically 2-10% gain over best single model

---

## How Ensemble Models Work

### 1️⃣ Soft Voting (Probability Averaging)

**Best for:** Well-calibrated probability outputs

**Process:**
```
Step 1: Each model outputs probability distribution
  DNN:     [0.1, 0.7, 0.2]  ← 70% confident it's class 2
  RF:      [0.2, 0.6, 0.2]  ← 60% confident it's class 2  
  XGBoost: [0.3, 0.5, 0.2]  ← 50% confident it's class 2

Step 2: Average probabilities (with optional weights)
  Equal weights (1/3 each):
  Average: [(0.1+0.2+0.3)/3, (0.7+0.6+0.5)/3, (0.2+0.2+0.2)/3]
         = [0.2, 0.6, 0.2]

Step 3: Pick class with highest average probability
  Final prediction: Class 2 (60% average confidence)
```

**Advantages:**
- Uses full information (confidence levels)
- More nuanced decision-making
- Better calibrated uncertainty estimates

**Custom Weights Example:**
If XGBoost has 70% accuracy but DNN only has 55%, you can weight them accordingly:
```python
weights = [0.3, 0.2, 0.5]  # DNN:30%, RF:20%, XGB:50%
```

---

### 2️⃣ Hard Voting (Majority Vote)

**Best for:** When probability calibration is questionable

**Process:**
```
Step 1: Each model predicts a single class
  DNN:     Class 2
  RF:      Class 2
  XGBoost: Class 1

Step 2: Count votes
  Class 1: 1 vote
  Class 2: 2 votes

Step 3: Pick class with most votes
  Final prediction: Class 2 (majority wins)
```

**Advantages:**
- Simple and interpretable
- Robust to poorly calibrated probabilities
- Less computational overhead

---

## Script Logic Breakdown

### `EnsembleClassifier` Class

#### Initialization (`__init__`)
```python
1. Auto-select best models (if not specified)
   - Group models by type (DNN, RF, XGBoost)
   - Pick highest accuracy from each group
   
2. Load models from disk
   - DNN: Load .keras file (TensorFlow)
   - RF/XGBoost: Load .joblib file (scikit-learn)
   
3. Load shared preprocessing
   - StandardScaler (feature normalization)
   - LabelEncoder (class labels)
   
4. Setup weights
   - Normalize to sum to 1.0
   - Default: equal weights
```

#### Prediction (`predict`)
```python
SOFT VOTING PATH:
1. Scale input features → X_scaled
2. For each model:
   - Get probability distribution
   - Apply weight
3. Average all weighted probabilities
4. Take argmax → predicted class

HARD VOTING PATH:  
1. Scale input features → X_scaled
2. For each model:
   - Get single class prediction
3. Count votes (with optional weights)
4. Return majority class
```

#### Evaluation (`evaluate`)
```python
1. Make predictions on test set
2. Calculate accuracy score
3. Generate classification report
   - Precision, recall, F1 per class
   - Support (samples per class)
4. Return metrics
```

---

## Usage Guide

### Method 1: Interactive Mode (Recommended for First Time)

```bash
python training/ensemble_model.py
```

**Interactive prompts guide you through:**
1. Model selection (auto or manual)
2. Custom weights (optional)
3. Voting strategy (soft/hard)
4. Feature type (angles/coordinates)

### Method 2: Programmatic Usage

```python
from training.ensemble_model import EnsembleClassifier, evaluate_ensemble

# Auto-select best models with soft voting
ensemble, accuracy = evaluate_ensemble(
    voting='soft',
    weights=None  # Equal weights
)

# Manual model selection with custom weights
ensemble = EnsembleClassifier(
    model_versions=['v019_ang4_xgb', 'v020_ang4_rf', 'v015_dnn'],
    voting='soft',
    weights=[0.4, 0.3, 0.3]  # XGB gets more weight
)

# Make predictions
predictions = ensemble.predict(X_test)

# Get individual model contributions
contributions = ensemble.get_model_contributions(X_test)
for contrib in contributions:
    print(f"{contrib['model']}: {contrib['predictions'][0]} (conf: {contrib['confidence'][0]:.2f})")
```

### Method 3: Custom Script Integration

```python
# In main_app.py or pose_analyzer.py
from training.ensemble_model import EnsembleClassifier

# Initialize once at startup
self.ensemble = EnsembleClassifier(
    model_versions=None,  # Auto-select
    voting='soft',
    weights=None
)

# Use in prediction loop
def classify_pose(self, features):
    prediction = self.ensemble.predict(features.reshape(1, -1))
    return prediction[0]
```

---

## Expected Results

### Typical Performance Gains

Based on ensemble research and your current models:

| Scenario | Individual Best | Ensemble Expected | Improvement |
|----------|----------------|-------------------|-------------|
| Low diversity models | 55% | 57-60% | +2-5% |
| High diversity models | 55% | 60-65% | +5-10% |
| Optimal tuning | 55% | 62-68% | +7-13% |

**Your Context:**
- Current best: ~55% (DNN)
- RF/XGBoost likely different: ~50-65%
- **Expected ensemble:** 58-65% accuracy

### When Ensemble May NOT Help

⚠️ **Warning:** Ensemble is less effective when:
- All models have similar errors (low diversity)
- One model is vastly superior (80%+ vs 60%)
- Small dataset (overfitting risk)
- Models trained on identical features

---

## Advanced Tips

### Optimizing Weights

```python
# Grid search for best weights
from itertools import product

best_acc = 0
best_weights = None

# Try different weight combinations
for w1 in [0.2, 0.3, 0.4]:
    for w2 in [0.2, 0.3, 0.4]:
        w3 = 1.0 - w1 - w2
        if w3 > 0:
            ensemble = EnsembleClassifier(weights=[w1, w2, w3])
            acc, _, _ = ensemble.evaluate(X_test, y_test)
            if acc > best_acc:
                best_acc = acc
                best_weights = [w1, w2, w3]

print(f"Best weights: {best_weights} → {best_acc:.2%}")
```

### Analyzing Model Agreement

```python
# See where models disagree
contributions = ensemble.get_model_contributions(X_test)

for i in range(len(X_test)):
    preds = [c['predictions'][i] for c in contributions]
    if len(set(preds)) > 1:  # Models disagree
        print(f"Sample {i}: disagreement {preds}")
        print(f"True label: {y_test[i]}")
        # These samples may need more training data
```

### Ensemble for Production

```python
# Save ensemble configuration
ensemble_config = {
    'model_versions': ['v019_ang4_xgb', 'v020_ang4_rf'],
    'voting': 'soft',
    'weights': [0.6, 0.4]
}

import json
with open('models/ensemble_config.json', 'w') as f:
    json.dump(ensemble_config, f)

# Load in production
with open('models/ensemble_config.json', 'r') as f:
    config = json.load(f)
    
ensemble = EnsembleClassifier(**config)
```

---

## Troubleshooting

### Issue: "No models found"
**Solution:** Train at least 2 different model types first
```bash
python training/model_manager.py
# Option 1 → Train new model → Select RF
# Option 1 → Train new model → Select XGBoost
```

### Issue: Low ensemble accuracy
**Causes:**
1. Models too similar → Train with different features/hyperparameters
2. One model dominates → Adjust weights manually
3. Data quality issues → Check feature extraction

### Issue: Slow predictions
**Solution:** Use hard voting (faster) or reduce number of models
```python
# Fast ensemble: only use RF + XGBoost (skip slow DNN)
ensemble = EnsembleClassifier(
    model_versions=['v019_ang4_xgb', 'v020_ang4_rf'],
    voting='hard'  # Fast
)
```

---

## Next Steps

1. **Run the script** to establish baseline ensemble performance
2. **Experiment with weights** to find optimal combination
3. **Train more diverse models** if accuracy gain is small
4. **Integrate into main app** once satisfied with results
5. **Monitor real-world performance** and retrain as needed

---

## Key Takeaways

✅ Ensemble combines strengths, reduces weaknesses  
✅ Soft voting: Better for well-calibrated probabilities  
✅ Hard voting: Simpler, more robust  
✅ Expected gain: 2-10% over best single model  
✅ Works best with diverse model architectures  
✅ Auto-selection feature makes it easy to get started
