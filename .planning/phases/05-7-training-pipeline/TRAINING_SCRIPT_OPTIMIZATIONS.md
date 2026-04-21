---
title: Training Script Optimizations for 4c_train_hybrid_gcn_v2.py
description: Recommended hyperparameter changes to fix overfitting and improve model performance
phase: 5.7
date: 2026-04-21
---

# Training Script Optimizations

Apply these changes to `TuroArnis-ML/hybrid_classifier/4c_train_hybrid_gcn_v2.py` before retraining.

## Summary of Changes

| Parameter | Old Value | New Value | Rationale |
|-----------|-----------|-----------|-----------|
| `HIDDEN_DIM` | 256 | 128 | Reduce overfitting for ~1,871 training images |
| `DROPOUT` | 0.5 | 0.7 | Aggressive regularization (left view had 21% gap) |
| `NODE_EMBED_DIM` | 8 | 16 | Better representation for 35 nodes |
| `NUM_LAYERS` | 3 | 3 | Keep (good depth) |
| BatchNorm | Default | Add `track_running_stats=False` | Fix inference variance |

---

## Change 1: Configuration Constants (Lines ~35-45)

### Original:
```python
# Configuration
HIDDEN_DIM = 256
NUM_LAYERS = 3
DROPOUT = 0.5
LEARNING_RATE = 0.001
EPOCHS = 150
PATIENCE = 20
NODE_EMBED_DIM = 8
```

### Optimized:
```python
# Configuration - Optimized for ~1,871 training samples
HIDDEN_DIM = 128              # Reduced from 256 to prevent overfitting
NUM_LAYERS = 3                # Keep - good depth for feature hierarchy
DROPOUT = 0.7                 # Increased from 0.5 for aggressive regularization
LEARNING_RATE = 0.001         # Keep - stable convergence
EPOCHS = 150                  # Keep - early stopping will prevent over-training
PATIENCE = 15                 # Reduced from 20 to stop earlier on overfit
NODE_EMBED_DIM = 16           # Increased from 8 for better node differentiation
WEIGHT_DECAY = 1e-4           # ADD: L2 regularization
```

---

## Change 2: Model Architecture - BatchNorm Fix (Lines ~55-75)

### Original:
```python
class HybridGCN(torch.nn.Module):
    def __init__(self, num_node_features, num_hybrid_features, num_classes, hidden_dim=256):
        super(HybridGCN, self).__init__()
        
        # Node embedding layer
        self.node_embedding = torch.nn.Embedding(35, NODE_EMBED_DIM)
        
        # GCN layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        # First layer: node_features + embedding -> hidden
        self.convs.append(GCNConv(num_node_features + NODE_EMBED_DIM, hidden_dim))
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(NUM_LAYERS - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))
```

### Optimized (Add track_running_stats=False):
```python
class HybridGCN(torch.nn.Module):
    def __init__(self, num_node_features, num_hybrid_features, num_classes, hidden_dim=128):
        super(HybridGCN, self).__init__()
        
        # Node embedding layer - increased dimension
        self.node_embedding = torch.nn.Embedding(35, NODE_EMBED_DIM)  # Now 16 dims
        
        # GCN layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        # First layer: node_features + embedding -> hidden
        self.convs.append(GCNConv(num_node_features + NODE_EMBED_DIM, hidden_dim))
        # FIX: Disable running stats for small batch stability
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim, track_running_stats=False))
        
        # Hidden layers
        for _ in range(NUM_LAYERS - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            # FIX: Disable running stats for small batch stability
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim, track_running_stats=False))
```

**Why `track_running_stats=False`:** With batch_size=32 and class imbalance, batch statistics vary wildly. Disabling running stats uses batch statistics at train time AND test time, preventing inference variance.

---

## Change 3: Optimizer - Add Weight Decay (Lines ~180-190)

### Original:
```python
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
```

### Optimized:
```python
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
```

**Alternative (AdamW for better weight decay behavior):**
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
```

---

## Change 4: Learning Rate Scheduler (Lines ~185-195)

### Add after optimizer definition:

```python
# ADD: Learning rate scheduler for better convergence
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max',           # Monitor validation accuracy
    factor=0.5,           # Halve LR when plateau
    patience=5,           # Wait 5 epochs before reducing
    verbose=True
)
```

### Update training loop to use scheduler:

Find the validation section (around line ~250) and add:

```python
# After computing val_acc
scheduler.step(val_acc)
```

---

## Change 5: Stratified Sampling for Class Imbalance (Lines ~300-320)

### Original (simple DataLoader):
```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
```

### Optimized (WeightedRandomSampler for class balance):

Add imports at top:
```python
from torch.utils.data import DataLoader, WeightedRandomSampler
```

Replace DataLoader creation:
```python
# Compute sample weights for stratified sampling
labels = [data.y.item() for data in train_dataset]
class_counts = np.bincount(labels, minlength=NUM_CLASSES)
class_weights = 1.0 / (class_counts + 1e-6)
sample_weights = [class_weights[label] for label in labels]

# Create weighted sampler
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    sampler=sampler,  # Use sampler instead of shuffle
    drop_last=True
)
```

**Effect:** Every batch will have roughly equal class representation, fixing the 6.7:1 imbalance bias.

---

## Change 6: Augmentation Strategy (Lines ~350-380)

### If augmentation is in training script, add this control:

```python
# Augmentation configuration
AUGMENTATION = True
AUGMENTATION_FACTOR = 2  # Generate 2 augmented copies per original

# Add noise augmentation to node features (if not already present)
def augment_node_features(node_features, noise_std=0.01):
    """Add small Gaussian noise to node coordinates for regularization."""
    if not AUGMENTATION:
        return node_features
    noise = torch.randn_like(node_features) * noise_std
    return node_features + noise
```

Apply in forward pass (around line ~100):
```python
def forward(self, data):
    x, edge_index, batch = data.x, data.edge_index, data.batch
    
    # Augment node features during training
    if self.training and AUGMENTATION:
        x = augment_node_features(x, noise_std=0.01)
```

---

## Change 7: Early Stopping - Monitor Gap (Lines ~280-300)

### Original (monitors test accuracy only):
```python
if val_acc > best_test_acc:
    best_test_acc = val_acc
    patience_counter = 0
    torch.save(model.state_dict(), best_model_path)
else:
    patience_counter += 1
```

### Optimized (monitor train-test gap to detect overfitting):
```python
# Compute generalization gap
gap = train_acc - val_acc

# Save best model based on validation accuracy
if val_acc > best_test_acc:
    best_test_acc = val_acc
    patience_counter = 0
    torch.save(model.state_dict(), best_model_path)
    print(f"  ✓ Saved best model (val_acc={val_acc:.1f}%, gap={gap:.1f}%)")
else:
    patience_counter += 1

# Early stop if overfitting severely (gap > 25%)
if gap > 25.0:
    print(f"  ⚠ Overfitting detected (gap={gap:.1f}%), early stopping...")
    break

if patience_counter >= PATIENCE:
    print(f"  Early stopping after {epoch} epochs")
    break
```

---

## Change 8: Model Initialization - Better Defaults (Lines ~60-70)

### Add after model creation:

```python
# Initialize weights for better convergence
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, GCNConv):
        # GCN layers use Glorot initialization by default in PyG
        pass

model.apply(init_weights)
print("✓ Applied Xavier initialization")
```

---

## Full Training Command

After applying changes, run:

```bash
cd TuroArnis-ML
python hybrid_classifier/4c_train_hybrid_gcn_v2.py --merged \
    --epochs 150 \
    --patience 15 \
    --dropout 0.7 \
    --hidden-dim 128
```

---

## Expected Improvements

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Left view overfitting | 21% gap | < 10% gap |
| Overall accuracy | ~65% | > 75% |
| Class balance bias | 6.7:1 | ~1:1 (via sampling) |
| Inference variance | High | Low (fixed BatchNorm) |
| Training stability | Early stops | Full 150 epochs |

---

## Validation Checklist

After training completes:

- [ ] Train-test gap < 15% for all viewpoints
- [ ] No class has < 60% accuracy
- [ ] Neutral class precision > 70% (not over-triggering)
- [ ] Model files < 50MB each (hidden_dim reduction helps)
- [ ] Training completed 100+ epochs without early stopping
