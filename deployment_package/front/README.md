# Front Viewpoint Deployment Package

This directory contains models, templates, and inference code for **front-viewpoint** Arnis technique classification.

## Models

| Model | File | Template | Note |
|-------|------|----------|------|
| v5 (standard) | `models/model_front_v5_standard.pth` | Standard | Front v5 model |
| v6 (standard) | `models/model_front_v6_standard.pth` | Standard | Front v6 production model |

## Templates

| File | Description | Count |
|------|-------------|-------|
| `src/feature_templates.json` | Standard front templates | 13 |
| `src/feature_templates_mirrored.json` | Mirrored front templates | 13 |

## Architecture Files

- `src/model_v5.py` — v5 HybridGCN (no masking, 46 hybrid features)
- `src/model_v6.py` — v6 HybridGCN (node masking, 49 hybrid features, has_stick flag)

Both include:
- **Batch inference bugfix** (`hybrid_features.view(batch_size, -1)`)
- **Smart checkpoint loading** (infers dimensions from state dict when config is incomplete)

## File Structure

```
front/
├── models/
│   ├── model_front_v5_standard.pth   # Front v5 model
│   └── model_front_v6_standard.pth   # Front v6 model
├── src/
│   ├── model_v5.py                  # v5 architecture (with batch bugfix)
│   ├── model_v6.py                  # v6 architecture (with batch bugfix)
│   ├── feature_templates.json       # Standard templates for front viewpoint
│   ├── feature_templates_mirrored.json  # Mirrored templates for front viewpoint
│   └── inference_example.py         # Verified working demo
└── README.md
```

## Usage

See `src/inference_example.py` for a complete loading demo.

Quick start:

```python
from deployment_package.front.src.model_v6 import load_deployment_model

model, class_names, config = load_deployment_model(
    'deployment_package/front/models/model_front_v6_standard.pth',
    device='cpu'
)
```

## Important Notes

1. **Model architecture bugfix:** The `model_v5.py` and `model_v6.py` in this directory include a fix for batched inference (`hybrid_features.view(batch_size, -1)`). The original files in `deployment_package/src/` do NOT have this fix and will fail on `batch_size > 1`.

2. **Smart checkpoint loading:** The `load_deployment_model()` function infers `num_node_features`, `num_hybrid_features`, and `num_classes` directly from the saved state dict. This handles checkpoints that were saved without complete config metadata.

3. **Original files preserved:** The original front deployment files in `deployment_package/src/` and `deployment_package/models/` remain unchanged for backward compatibility.

## Comparison with Left Viewpoint

| Aspect | Front | Left |
|--------|-------|------|
| Training data | ~2,500 samples | ~1,240 samples |
| Best accuracy | Higher (more data) | 88.70% (v5 mirrored) |
| Hidden dim | 128 | 256 |
| Architecture versions | v5, v6 | v5, v6 |

See `deployment_package/left/README.md` for left viewpoint details.
