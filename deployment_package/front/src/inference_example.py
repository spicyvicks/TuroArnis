"""
Front Viewpoint Inference Example
Demonstrates loading and running front-viewpoint models.
"""

import sys
from pathlib import Path

# Add paths for imports
script_dir = Path(__file__).parent
front_dir = script_dir.parent  # deployment_package/front/
repo_root = front_dir.parent.parent  # repo root
# Add lower-priority paths first
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'deployment_package' / 'src'))
# Add local script_dir LAST so it has highest priority (index 0)
sys.path.insert(0, str(script_dir))

import torch
import json

# For v6 inference (recommended for production)
from model_v6 import load_deployment_model, SKELETON_EDGES

import numpy as np


def demo_load_models():
    """Demonstrate loading both front viewpoint models."""
    base_dir = Path(__file__).parent.parent  # deployment_package/front/
    
    print("="*60)
    print("FRONT VIEWPOINT MODEL LOADING DEMO")
    print("="*60)
    
    # Load v6 standard (production model)
    print("\n1. Loading v6 standard model (production)...")
    v6_model_path = base_dir / "models" / "model_front_v6_standard.pth"
    v6_templates_path = base_dir / "src" / "feature_templates.json"
    
    if v6_model_path.exists():
        model_v6, _, config_v6 = load_deployment_model(v6_model_path, device='cpu')
        print(f"   OK: v6 model loaded")
        print(f"   Hidden dim: {config_v6.get('hidden_dim', 'N/A')}")
        print(f"   Num layers: {config_v6.get('num_layers', 'N/A')}")
        print(f"   Dropout: {config_v6.get('dropout', 'N/A')}")
        print(f"   Num node features: {config_v6.get('num_node_features', 'N/A')}")
        print(f"   Num hybrid features: {config_v6.get('num_hybrid_features', 'N/A')}")
        print(f"   Num classes: {config_v6.get('num_classes', 'N/A')}")
    else:
        print(f"   SKIP: {v6_model_path} not found")
    
    # Load v5 standard
    print("\n2. Loading v5 standard model...")
    v5_model_path = base_dir / "models" / "model_front_v5_standard.pth"
    v5_templates_path = base_dir / "src" / "feature_templates.json"
    
    if v5_model_path.exists():
        from model_v5 import load_deployment_model as load_v5
        model_v5, _, config_v5 = load_v5(v5_model_path, device='cpu')
        print(f"   OK: v5 model loaded")
        print(f"   Hidden dim: {config_v5.get('hidden_dim', 'N/A')}")
        print(f"   Num layers: {config_v5.get('num_layers', 'N/A')}")
        print(f"   Dropout: {config_v5.get('dropout', 'N/A')}")
        print(f"   Num node features: {config_v5.get('num_node_features', 'N/A')}")
        print(f"   Num hybrid features: {config_v5.get('num_hybrid_features', 'N/A')}")
        print(f"   Num classes: {config_v5.get('num_classes', 'N/A')}")
    else:
        print(f"   SKIP: {v5_model_path} not found")
    
    # Check templates
    print("\n3. Checking templates...")
    if v6_templates_path.exists():
        with open(v6_templates_path) as f:
            templates = json.load(f)
        print(f"   Standard templates: {len(templates)} keys")
        for k in sorted(templates.keys())[:3]:
            print(f"     {k}")
    
    mirrored_path = base_dir / "src" / "feature_templates_mirrored.json"
    if mirrored_path.exists():
        with open(mirrored_path) as f:
            templates = json.load(f)
        print(f"   Mirrored templates: {len(templates)} keys")
        for k in sorted(templates.keys())[:3]:
            print(f"     {k}")
    
    print("\n" + "="*60)
    print("All front viewpoint models loaded successfully!")
    print("="*60)


if __name__ == '__main__':
    demo_load_models()
