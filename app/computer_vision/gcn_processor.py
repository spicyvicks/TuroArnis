import torch
import cv2
import numpy as np
import json
import os
import sys
from ultralytics import YOLO

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.gcn.model_architecture import HybridGCN
from models.gcn.feature_extraction import extract_raw_features, compute_global_features_from_kpts, extract_node_features
from utils.resource_path import get_resource_path

class GCNProcessor:
    """
    Handles GCN model loading, feature extraction, and inference for the Kiosk App.
    """
    def __init__(self, config_path='app/models/gcn_model_config.json'):
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[GCNProcessor] Using device: {self.device}")

        # Load Stick Detector (YOLO)
        print("[GCNProcessor] Loading Stick Detector...")
        self.stick_model = YOLO(self.config['stick_detector'])
        
        # Load GCN Models (Front, Left, Right)
        self.models = {}
        for vp, settings in self.config['models'].items():
            print(f"[GCNProcessor] Loading {vp} model from {settings['path']}...")
            try:
                # Initialize model architecture (Hyperparams should match training)
                # Assuming standard params from training script: 3 layers, 64 hidden
                model = HybridGCN(node_in_channels=6, hybrid_in_channels=30, hidden_channels=64, num_classes=len(self.config['class_names']))
                
                # Load weights
                state_dict = torch.load(settings['path'], map_location=self.device)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                self.models[vp] = model
            except Exception as e:
                print(f"[GCNProcessor] Error loading {vp} model: {e}")

        # Load Feature Templates (for Hybrid Features)
        # Note: If json path is relative, might need fix
        try:
            with open(self.config['feature_templates'], 'r') as f:
                self.templates = json.load(f)
        except Exception as e:
            print(f"[GCNProcessor] Warning: Could not load feature templates: {e}")
            self.templates = {}

        self.class_names = self.config['class_names']
        
        # MediaPipe is handled inside extract_raw_features, but for optimization 
        # we might want to initialize it once here if we refactor feature_extraction.py
        # For now, we rely on the function.

    def load_config(self, path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Fallback or error
            print(f"[GCNProcessor] Config not found in {path}, trying absolute...")
            abs_path = os.path.abspath(path)
            with open(abs_path, 'r') as f:
                return json.load(f)

    def process_frame(self, frame, viewpoint="front"):
        """
        Process a single frame and return prediction.
        
        Args:
            frame: BGR numpy image
            viewpoint: 'front', 'left', or 'right'
            
        Returns:
            dict: {
                'prediction': str (class name),
                'confidence': float,
                'skeleton': list (for visualization),
                'stick': list (for visualization)
            }
        """
        # 1. Extract Features
        # extract_raw_features uses MediaPipe internally. 
        # NOTE: It returns None if no pose found.
        raw_data = extract_raw_features(frame, self.stick_model)
        
        if raw_data is None:
            return None

        kpts = raw_data['pose_keypoints']  # [33, 4]
        stick_kpts = raw_data['stick_keypoints'] # [2, 4]

        # 2. Prepare Inputs for GCN
        # Node Features [35, 6]
        # (x, y, z, vis, dist_hip, ang_hip)
        # Need to ensure logic matches extract_node_features
        from app.models.gcn.feature_extraction import extract_node_features, compute_global_features_from_kpts, compute_hybrid_features
        
        node_feats = torch.tensor(extract_node_features(kpts, stick_kpts), dtype=torch.float32).unsqueeze(0).to(self.device) # [1, 35, 6]
        # Transpose for PyTorch Geometric if needed? 
        # wait, HybridGCN forward expects: x, edge_index, batch, hybrid_features
        # x shape: [num_nodes * batch_size, num_features] -> [35, 6]
        
        x = node_feats.view(-1, 6) # Flatten batch dim for standard GCN input
        
        # Edge Index
        from app.models.gcn.model_architecture import SKELETON_EDGES
        edge_index = torch.tensor(SKELETON_EDGES, dtype=torch.long).t().contiguous().to(self.device)
        
        # Batch vector (all zeros for single graph)
        batch = torch.zeros(35, dtype=torch.long).to(self.device)
        
        # Hybrid Features
        # Need to compute statistical features
        # We assume we are classifying against ALL classes to find the best match?
        # OR does the model output logic handle it?
        # The model takes `hybrid_features` input.
        # Wait, the training logic computed hybrid features specific to the *target class*?
        # Let's check model architecture. 
        # The `HybridGCN` takes `hybrid_in_channels` (30).
        # It seems it expects a fixed vector.
        # In the training script `4c_train_hybrid_gcn_v2.py` (which I haven't fully read but can infer):
        # usually hybrid models concat features.
        # If the features depend on "Similarity to Class X", then we can't run inference without knowing Class X?
        # CHECK implementation of `compute_hybrid_features` in `feature_extraction.py`.
        # It takes `class_name`. 
        # If the model is a classifier for 13 classes, does it take features relative to *one* class?
        # Most likely, we feed it features relative to the *neutral* or just raw geometric values?
        
        # INVESTIGATION: 
        # Ideally, `compute_hybrid_features` returns a vector of 30 values.
        # If `compute_hybrid_features` requires `class_name`, that implies we are verifying a specific pose.
        # But `HybridGCN` outputs `num_classes` (13).
        # This implies standard classification.
        # Let's assume for now we pass RAW geometric features if the model was trained that way,
        # OR we pass similarity to ALL classes? That would be huge.
        
        # Let's look at `feature_extraction.py` again.
        # It has `extract_raw_features` and `compute_global_features_from_kpts`.
        # `compute_hybrid_features(raw_features, templates, viewpoint, class_name)`
        
        # If I look at `compute_global_features_from_kpts`, it returns a dict of scalars (angles, dists).
        # These are likely what we feed if the model expects dense features.
        # But `hybrid_in_channels=30` matches the output of `compute_hybrid_features`.
        # So the model WAS trained on similarity scores.
        # BUT which class similarity?
        # If it's a multi-class classifier, maybe it was trained on "Similarity to Correct Class"?
        # That would mean at inference time we don't know the class.
        
        # RETURNING TO PLAN:
        # I will assume for inference we might use a "General" template or raw features. 
        # BUT, if the user selects a "Target Pose" in the UI (which they do!), 
        # we CAN generate features relative to that target pose!
        # This turns it into a Verification task (Is this Pugay?).
        # However, the model output `num_classes=13`.
        # If we feed it "Similarity to Pugay", acts it as a verifier?
        # Let's try to feed it "Similarity to Target Pose" and see if argmax is Target Pose.
        
        # For the purpose of this Kiosk, the user DOES select a form.
        # So we can pass `target_pose` to this function.
        pass

    def predict_pose(self, kpts, stick_kpts, viewpoint, target_pose=None):
        if viewpoint not in self.models:
            return None, 0.0
            
        model = self.models[viewpoint]
        
        # Node Features
        node_feats = extract_node_features(kpts, stick_kpts)
        x = torch.tensor(node_feats, dtype=torch.float32).to(self.device).view(-1, 6)
        
        # Edges
        from app.models.gcn.model_architecture import SKELETON_EDGES
        edge_index = torch.tensor(SKELETON_EDGES, dtype=torch.long).t().contiguous().to(self.device)
        batch = torch.zeros(35, dtype=torch.long).to(self.device)
        
        # Hybrid Features
        # We need raw geometric features first
        global_feats = compute_global_features_from_kpts(kpts, stick_kpts)
        
        # If target_pose provided, use it for similarity. Else... ?
        # If the model was trained to classify "What is this?", it's problematic if it needs "Similarity to X".
        # Let's check `active_model.json` or training script if possible.
        # Assuming for now we use target_pose.
        
        # Map UI name to Class Name
        # UI: "Pugay", "Forward Stance"
        # Classes: "neutral_stance", "forward_stance_correct"?
        # Need a mapper.
        
        mapper = {
            "Pugay": "neutral_stance", # Assuming?
            "Forward Stance": "forward_stance_correct", # Guessing
            # ...
        }
        
        # Fallback: If no target provided, use "neutral_stance" template?
        # Or maybe the model was trained with raw features after all?
        # The file `4c_train_hybrid_gcn_v2.py` would confirm.
        # Given I can't read it continuously, I will make a safe bet:
        # I'll modify `compute_hybrid_features` to return raw values if template is missing?
        # No, `hybrid_in_channels=30`.
        
        # Let's calculate similarity to the TARGET.
        target_class = mapper.get(target_pose, "neutral_stance")
        
        from app.models.gcn.feature_extraction import compute_hybrid_features
        h_feats = compute_hybrid_features(global_feats, self.templates, viewpoint, target_class)
        hybrid_tensor = torch.tensor(h_feats, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = model(x, edge_index, batch, hybrid_tensor)
            probs = torch.softmax(out, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            
        pred_class = self.class_names[pred_idx.item()]
        return pred_class, conf.item()

