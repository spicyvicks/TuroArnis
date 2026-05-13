"""
HybridGCN V5 Deployment Model Architecture
Exact match to training model from 4e_train_hybrid_gcn_v2_with_synthetic_v5.py
- 6-dim node features
- global_mean_pool (no masking)
- 46 hybrid features (33 base + 13 signed)
- 13 classes (including neutral)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class HybridGCN(nn.Module):
    """
    GCN with Node-Specific Features + Global Hybrid Context
    - GCN processes spatial node features
    - Global hybrid features provide expert knowledge
    - Both are combined for final classification
    """
    def __init__(self, num_node_features, num_hybrid_features, num_classes=13,
                 hidden_dim=128, num_layers=3, dropout=0.5, node_embed_dim=8):
        super(HybridGCN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.node_embed_dim = node_embed_dim
        
        # Node identity embeddings (35 nodes: 0-34)
        self.node_embedding = nn.Embedding(35, node_embed_dim)
        
        # GCN layers for node features (geometric + embedding)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.convs.append(GCNConv(num_node_features + node_embed_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim, track_running_stats=False))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim, track_running_stats=False))
        
        # MLP for global hybrid features
        self.hybrid_mlp = nn.Sequential(
            nn.Linear(num_hybrid_features, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Fusion and classification
        self.fc1 = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, data):
        """
        Args:
            data: PyG Data object with:
                - x: [num_nodes, num_node_features] node features
                - edge_index: [2, num_edges] graph connectivity
                - batch: [num_nodes] batch assignment
                - hybrid_features: [batch_size, num_hybrid_features] global features
        
        Returns:
            logits: [batch_size, num_classes]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        hybrid_features = data.hybrid_features
        
        batch_size = batch.max().item() + 1
        
        # Generate node indices (0-34 for each graph in batch)
        node_indices = torch.arange(35, device=x.device).unsqueeze(0).expand(batch_size, -1)
        node_emb = self.node_embedding(node_indices).view(-1, self.node_embed_dim)
        x = torch.cat([x, node_emb], dim=-1)
        
        # Process node features with GCN
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout_layer(x_new)
            if x_new.size(-1) == x.size(-1):
                x = x_new + x
            else:
                x = x_new
        
        # Global pooling (mean over all nodes)
        x_pool = global_mean_pool(x, batch)
        
        # Process global hybrid features
        # PyG Batch concatenates 1D graph-level tensors along dim 0.
        # Reshape [batch_size * num_hybrid_features] -> [batch_size, num_hybrid_features].
        if hybrid_features.dim() == 1:
            hybrid_features = hybrid_features.view(batch_size, -1)
        hybrid_out = self.hybrid_mlp(hybrid_features)
        
        # Fusion
        combined = torch.cat([x_pool, hybrid_out], dim=-1)
        x_out = F.relu(self.fc1(combined))
        x_out = self.dropout_layer(x_out)
        logits = self.fc2(x_out)
        return logits


def load_deployment_model(checkpoint_path, device='cpu'):
    """Load deployment model from checkpoint.

    Handles both full deployment checkpoints and raw training checkpoints
    that may be missing num_node_features, num_hybrid_features, or num_classes.
    Missing values are inferred directly from the state dict shapes.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get('config', {})
    state = ckpt['model_state_dict']

    # V5 defaults
    defaults = {
        'num_node_features': 6,
        'num_hybrid_features': 46,
        'num_classes': 13,
        'hidden_dim': 128,
        'num_layers': 3,
        'dropout': 0.5,
        'node_embed_dim': 8,
    }

    merged = {**defaults, **config}

    # Infer from state dict when keys are still missing / mismatched
    if 'hybrid_mlp.0.weight' in state:
        merged['num_hybrid_features'] = state['hybrid_mlp.0.weight'].shape[1]
        merged['hidden_dim'] = state['hybrid_mlp.0.weight'].shape[0] * 2
    if 'fc2.weight' in state:
        merged['num_classes'] = state['fc2.weight'].shape[0]
    if 'convs.0.lin.weight' in state:
        conv_in = state['convs.0.lin.weight'].shape[1]
        merged['num_node_features'] = conv_in - merged.get('node_embed_dim', 8)
    # Count conv layers from state dict keys
    conv_keys = [k for k in state.keys() if k.startswith('convs.') and k.endswith('.lin.weight')]
    if conv_keys:
        merged['num_layers'] = len(conv_keys)

    model = HybridGCN(
        num_node_features=merged['num_node_features'],
        num_hybrid_features=merged['num_hybrid_features'],
        num_classes=merged['num_classes'],
        hidden_dim=merged['hidden_dim'],
        num_layers=merged['num_layers'],
        dropout=merged['dropout'],
        node_embed_dim=merged['node_embed_dim']
    ).to(device)

    model.load_state_dict(state)
    model.eval()

    return model, ckpt.get('class_names', []), merged


# Skeleton edges for Arnis pose graph
SKELETON_EDGES = [
    (11, 12), (12, 11), (11, 23), (23, 11), (12, 24), (24, 12),
    (23, 24), (24, 23), (11, 13), (13, 11), (13, 15), (15, 13),
    (12, 14), (14, 12), (14, 16), (16, 14), (23, 25), (25, 23),
    (25, 27), (27, 25), (24, 26), (26, 24), (26, 28), (28, 26),
    (15, 33), (33, 15), (16, 33), (33, 16), (33, 34), (34, 33)
]
