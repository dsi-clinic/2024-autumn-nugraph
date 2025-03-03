"""NuGraph3 feature enhancers for improved Michel and diffuse classification"""
import torch
from torch import nn
from torch_geometric.utils import degree, softmax
from torch_scatter import scatter_sum
from torch_geometric.nn import MessagePassing
from .types import Data

class MichelEnhancer(nn.Module):
    """
    NuGraph3 Michel electron enhancer module

    Improves Michel electron classification by focusing on track endpoints
    and their relationship to muon tracks. Michel electrons appear at muon
    track endpoints, so we enhance features of hits near endpoints.

    Args:
        hit_features: Number of hit node features
    """
    def __init__(self, hit_features: int):
        super().__init__()

        # A more lightweight network with intermediate dimension reduction
        self.endpoint_net = nn.Sequential(
            nn.Linear(hit_features, hit_features // 2),
            nn.ReLU(),
            nn.Linear(hit_features // 2, hit_features),
        )
        
        # Create trainable parameters for feature weighting
        self.feature_weights = nn.Parameter(torch.ones(2))
        self.feature_norm = nn.LayerNorm(hit_features)
        
    def forward(self, data: Data) -> None:
        """
        NuGraph3 Michel enhancer forward pass with improved efficiency

        Args:
            data: Graph data object
        """
        edge_index = data["hit", "delaunay-planar", "hit"].edge_index
        src, dst = edge_index
        
        # Get node degrees (number of connections per hit)
        node_degree = degree(src, data["hit"].num_nodes)
        
        # Identify potential track endpoints (hits with few connections)
        endpoint_score = torch.clamp(1.0 / (node_degree + 1.0), 0.0, 1.0)
        
        # Rather than iterative propagation, use a one-step message passing
        edge_weight = softmax(endpoint_score[src], dst, num_nodes=data["hit"].num_nodes)
        endpoint_propagation = scatter_sum(edge_weight * endpoint_score[src], dst, dim=0, dim_size=data["hit"].num_nodes)
        
        # Create compact feature matrix that focuses on endpoint characteristics
        x_orig = data["hit"].x
        
        # Apply feature-wise weighting (scaled to sum to 1.0)
        weights = torch.softmax(self.feature_weights, dim=0)
        
        # Process with reduced dimensionality network and use endpoint information
        endpoint_factor = weights[0] * endpoint_score.unsqueeze(1) + weights[1] * endpoint_propagation.unsqueeze(1)
        enhanced = self.endpoint_net(x_orig) * endpoint_factor
        
        # Mix original and enhanced features with layernorm for stability
        # This is a residual connection that helps prevent loss of information
        alpha = 0.2  # Small contribution from enhanced features to avoid instability
        data["hit"].x = self.feature_norm(x_orig + alpha * enhanced)


class DiffuseMP(MessagePassing):
    """
    Message passing module for diffuse feature enhancement
    
    This implementation is more efficient than manual operations.
    """
    def __init__(self):
        super().__init__(aggr="mean")
        
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)


class DiffuseEnhancer(nn.Module):
    """
    NuGraph3 diffuse enhancer module

    Improves diffuse classification by analyzing local hit clustering patterns.
    Diffuse hits typically have less coherent local structure compared to 
    showers or tracks, so we compute clustering metrics to enhance their features.

    Args:
        hit_features: Number of hit node features
    """
    def __init__(self, hit_features: int):
        super().__init__()
        
        # Use message passing for more efficient feature computation
        self.mp = DiffuseMP()
        
        # A more lightweight network with intermediate dimension reduction
        self.feature_net = nn.Sequential(
            nn.Linear(hit_features, hit_features // 2),
            nn.ReLU(),
            nn.Linear(hit_features // 2, hit_features),
        )
        
        # Use layer normalization for numerical stability
        self.feature_norm = nn.LayerNorm(hit_features)

    def forward(self, data: Data) -> None:
        """
        NuGraph3 diffuse enhancer forward pass with improved efficiency

        Args:
            data: Graph data object
        """
        # Get edge index for hits
        edge_index = data["hit", "delaunay-planar", "hit"].edge_index
        
        # Original hit features
        x_orig = data["hit"].x
        
        # Use message passing for neighbor mean computation
        # This is much more efficient than manual scatter operations
        local_mean = self.mp(x_orig, edge_index)
        
        # Compute a bounded measure of feature deviation for stability
        # This avoids division by very small numbers
        feature_diff = torch.norm(x_orig - local_mean, dim=1, keepdim=True)
        feature_diff = torch.tanh(feature_diff)  # Bounded activation for stability
        
        # Process with reduced dimensionality network
        enhanced = self.feature_net(x_orig)
        
        # Mix original and enhanced features with layernorm for stability
        # This is a residual connection that helps prevent loss of information
        alpha = 0.2  # Small contribution from enhanced features to avoid instability
        data["hit"].x = self.feature_norm(x_orig + alpha * enhanced * feature_diff)