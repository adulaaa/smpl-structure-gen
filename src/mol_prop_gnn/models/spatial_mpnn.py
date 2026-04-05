"""Spatial MPNN (GNoME-inspired) for 3D molecular property prediction.

Uses E(3) invariant continuous-filter convolutions and RBF-expanded 
Euclidean distances.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import scatter


class GaussianSmearing(nn.Module):
    """Expands continuous distances into a discrete radial basis function set.
    
    Parameters
    ----------
    start : float
        Minimum distance for the RBF range.
    stop : float
        Maximum distance for the RBF range.
    num_gaussians : int
        Number of Gaussian kernels to use (RBF dimension).
    """

    def __init__(self, start: float = 0.0, stop: float = 5.0, num_gaussians: int = 50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        # dist is [E, 1]
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class CFConv(nn.Module):
    """Continuous Filter Convolution (as seen in SchNet/GNoME).
    
    Message passing is a continuous function of distance.
    M_ij = h_j * FilterNet(RBF(d_ij))
    """

    def __init__(self, in_channels: int, out_channels: int, num_rbf: int):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(num_rbf, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_rbf: torch.Tensor) -> torch.Tensor:
        # 1. Compute filter weights using the filter network
        weight = self.filter_net(edge_rbf) # [E, out_channels]
        
        # 2. Map nodes to message space
        x = self.lin(x) # [N, out_channels]
        
        # 3. Propagate messages
        row, col = edge_index
        msg = x[col] * weight # [E, out_channels]
        
        # 4. Aggregate via scatter sum (using PyG 2.4+ scatter)
        out = scatter(msg, row, dim=0, dim_size=x.size(0), reduce='sum')
        
        return out


class SpatialMPNN(nn.Module):
    """GNoME-inspired Spatial Message Passing Neural Network.
    
    Parameters
    ----------
    hidden_dim : int
        Dimensionality of node embeddings.
    num_layers : int
        Number of spatial message passing layers.
    num_rbf : int
        Number of Gaussian kernels for distance expansion.
    cutoff : float
        Spatial cutoff distance for radius graph.
    """

    def __init__(
        self, 
        hidden_dim: int = 128, 
        num_layers: int = 4, 
        num_rbf: int = 50, 
        cutoff: float = 4.0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Node Embedding strictly based on Atomic Number
        # Max atomic number is 118, so 128 covers all possible elements
        self.embedding = nn.Embedding(128, hidden_dim)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_rbf)
        
        self.layers = nn.ModuleList([
            CFConv(hidden_dim, hidden_dim, num_rbf)
            for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode the 3D graph into local atom representations."""
        # x is [N] atomic numbers
        # edge_attr is [E, 1] raw Euclidean distances
        
        # Expand distances to RBF features
        edge_rbf = self.distance_expansion(edge_attr)
        
        # Initial atom embeddings
        h = self.embedding(x)
        
        # Message passing layers
        for conv, norm in zip(self.layers, self.norms):
            # Residual-style spatial communication
            msg_out = conv(h, edge_index, edge_rbf)
            h = h + msg_out
            h = norm(h)
            
        return h

    @property
    def out_channels(self) -> int:
        """The output dimension of the encoder's atom representations."""
        return self.hidden_dim

    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_attr: torch.Tensor, 
        batch: torch.Tensor | None = None, 
        **kwargs
    ) -> torch.Tensor:
        """Graph-level encoding with spatial pooling."""
        h = self.encode(x, edge_index, edge_attr)
        # Pool to graph-level embedding [B, hidden_dim]
        h_graph = global_mean_pool(h, batch)
        return h_graph
