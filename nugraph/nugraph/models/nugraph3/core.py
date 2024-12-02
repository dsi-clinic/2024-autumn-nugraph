"""NuGraph core message-passing engine"""
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import MessagePassing
from .types import T, TD, Data

class MessageGate(nn.Module):
    """
    Message gating mechanism for controlling information flow in message passing.

    This module generates a gating value based on both the new and old message features,
    and uses it to control how much information from each message is retained. The gate
    is computed using a linear projection followed by sigmoid activation.

    Args:
        dim: Number of feature dimensions in the messages
    """
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
    def forward(self, new_msg, old_msg):
        """
        MessageGate forward pass

        Args:
            new_msg: New message features tensor
            old_msg: Previous message features tensor
        """
        gate = self.gate(torch.cat([new_msg, old_msg], dim=-1))
        return gate * new_msg + (1 - gate) * old_msg
    
class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for inter-node type feature interaction.
    
    This module enables direct attention between different types of nodes
    by projecting their features to a common space, computing attention
    weights, and using these weights to update the query features. The
    module includes projection layers, multi-head attention computation,
    and layer normalization.

    Args:
        query_dim: Number of input features in query tensor
        key_dim: Number of input features in key/value tensors
    """
    def __init__(self, query_dim, key_dim):
        super().__init__()
        self.hidden_dim = 256  # Common attention dimension
        self.num_heads = 4
        self.head_dim = self.hidden_dim // self.num_heads
        
        self.query_proj = nn.Linear(query_dim, self.hidden_dim)
        self.key_proj = nn.Linear(key_dim, self.hidden_dim)
        self.value_proj = nn.Linear(key_dim, self.hidden_dim)
        
        # Single linear layer for attention
        self.attention = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, query_dim)
        self.norm = nn.LayerNorm(query_dim)
        
    def forward(self, x, context):
        """
        CrossAttention forward pass
        
        This function computes cross-attention between query and context features.
        Features are first projected to a common dimension, then attention scores
        are computed and used to weight the value features. The result is
        projected back to the original query dimension and combined with the
        input through a residual connection.

        Args:
            x: Query features tensor
            context: Context features tensor for keys and values
        """
        # Project inputs
        q = self.query_proj(x)
        k = self.key_proj(context)
        v = self.value_proj(context)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, v)
        out = self.attention(out)
        out = self.output_proj(out)
        
        return self.norm(out + x)

class NuGraphBlock(MessagePassing):
    """
    Standard NuGraph message-passing block
    
    This block generates attention weights for each graph edge based on both
    the source and target node features, and then applies those weights to
    the source node features in order to form messages. These messages are
    then aggregated into the target nodes using softmax aggregation, and
    then fed into a two-layer MLP to generate updated target node features.

    Args:
        source_features: Number of source node input features
        target_features: Number of target node input features
        out_features: Number of target node output features
    """
    def __init__(self, source_features: int, target_features: int,
                 out_features: int):
        super().__init__(aggr="softmax")

        self.edge_net = nn.Sequential(
            nn.Linear(source_features+target_features, 1),
            nn.Sigmoid())

        self.net = nn.Sequential(
            nn.Linear(source_features+target_features, out_features),
            nn.Mish(),
            nn.Linear(out_features, out_features),
            nn.Mish())
        
        self.message_gate = MessageGate(source_features)

    def forward(self, x: T, edge_index: T) -> T:
        """
        NuGraphBlock forward pass
        
        Args:
            x: Node feature tensor
            edge_index: Edge index tensor
        """
        return self.propagate(edge_index, x=x)

    def message(self, x_i: T, x_j: T) -> T:
        """
        NuGraphBlock message function

        This function constructs messages on graph edges. Features from the
        source and target nodes are concatenated and fed into a linear layer
        to construct attention weights. Messages are then formed on edges by
        weighting the source node features by these attention weights.
        
        Args:
            x_i: Edge features from target nodes
            x_j: Edge features from source nodes
        """
        edge_weight = self.edge_net(torch.cat((x_i, x_j), dim=1).detach())
        new_msg = edge_weight * x_j
        return self.message_gate(new_msg, x_j)

    def update(self, aggr_out: T, x: T) -> T:
        """
        NuGraphBlock update function

        This function takes the output node features and combines them with
        the input features

        Args:
            aggr_out: Tensor of aggregated node features
            x: Target node features
        """
        if isinstance(x, tuple):
            _, x = x
        return self.net(torch.cat((aggr_out, x), dim=1))

class NuGraphCore(nn.Module):
    """
    NuGraph core message-passing engine
    
    This is the core NuGraph message-passing loop

    Args:
        hit_features: Number of features in planar embedding
        nexus_features: Number of features in nexus embedding
        interaction_features: Number of features in interaction embedding
        ophit_features: Number of features in optical hit embedding
        pmt_features: Number of features in PMT (flashsumpe) embedding
        flash_features: Number of features in optical flash embedding
        use_checkpointing: Whether to use checkpointing
    """
    def __init__(self,
                 hit_features: int,
                 nexus_features: int,
                 interaction_features: int,
                 ophit_features: int,
                 pmt_features: int,
                 flash_features: int,
                 use_checkpointing: bool = True):
        super().__init__()

        self.use_checkpointing = use_checkpointing

        # Modified cross-attention modules to handle different dimensions
        self.hit_flash_attention = CrossAttention(hit_features, flash_features)
        self.nexus_pmt_attention = CrossAttention(nexus_features, pmt_features)
        self.interaction_ophit_attention = CrossAttention(interaction_features, ophit_features)

        # internal planar message-passing
        self.plane_net = NuGraphBlock(hit_features, hit_features,
                                      hit_features)

        # message-passing from planar nodes to nexus nodes
        self.plane_to_nexus = NuGraphBlock(hit_features, nexus_features,
                                           nexus_features)

        # message-passing from nexus nodes to interaction nodes
        self.nexus_to_interaction = NuGraphBlock(nexus_features,
                                                 interaction_features,
                                                 interaction_features)


        # message-passing from interaction nodes to nexus nodes
        self.interaction_to_nexus = NuGraphBlock(interaction_features,
                                                 nexus_features,
                                                 nexus_features)

        # message-passing from nexus nodes to planar nodes
        self.nexus_to_plane = NuGraphBlock(nexus_features, hit_features,
                                           hit_features)
        
        # hierarchical message-passing for optical system
        self.ophit_to_pmt = NuGraphBlock(ophit_features, pmt_features, pmt_features)
        self.pmt_to_flash = NuGraphBlock(pmt_features, flash_features, flash_features)
        self.flash_to_interaction = NuGraphBlock(flash_features,
                                                 interaction_features,
                                                 interaction_features)
        self.interaction_to_flash = NuGraphBlock(interaction_features,
                                                 flash_features, flash_features)
        self.flash_to_pmt = NuGraphBlock(flash_features, pmt_features, pmt_features)
        self.pmt_to_ophit = NuGraphBlock(pmt_features, ophit_features, ophit_features)

        # Layer normalization for skip connections
        self.hit_norm = nn.LayerNorm(hit_features)
        self.nexus_norm = nn.LayerNorm(nexus_features)
        self.interaction_norm = nn.LayerNorm(interaction_features)
        self.ophit_norm = nn.LayerNorm(ophit_features)
        self.pmt_norm = nn.LayerNorm(pmt_features)
        self.flash_norm = nn.LayerNorm(flash_features)

    def checkpoint(self, net: nn.Module, *args) -> TD:
        """
        Checkpoint module, if enabled.
        
        Args:
            net: Network module
            args: Arguments to network module
        """
        if self.use_checkpointing and self.training:
            return checkpoint(net, *args, use_reentrant=False)
        else:
            return net(*args)

    def forward(self, data: Data) -> None:
        """
        NuGraphCore forward pass
        
        Args:
            data: Graph data object
        """

        # Store initial states for skip connections
        hit_initial = data["hit"].x.clone()
        sp_initial = data["sp"].x.clone()
        evt_initial = data["evt"].x.clone()
        ophits_initial = data["ophits"].x.clone()
        opflashsumpe_initial = data["opflashsumpe"].x.clone()
        opflash_initial = data["opflash"].x.clone()

        # message-passing in hits with skip connection
        hit_out = self.checkpoint(
            self.plane_net, data["hit"].x,
            data["hit", "delaunay-planar", "hit"].edge_index)
        data["hit"].x = self.hit_norm(hit_out + hit_initial)

        # Add cross-attention between hit and flash features
        if data["opflash"].x.shape[0] > 0:
            data["hit"].x = self.hit_flash_attention(data["hit"].x, data["opflash"].x)
            
        # message-passing from hits to nexus with skip connection
        nexus_out = self.checkpoint(
            self.plane_to_nexus, (data["hit"].x, data["sp"].x),
            data["hit", "nexus", "sp"].edge_index)
        data["sp"].x = self.nexus_norm(nexus_out + sp_initial)

        # Add cross-attention between nexus and PMT features
        if data["opflashsumpe"].x.shape[0] > 0:
            data["sp"].x = self.nexus_pmt_attention(data["sp"].x, data["opflashsumpe"].x)

        # message-passing from nexus to interaction with skip connection
        interaction_out = self.checkpoint(
            self.nexus_to_interaction, (data["sp"].x, data["evt"].x),
            data["sp", "in", "evt"].edge_index)
        data["evt"].x = self.interaction_norm(interaction_out + evt_initial)

        # message-passing from ophit to pmt with skip connection
        pmt_out = self.checkpoint(
            self.ophit_to_pmt, (data["ophits"].x, data["opflashsumpe"].x),
            data["ophits", "sumpe", "opflashsumpe"].edge_index)
        data["opflashsumpe"].x = self.pmt_norm(pmt_out + opflashsumpe_initial)

        # message-passing from pmt to flash with skip connection
        flash_out = self.checkpoint(
            self.pmt_to_flash, (data["opflashsumpe"].x, data["opflash"].x),
            data["opflashsumpe", "flash", "opflash"].edge_index)
        data["opflash"].x = self.flash_norm(flash_out + opflash_initial)

        # Add cross-attention between interaction and optical hit features
        if data["ophits"].x.shape[0] > 0:
            data["evt"].x = self.interaction_ophit_attention(data["evt"].x, data["ophits"].x)

        # message-passing from flash to interaction with skip connection
        interaction_out = self.checkpoint(
            self.flash_to_interaction, (data["opflash"].x, data["evt"].x),
            data["opflash", "in", "evt"].edge_index)
        data["evt"].x = self.interaction_norm(interaction_out + evt_initial)

        # message-passing from interaction to flash with skip connection
        flash_out = self.checkpoint(
            self.interaction_to_flash, (data["evt"].x, data["opflash"].x),
            data["opflash", "in", "evt"].edge_index[(1,0), :])
        data["opflash"].x = self.flash_norm(flash_out + opflash_initial)

        # message-passing from flash to pmt with skip connection
        pmt_out = self.checkpoint(
            self.flash_to_pmt, (data["opflash"].x, data["opflashsumpe"].x),
            data["opflashsumpe", "flash", "opflash"].edge_index[(1,0), :])
        data["opflashsumpe"].x = self.pmt_norm(pmt_out + opflashsumpe_initial)

        # message-passing from pmt to ophit with skip connection
        ophit_out = self.checkpoint(
            self.pmt_to_ophit, (data["opflashsumpe"].x, data["ophits"].x),
            data["ophits", "sumpe", "opflashsumpe"].edge_index[(1,0), :])
        data["ophits"].x = self.ophit_norm(ophit_out + ophits_initial)

        # message-passing from interaction to nexus with skip connection
        nexus_out = self.checkpoint(
            self.interaction_to_nexus, (data["evt"].x, data["sp"].x),
            data["sp", "in", "evt"].edge_index[(1,0), :])
        data["sp"].x = self.nexus_norm(nexus_out + sp_initial)

        # message-passing from nexus to hits with skip connection
        hit_out = self.checkpoint(
            self.nexus_to_plane, (data["sp"].x, data["hit"].x),
            data["hit", "nexus", "sp"].edge_index[(1,0), :])
        data["hit"].x = self.hit_norm(hit_out + hit_initial)
