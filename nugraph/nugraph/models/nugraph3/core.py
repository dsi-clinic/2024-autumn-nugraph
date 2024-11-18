"""NuGraph core message-passing engine"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import MessagePassing, HeteroConv
from .types import T, TD, Data

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
            nn.Linear(source_features + target_features, 1),
            nn.Sigmoid())

        self.net = nn.Sequential(
            nn.Linear(source_features + target_features, out_features),
            nn.Mish(),
            nn.Linear(out_features, out_features),
            nn.Mish())

    def forward(self, x: T, edge_index: T) -> T:
        """
        NuGraphBlock forward pass
        
        Args:
            x: Node feature tensor
            edge_index: Edge index tensor
        """
        a, b = x
        if a is None:
            x = (b, b)
        if b is None:
            x = (a, a)
        print('edge', edge_index)
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
        print('message')
        print((x_i.shape, x_j.shape))
        return self.edge_net(torch.cat((x_i, x_j), dim=1).detach()) * x_j  

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

# class NuGraphCore(nn.Module):
#     """
#     NuGraph core message-passing engine
    
#     This is the core NuGraph message-passing loop

#     Args:
#         hit_features: Number of features in planar embedding
#         nexus_features: Number of features in nexus embedding
#         interaction_features: Number of features in interaction embedding
#         ophit_features: Number of features in optical hit embedding
#         pmt_features: Number of features in PMT (flashsumpe) embedding
#         flash_features: Number of features in optical flash embedding
#         use_checkpointing: Whether to use checkpointing
#     """
#     def __init__(self,
#                  hit_features: int,
#                  nexus_features: int,
#                  interaction_features: int,
#                  ophit_features: int,
#                  pmt_features: int,
#                  flash_features: int,
#                  use_checkpointing: bool = True):
#         super().__init__()

#         self.use_checkpointing = use_checkpointing

#         # internal planar message-passing
#         self.plane_net = NuGraphBlock(hit_features, hit_features,
#                                       hit_features)

#         # message-passing from planar nodes to nexus nodes
#         self.plane_to_nexus = NuGraphBlock(hit_features, nexus_features,
#                                            nexus_features)

#         # message-passing from nexus nodes to interaction nodes
#         self.nexus_to_interaction = NuGraphBlock(nexus_features,
#                                                  interaction_features,
#                                                  interaction_features)


#         # message-passing from interaction nodes to nexus nodes
#         self.interaction_to_nexus = NuGraphBlock(interaction_features,
#                                                  nexus_features,
#                                                  nexus_features)

#         # message-passing from nexus nodes to planar nodes
#         self.nexus_to_plane = NuGraphBlock(nexus_features, hit_features,
#                                            hit_features)
        
#         # hierarchical message-passing for optical system
#         self.ophit_to_pmt = NuGraphBlock(ophit_features, pmt_features, pmt_features)
#         self.pmt_to_flash = NuGraphBlock(pmt_features, flash_features, flash_features)
#         self.flash_to_interaction = NuGraphBlock(flash_features,
#                                                  interaction_features,
#                                                  interaction_features)
#         self.interaction_to_flash = NuGraphBlock(interaction_features,
#                                                  flash_features, flash_features)
#         self.flash_to_pmt = NuGraphBlock(flash_features, pmt_features, pmt_features)
#         self.pmt_to_ophit = NuGraphBlock(pmt_features, ophit_features, ophit_features)

#     def checkpoint(self, net: nn.Module, *args) -> TD:
#         """
#         Checkpoint module, if enabled.
        
#         Args:
#             net: Network module
#             args: Arguments to network module
#         """
#         if self.use_checkpointing and self.training:
#             return checkpoint(net, *args, use_reentrant=False)
#         else:
#             return net(*args)

#     def forward(self, data: Data) -> None:
#         """
#         NuGraphCore forward pass
        
#         Args:
#             data: Graph data object
#         """

#         # message-passing in hits
#         data["hit"].x = self.checkpoint(
#             self.plane_net, data["hit"].x,
#             data["hit", "delaunay-planar", "hit"].edge_index)

#         # message-passing from hits to nexus
#         data["sp"].x = self.checkpoint(
#             self.plane_to_nexus, (data["hit"].x, data["sp"].x),
#             data["hit", "nexus", "sp"].edge_index)

#         # message-passing from nexus to interaction
#         data["evt"].x = self.checkpoint(
#             self.nexus_to_interaction, (data["sp"].x, data["evt"].x),
#             data["sp", "in", "evt"].edge_index)

#         # message-passing from ophit to pmt
#         data["opflashsumpe"].x = self.checkpoint(
#             self.ophit_to_pmt, (data["ophits"].x, data["opflashsumpe"].x),
#             data["ophits", "sumpe", "opflashsumpe"].edge_index)

#         # message-passing from pmt to flash
#         data["opflash"].x = self.checkpoint(
#             self.pmt_to_flash, (data["opflashsumpe"].x, data["opflash"].x),
#             data["opflashsumpe", "flash", "opflash"].edge_index)

#         # message-passing from flash to interaction
#         data["evt"].x = self.checkpoint(
#             self.flash_to_interaction, (data["opflash"].x, data["evt"].x),
#             data["opflash", "in", "evt"].edge_index)

#         # message-passing from interaction to flash
#         data["opflash"].x = self.checkpoint(
#             self.interaction_to_flash, (data["evt"].x, data["opflash"].x),
#             data["opflash", "in", "evt"].edge_index[(1,0), :])

#         # message-passing from flash to pmt
#         data["opflashsumpe"].x = self.checkpoint(
#             self.flash_to_pmt, (data["opflash"].x, data["opflashsumpe"].x),
#             data["opflashsumpe", "flash", "opflash"].edge_index[(1,0), :])

#         # message-passing from pmt to ophit
#         data["ophits"].x = self.checkpoint(
#             self.pmt_to_ophit, (data["opflashsumpe"].x, data["ophits"].x),
#             data["ophits", "sumpe", "opflashsumpe"].edge_index[(1,0), :])

#         # message-passing from interaction to nexus
#         data["sp"].x = self.checkpoint(
#             self.interaction_to_nexus, (data["evt"].x, data["sp"].x),
#             data["sp", "in", "evt"].edge_index[(1,0), :])

#         # message-passing from nexus to hits
#         data["hit"].x = self.checkpoint(
#             self.nexus_to_plane, (data["sp"].x, data["hit"].x),
#             data["hit", "nexus", "sp"].edge_index[(1,0), :])


class NuGraphCore(nn.Module):
    """
    NuGraph core message-passing engine
    
    This is the core NuGraph message-passing loop

    Args:
        planar_features: Number of features in planar embedding
        nexus_features: Number of features in nexus embedding
        interaction_features: Number of features in interaction embedding
        ophit_features: Number of features in optical hit embedding
        pmt_features: Number of features in PMT (flashsumpe) embedding
        flash_features: Number of features in optical flash embedding
        planes: List of detector planes
    """
    def __init__(self,
                 hit_features: int,
                 nexus_features: int,
                 interaction_features: int,
                 ophit_features: int,
                 pmt_features: int,
                 flash_features: int,
               #  planes: list[str]):
                 planes: int):

        super().__init__()

     #   full_nexus_features = len(planes) * nexus_features
        full_nexus_features = planes * nexus_features

        # internal planar message-passing
        self.plane_net = HeteroConv({
            ("hit", "delaunay-planar", "hit"): NuGraphBlock(hit_features,
                                          hit_features,
                                          hit_features)})
       #     for p in planes})

        # message-passing from planar nodes to nexus nodes
        self.plane_to_nexus = HeteroConv({
            ("hit", "nexus", "sp"): NuGraphBlock(hit_features,
                                             nexus_features,
                                             nexus_features)})
        #    for p in planes}, aggr="cat")
        
        # message-passing from optical hit nodes to PMT nodes
        self.hit_to_pmt = HeteroConv({
           ('ophits', 'sumpe', 'opflashsumpe'): NuGraphBlock(ophit_features,
                                                 pmt_features,
                                                 pmt_features)})
        
        # message-passing from PMT nodes to flash nodes
        self.pmt_to_flash = HeteroConv({
           ('opflashsumpe', 'flash', 'opflash'): NuGraphBlock(pmt_features,
                                                flash_features,
                                                flash_features)})

        # message-passing from nexus and flash nodes to interaction nodes
        self.nexus_flash_to_interaction = HeteroConv({
            ("sp", "in", "evt"): NuGraphBlock(full_nexus_features,
                                              interaction_features,
                                              interaction_features), 
            ('opflash', 'in', 'evt'): NuGraphBlock(flash_features, 
                                                interaction_features, 
                                                interaction_features)})

        # message-passing from interaction nodes to nexus nodes
        self.interaction_to_nexus = HeteroConv({
            ("evt", "owns", "sp"): NuGraphBlock(interaction_features,
                                                full_nexus_features,
                                                nexus_features)})

        # message-passing from nexus nodes to planar nodes
        self.nexus_to_plane = HeteroConv({
            ("sp", "nexus", "hit"): NuGraphBlock(nexus_features,
                                             hit_features,
                                             hit_features)})
         #   for p in planes})


        # message-passing from interaction nodes to flash node
        self.interaction_to_flash = HeteroConv({
            ("evt", "in", "opflash"): NuGraphBlock(interaction_features,
                                                   flash_features,
                                                   flash_features)})
        
        # message-passing from flash node to PMT nodes
        self.flash_to_pmt = HeteroConv({
            ("opflash", "flash", "opflashsumpe"): NuGraphBlock(flash_features,
                                                pmt_features,
                                                pmt_features)})

        # message-passing from PMT nodes to optical hit nodes
        self.pmt_to_ophit = HeteroConv({
            ("opflashsumpe", "sumpe", "ophits"): NuGraphBlock(pmt_features,
                                                ophit_features,
                                                ophit_features)})
         
    def forward(self, p: TD, n: TD,  oph: TD, pmt: TD,
                opf: TD, i: TD, edges: TD) -> tuple[TD, TD, TD, TD, TD, TD]:
        """
        NuGraphCore forward pass
        
        Args:
            p: Planar embedding tensor dictionary
            n: Nexus embedding tensor dictionary
            i: Interaction embedding tensor dictionary
            oph: Optical hit embedding tensor dictionary
            pmt: PMT (flashsumpe) embedding tensor dictionary
            opf: Optical flash embedding tensor dictionary
            edges: Edge index tensor dictionary
        """ 
        # one upward pass
        #print('p before plane_net', p)
        #p = self.plane_net(p, edges)
        #print('p after plane_net', p)
        if p is not None and n is not None and edges is not None:
            print('edges', edges)
            print('plane to nexus')
            n = self.plane_to_nexus((p|n), edges)
        if oph is not None and pmt is not None and edges is not None:
            print('hit to pmt')
            pmt = self.hit_to_pmt((oph|pmt), edges)
        if oph is not None and pmt is not None and edges is not None:
            print('pmt to flash')
            opf = self.pmt_to_flash((pmt|opf), edges)
        if n is not None and opf is not None and i is not None and edges is not None:
            print('nexus flash to interaction')
            i = self.nexus_flash_to_interaction((n|opf|i), edges)

        # one downward pass
        if i is not None and n is not None and edges is not None:
            print('interaction to nexus')
            n = self.interaction_to_nexus((i|n), edges)
        if n is not None and p is not None and edges is not None:
            print('nexus to plane')
            p = self.nexus_to_plane((n|p), edges)
        if opf is not None and i is not None and edges is not None:
            print('interaction to flash')
            opf = self.interaction_to_flash((i|opf), edges)
        if opf is not None and pmt is not None and edges is not None:
            print('flash to pmt')
            pmt = self.flash_to_pmt((opf|pmt), edges)
        if pmt is not None and oph is not None and edges is not None:
            print('pmt to ophit')
            oph = self.pmt_to_ophit((pmt|oph), edges)

        return p, n, oph, pmt, opf, i
