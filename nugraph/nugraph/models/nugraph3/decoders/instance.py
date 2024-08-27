"""NuGraph3 instance decoder"""
from typing import Any
import pathlib
import torch
from torch import nn
from torchmetrics.clustering import AdjustedRandScore
from torch_scatter import scatter_min
from torch_geometric.data import Batch, HeteroData
from torch_geometric.utils import cumsum
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
from ....util import ObjCondensationLoss
from ..types import Data

class InstanceDecoder(nn.Module):
    """
    NuGraph3 instance decoder module

    Convolve object condensation node embedding into a beta value and a set of
    coordinates for each hit.

    Args:
        hit_features: Number of hit node features
        instance_features: Number of instance features
    """
    def __init__(self, hit_features: int, instance_features: int,
                 debug_plots: bool = False):
        super().__init__()

        # loss function
        self.loss = ObjCondensationLoss()

        # Adjusted Rand Index metric
        self.rand = AdjustedRandScore()

        # temperature parameter
        self.temp = nn.Parameter(torch.tensor(0.))

        # network
        self.beta_net = nn.Linear(hit_features, 1)
        self.coord_net = nn.Linear(hit_features, instance_features)

        self.debug_plots = debug_plots
        self.dfs = []

    def forward(self, data: Data, stage: str = None) -> dict[str, Any]:
        """
        NuGraph3 instance decoder forward pass

        Args:
            data: Graph data object
            stage: Stage name (train/val/test)
        """

        # run network and add output to graph object
        data["hit"].of = self.beta_net(data["hit"].x).squeeze(dim=-1).sigmoid()
        data["hit"].ox = self.coord_net(data["hit"].x)
        if isinstance(data, Batch):
            data._slice_dict["hit"]["of"] = data["hit"].ptr
            data._slice_dict["hit"]["ox"] = data["hit"].ptr
            data._inc_dict["hit"]["of"] = data._inc_dict["hit"]["x"]
            data._inc_dict["hit"]["ox"] = data._inc_dict["hit"]["x"]

        # materialize instances
        materialize = (data["hit"].of > 0.1).sum() < 2000
        if materialize:
            # form instances across batch
            device = data["hit"].x.device
            imask = data["hit"].of > 0.1
            data["particle"].x = torch.empty(imask.sum(), 0, device=device)
            data["particle"].ox = data["hit"].ox[imask]
            if isinstance(data, Batch):
                repeats = torch.empty(data.num_graphs, dtype=torch.long, device=device)
                data["particle"].batch = torch.empty(data["particle"].num_nodes,
                                                     dtype=torch.long, device=device)
                for i in range(data.num_graphs):
                    lo, hi = data._slice_dict["hit"]["x"][i:i+2]
                    repeats[i] = imask[lo:hi].sum()
                    data["particle"].batch[lo:hi] = i
                data["particle"].ptr = cumsum(repeats)
                data._slice_dict["particle"] = {
                    "x": data["particle"].ptr,
                    "ox": data["particle"].ptr,
                }
                data._inc_dict["particle"] = {
                    "x": data._inc_dict["hit"]["x"],
                    "ox": data._inc_dict["hit"]["x"],
                }
                data = Batch.from_data_list([self.materialize(b) for b in data.to_data_list()])
            else:
                self.materialize(data)

            # collapse instance edges into labels
            e = data["hit", "cluster", "particle"]
            _, instances = scatter_min(e.distance, e.edge_index[0], dim_size=data["hit"].num_nodes)
            mask = instances < e.num_edges
            instances[~mask] = -1
            instances[mask] = e.edge_index[1, instances[mask]]
            data["hit"].i = instances
            if isinstance(data, Batch):
                data._slice_dict["hit"]["i"] = data["hit"].ptr
                data._inc_dict["hit"]["i"] = data._inc_dict["hit"]["x"]

        # calculate loss
        y = torch.full_like(data["hit"].y_semantic, -1)
        i, j = data["hit", "cluster-truth", "particle-truth"].edge_index
        y[i] = j
        data["hit"].y_instance = y
        loss = (-1 * self.temp).exp() * self.loss(data, y) + self.temp

        # calculate metrics
        metrics = {}
        if stage:
            metrics[f"instance/loss-{stage}"] = loss
            if materialize:
                x = data["hit"].i
                metrics[f"instance/adjusted-rand-{stage}"] = self.rand(x, y)
        if stage == "train":
            metrics["temperature/instance"] = self.temp

        if self.debug_plots and stage == "val" and isinstance(data, Batch):
            for d in data.to_data_list():
                if len(self.dfs) >= 100:
                    break
                self.dfs.append(self.draw_event_display(d))

        return loss, metrics

    def materialize(self, data: Data) -> None:
        """
        Materialize object condensation embedding into instances

        Args:
            data: Heterodata graph object
        """
        e = data["hit", "cluster", "particle"]
        x_hit = data["hit"].ox
        x_part = data["particle"].ox
        dist = (x_hit[:, None, :] - x_part[None, :, :]).square().sum(dim=2)
        e.edge_index = (dist < 1).nonzero().transpose(0, 1).detach()
        e.distance = dist[e.edge_index[0], e.edge_index[1]].detach()
        return data

    def draw_event_display(self, data: HeteroData) -> pd.DataFrame:
        """
        Draw event displays for NuGraph3 object condensation embedding

        Args:
            data: Graph data object
        """
        coords = data["hit"].ox.cpu()
        pca = PCA(n_components=2)
        c1, c2 = pca.fit_transform(coords).transpose()
        beta = data["hit"].of.cpu()
        logbeta = beta.log10()
        xy = data["hit"].pos.cpu()
        i = data["hit"].y_instance.cpu()
        plane = data["hit"].plane.cpu()
        plane = data["hit"].plane.cpu()
        return pd.DataFrame(dict(c1=c1, c2=c2, beta=beta, logbeta=logbeta,
                                 plane=plane, x=xy[:,0], y=xy[:,1],
                                 instance=pd.Series(i).astype(str)))

    def on_epoch_end(self,
                     logger: TensorBoardLogger,
                     stage: str,
                     epoch: int) -> None:
        """
        NuGraph3 instance decoder end-of-epoch callback function

        Args:
            logger: Tensorboard logger object
            stage: Training stage
            epoch: Training epoch index
        """
        if not self.debug_plots or not logger:
            return
        path = pathlib.Path(logger.log_dir) / "objcon-plots"
        path.mkdir(exist_ok=True)
        if stage == "val":
            for i, df in enumerate(self.dfs):

                # object condensation true instance plot
                fig = px.scatter(df, x="x", y="y", facet_col="plane",
                                 color="instance", title=f"epoch {epoch}")
                fig.update_xaxes(matches=None)
                for a in fig.layout.annotations:
                    a.text = a.text.replace("plane=", "")
                fig.write_image(file=path/f"evt{i+1}-truth.png")

                # object condensation beta plot
                fig = px.scatter(df, x="x", y="y", facet_col="plane",
                                 color="logbeta", title=f"epoch {epoch}")
                fig.update_xaxes(matches=None)
                for a in fig.layout.annotations:
                    a.text = a.text.replace("plane=", "")
                fig.write_image(file=path/f"evt{i+1}-beta.png")

                # object condensation coordinate plot
                fig = px.scatter(df, x="c1", y="c2",
                                 color="instance", title=f"epoch {epoch}")
                fig.write_image(file=path/f"evt{i+1}-coords.png")

        self.dfs = []
