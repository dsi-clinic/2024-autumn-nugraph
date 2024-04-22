from typing import Any, Callable

from abc import ABC

import torch
import torch.nn as nn
from torch_geometric.nn.aggr import SoftmaxAggregation, LSTMAggregation
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver

import torchmetrics as tm

import matplotlib.pyplot as plt
import seaborn as sn

from ...util import RecallLoss, LogCoshLoss, ObjCondensationLoss

T = torch.Tensor
TD = dict[str, T]

class DecoderBase(nn.Module, ABC):
    '''Base class for all NuGraph decoders'''
    def __init__(self,
                 name: str,
                 planes: list[str],
                 classes: list[str],
                 loss_func: Callable,
                 weight: float,
                 temperature: float = 0.):
        super().__init__()
        self.name = name
        self.planes = planes
        self.classes = classes
        self.loss_func = loss_func
        self.weight = weight
        self.temp = nn.Parameter(torch.tensor(temperature))
        self.confusion = nn.ModuleDict()

    def arrange(self, batch) -> tuple[T, T]:
        raise NotImplementedError

    def metrics(self, x: T, y: T, stage: str) -> dict[str, Any]:
        raise NotImplementedError

    def loss(self,
             batch,
             stage: str,
             confusion: bool = False):
        x, y = self.arrange(batch)
        w = self.weight * (-1 * self.temp).exp()
        loss = w * self.loss_func(x, y) + self.temp
        metrics = {}
        if stage:
            metrics = self.metrics(x, y, stage)
            metrics[f'loss_{self.name}/{stage}'] = loss
            if stage == 'train':
                metrics[f'temperature/{self.name}'] = self.temp
            if confusion:
                for cm in self.confusion.values():
                    cm.update(x, y)
        return loss, metrics

    def finalize(self, batch) -> None:
        return

    def draw_confusion_matrix(self, cm: tm.ConfusionMatrix) -> plt.Figure:
        '''Produce confusion matrix at end of epoch'''
        confusion = cm.compute().cpu()
        fig = plt.figure(figsize=[8,6])
        sn.heatmap(confusion,
                   xticklabels=self.classes,
                   yticklabels=self.classes,
                   vmin=0, vmax=1,
                   annot=True)
        plt.ylim(0, len(self.classes))
        plt.xlabel('Assigned label')
        plt.ylabel('True label')
        return fig

    def on_epoch_end(self,
                     logger: 'pl.loggers.TensorBoardLogger',
                     stage: str,
                     epoch: int) -> None:
        if not logger: return
        for name, cm in self.confusion.items():
            logger.experiment.add_figure(
                f'{name}/{stage}',
                self.draw_confusion_matrix(cm),
                global_step=epoch)
            cm.reset()

class SemanticDecoder(DecoderBase):
    """NuGraph semantic decoder module.

    Convolve down to a single node score per semantic class for each 2D graph,
    node, and remove intermediate node stores from data object.
    """
    def __init__(self,
                 node_features: int,
                 planes: list[str],
                 semantic_classes: list[str]):
        super().__init__('semantic',
                         planes,
                         semantic_classes,
                         RecallLoss(),
                         weight=2.)

        # torchmetrics arguments
        metric_args = {
            'task': 'multiclass',
            'num_classes': len(semantic_classes),
            'ignore_index': -1
        }

        self.recall = tm.Recall(**metric_args)
        self.precision = tm.Precision(**metric_args)
        self.confusion['recall_semantic_matrix'] = tm.ConfusionMatrix(
            normalize='true', **metric_args)
        self.confusion['precision_semantic_matrix'] = tm.ConfusionMatrix(
            normalize='pred', **metric_args)

        self.net = nn.ModuleDict()
        for p in planes:
            self.net[p] = nn.Linear(node_features, len(semantic_classes))

    def forward(self, x: TD) -> dict[str, TD]:
        return {"s": {p: net(x[p]) for p, net in self.net.items()}}

    def arrange(self, batch) -> tuple[T, T]:
        x = torch.cat([batch[p].s for p in self.planes], dim=0)
        y = torch.cat([batch[p].y_semantic for p in self.planes], dim=0)
        return x, y

    def metrics(self, x: T, y: T, stage: str) -> dict[str, Any]:
        return {
            f'recall_semantic/{stage}': self.recall(x, y),
            f'precision_semantic/{stage}': self.precision(x, y)
        }

    def finalize(self, batch) -> None:
        for p in self.planes:
            batch[p].s = batch[p].s.softmax(dim=1)

class FilterDecoder(DecoderBase):
    """NuGraph filter decoder module.

    Convolve down to a single node score, to identify and filter out
    graph nodes that are not part of the primary physics interaction
    """
    def __init__(self,
                 node_features: int,
                 planes: list[str]):
        super().__init__('filter',
                         planes,
                         ('noise', 'signal'),
                         nn.BCELoss(),
                         weight=2.)

        # torchmetrics arguments
        metric_args = {
            'task': 'binary'
        }

        self.recall = tm.Recall(**metric_args)
        self.precision = tm.Precision(**metric_args)
        self.confusion['recall_filter_matrix'] = tm.ConfusionMatrix(
            normalize='true', **metric_args)
        self.confusion['precision_filter_matrix'] = tm.ConfusionMatrix(
            normalize='pred', **metric_args)

        self.net = nn.ModuleDict()
        for p in planes:
            self.net[p] = nn.Sequential(
                nn.Linear(node_features, 1),
                nn.Sigmoid(),
            )

    def forward(self, x: TD) -> dict[str, TD]:
        return {"f": {p: net(x[p]).squeeze(dim=-1) for p, net in self.net.items()}}

    def arrange(self, batch: TD) -> tuple[T, T]:
        x = torch.cat([batch[p].f for p in self.planes], dim=0)
        y = torch.cat([(batch[p].y_semantic!=-1).float() for p in self.planes], dim=0)
        return x, y

    def metrics(self, x: T, y: T, stage: str) -> dict[str, Any]:
        return {
            f'recall_filter/{stage}': self.recall(x, y),
            f'precision_filter/{stage}': self.precision(x, y)
        }

class EventDecoder(DecoderBase):
    '''NuGraph event decoder module.

    Convolve graph node features down to a single classification score
    for the entire event
    '''
    def __init__(self,
                 interaction_features: int,
                 planes: list[str],
                 event_classes: list[str]):
        super().__init__('event',
                         planes,
                         event_classes,
                         RecallLoss(),
                         weight=2.)

        # torchmetrics arguments
        metric_args = {
            'task': 'multiclass',
            'num_classes': len(event_classes)
        }

        self.recall = tm.Recall(**metric_args)
        self.precision = tm.Precision(**metric_args)
        self.confusion['recall_event_matrix'] = tm.ConfusionMatrix(
            normalize='true', **metric_args)
        self.confusion['precision_event_matrix'] = tm.ConfusionMatrix(
            normalize='pred', **metric_args)

        self.net = nn.Linear(in_features=interaction_features,
                             out_features=len(event_classes))

    def forward(self, x: TD) -> dict[str, TD]:
        return {"e": {"evt": self.net(x["evt"])}}

    def arrange(self, batch) -> tuple[T, T]:
        return batch['evt'].e, batch['evt'].y

    def metrics(self, x: T, y: T, stage: str) -> dict[str, Any]:
        return {
            f'recall_event/{stage}': self.recall(x, y),
            f'precision_event/{stage}': self.precision(x, y)
        }

    def finalize(self, batch) -> None:
        batch['evt'].e = batch['evt'].e.softmax(dim=1)

class VertexDecoder(DecoderBase):
    """
    """
    def __init__(self,
                 interaction_features: int,
                 planes: list[str],
                 semantic_classes: list[str]):
        super().__init__('vertex',
                         planes,
                         semantic_classes,
                         LogCoshLoss(),
                         weight=1.,
                         temperature=5.)

        self.net = nn.Linear(interaction_features, 3)

    def forward(self, x: TD) -> dict[str, TD]:
        return {"v": {"evt": self.net(x["evt"])}}

    def arrange(self, batch) -> tuple[T, T]:
        x = batch['evt'].v
        y = batch['evt'].y_vtx
        return x, y

    def metrics(self, x: T, y: T, stage: str) -> dict[str, Any]:
        xyz = (x-y).abs().mean(dim=0)
        return {
            f'vertex-resolution-x/{stage}': xyz[0],
            f'vertex-resolution-y/{stage}': xyz[1],
            f'vertex-resolution-z/{stage}': xyz[2],
            f'vertex-resolution/{stage}': xyz.square().sum().sqrt()
        }

# class InstanceDecoder(DecoderBase):
#     def __init__(self,
#                  node_features: int,
#                  planes: list[str],
#                  classes: list[str]):
#         super().__init__('Instance',
#                          planes,
#                          event_classes,
#                          ObjCondensationLoss(),
#                          'multiclass',
#                          confusion=False)

#         num_features = len(classes) * node_features

#         self.net = nn.ModuleDict()
#         for p in planes:
#             self.net[p] = nn.Sequential(
#                 nn.Linear(num_features, 1),
#                 nn.Sigmoid())

#     def forward(self, x: TD, e: T, batch: TD) -> dict[str, TD]:
#         return {'x_instance': {p: self.net[p](x[p].flatten(start_dim=1)).squeeze(dim=-1) for p in self.net.keys()}}

#     def arrange(self, batch: TD) -> tuple[T, T]:
#         x = torch.cat([batch[p]['x_instance'] for p in self.planes], dim=0)
#         y = torch.cat([batch[p]['y_instance'] for p in self.planes], dim=0)
#         return x, y

#     def metrics(self, x: T, y: T, stage: str) -> dict[str, Any]:
#         metrics = {}
#         predictions = self.predict(x)
#         acc = self.acc_func(predictions, y)
#         metrics[f'{self.name}_accuracy/{stage}'] = accuracy
#         return metrics
