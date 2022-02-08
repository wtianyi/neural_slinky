from typing import Dict, Iterator
import torch

from emlp.groups import SO, S
import emlp.reps as reps
import emlp.nn.pytorch as enn
import numpy as np


so2_rep = reps.V(SO(2))
node_rep = so2_rep ** 0 + so2_rep
triplet_rep = 3*node_rep

class SlinkyForcePredictorEMLP(torch.nn.Module):
    def __init__(
        self,
        layers,
    ) -> None:
        super().__init__()
        self.emlp = enn.EMLP(triplet_rep, triplet_rep, SO(2), num_layers=layers)

    def forward(self, bar_alpha_node_pos) -> torch.Tensor:
        force_pred = self.emlp(bar_alpha_node_pos)

        return force_pred