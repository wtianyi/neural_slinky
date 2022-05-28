from typing import Dict, Iterator, Tuple

import emlp.nn.pytorch as enn
import emlp.reps as reps
import numpy as np
import torch
from emlp.groups import SO, S

so2_rep = reps.V(SO(2))
node_rep = so2_rep**0 + so2_rep
triplet_rep = 3 * node_rep


class SlinkyForcePredictorEMLP(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.emlp = enn.EMLP(triplet_rep, triplet_rep, SO(2), num_layers=num_layers)

    def forward(self, bar_alpha_node_pos) -> torch.Tensor:
        force_pred = self.emlp(bar_alpha_node_pos)

        return force_pred


class SlinkyForceODEFunc(torch.nn.Module):
    def __init__(self, num_layers: int, boundaries: Tuple[int, int]) -> None:
        super().__init__()
        self.slinky_force_predictor = SlinkyForcePredictorEMLP(num_layers)
        self.boundaries = boundaries

    @staticmethod
    def _make_triplet_alpha_node_pos(x: torch.Tensor) -> torch.Tensor:
        """Convert slinky coords to overlapping triplet coords, differentiably

        Args:
            x (torch.Tensor): slinky coords (n_batch x n_cycles x 3)
        Returns:
            torch.Tensor: Shape of (n_batch x (n_cycles-2) x 9). The last axis is
            (alpha_1, alpha_2, alpha_3, x_1, y_1, x_2, y_2, x_3, y_3)
        """
        x_prev = x[:, :-2, :]
        x_mid = x[:, 1:-1, :]
        x_next = x[:, 2:, :]
        offset = torch.atan2(
            x_mid[..., 1] - x_prev[..., 1], x_mid[..., 0] - x_prev[..., 0]
        )
        return torch.stack(
            (
                x_prev[..., 2] - offset,
                x_mid[..., 2] - offset,
                x_next[..., 2] - offset,
                x_prev[..., 0],
                x_prev[..., 1],
                x_mid[..., 0],
                x_mid[..., 1],
                x_next[..., 0],
                x_next[..., 1],
            ),
            dim=-1,
        )

    def forward(self, y):
        # if self.boundaries != (1,1): # not both ends fixed
        #     raise ValueError(f"Boundary condition {self.boundaries} is not implemented for")
        coords = y[..., :3]
        triplet_alpha_node_pos = self._make_triplet_alpha_node_pos(coords)
        forces = self.slinky_force_predictor(triplet_alpha_node_pos)
        result = torch.zeros_like(coords)  # (n_batch x n_cycles x 3)
        result[..., :-2, 0] += forces[..., 3]
        result[..., :-2, 1] += forces[..., 4]
        result[..., :-2, 2] += forces[..., 0]
        result[..., 1:-1, 0] += forces[..., 5]
        result[..., 1:-1, 1] += forces[..., 6]
        result[..., 1:-1, 2] += forces[..., 1]
        result[..., 2:, 0] += forces[..., 7]
        result[..., 2:, 1] += forces[..., 8]
        result[..., 2:, 2] += forces[..., 2]
        if self.boundaries[0] == 1:
            result[..., 0, :] = 0
        if self.boundaries[1] == 1:
            result[..., -1, :] = 0
        # if torch.any(torch.isnan(result)):
        #     print(f"{y=}")
        print(f"{torch.abs(result).max()=}")
        return result
