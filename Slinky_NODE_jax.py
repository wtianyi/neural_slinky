import os
from random import choices
from typing import Callable, List, Optional, Tuple

import configargparse
import numpy as np
import pytorch_lightning as pl

# import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

# from torchdiffeq import odeint_adjoint as odeint

# from torchdiffeq import odeint as odeint


import wandb
from neural_slinky import trajectory_dataset, coords_transform
from neural_slinky.utils import (
    animate_2d_slinky_trajectories,
    get_model_device,
)
from priority_memory import batch
from slinky import func as node_f

# from neural_slinky import emlp_models

import pandas as pd

import equinox as eqx
from jaxtyping import Array, Float, PyTree, jaxtyped
from typeguard import typechecked
import jax
import jax.tree_util as jtu
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import chex
import diffrax
import optax

import plotly.graph_objects as go
import plotly.express as px
import io

import itertools
import tqdm

import warnings

# Suppress Pytorch Lightning warnings about dataloader workers
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*`training_step` returned `None`.*")
warnings.filterwarnings("ignore", ".*GPU available but not used.*")
warnings.filterwarnings(
    "ignore", ".*`LightningModule\.configure_optimizers` returned `None`.*"
)


def animate_2d_slinky_trajectories(
    trajectories: np.ndarray, dt: float, animation_slice=100
):
    """Plot a plotly animation of the motion of a DER

    Args:
        trajectory (torch.Tensor): 2D slinky trajectory (n_trajectories x n_frames x num_cycles x 3)
        dt (float): The time step during simulation
    """
    simulation_steps = trajectories.shape[1]
    interval = int(simulation_steps / animation_slice)
    trajectories = trajectories[:, ::interval]  # downsampling
    node_coords = trajectories.swapaxes(0, 1)
    # node_coords = trajectories
    # node_coords = (
    #     trajectories.clone().detach().cpu().transpose(1, 0).numpy()
    # )  # transpose the time axis to the first dim
    # num_frames = len(node_coords)
    # node_coords = [t.clone().cpu().numpy() for t in trajectories]
    bar_len = 0.02

    def calculate_bar_df(x, y, a, l=1):
        a = a.reshape(-1, 1)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        dx = -0.5 * l * np.sin(a)
        dy = 0.5 * l * np.cos(a)
        xx = np.hstack([x + dx, x - dx, np.ones_like(x) * np.nan]).ravel()
        yy = np.hstack([y + dy, y - dy, np.ones_like(y) * np.nan]).ravel()
        return pd.DataFrame({"x": xx, "y": yy})

    def make_frame_data(coords):
        frame_data = []
        if len(coords.shape) == 2:
            coords = coords[None, :]
        if len(coords.shape) != 3:
            raise ValueError(f"`coords` must have 2 or 3 dims. got {coords.shape=}")
        colors = px.colors.qualitative.Plotly
        for coord, color in zip(coords, itertools.cycle(colors)):
            x = coord[:, 0]
            y = coord[:, 1]
            a = coord[:, 2]
            bar_df = calculate_bar_df(x, y, a, bar_len)
            frame_data.extend(
                [
                    {
                        "type": "scatter",
                        "x": x,
                        "y": y,
                        "line": {"color": color, "width": 2},
                        # "marker": {"size": 4, "color": "red"},
                        "mode": "lines",
                    },
                    {
                        "type": "scatter",
                        "x": bar_df["x"],
                        "y": bar_df["y"],
                        "line": {"color": color, "width": 1},
                    },
                ]
            )
        return frame_data

    data = make_frame_data(node_coords[0])
    # axis = dict(
    #     showbackground=True,
    #     backgroundcolor="rgb(230, 230,230)",
    #     gridcolor="rgb(255, 255, 255)",
    #     zerolinecolor="rgb(255, 255, 255)",
    # )
    frames = []
    for k, coords in enumerate(node_coords):
        frame_data = make_frame_data(coords)
        frames.append(
            {
                "data": frame_data,
                # "traces": [0, 1],
                "name": f"frame {k}",
            }
        )

    sliders = [
        dict(
            steps=[
                dict(
                    method="animate",  # Sets the Plotly method to be called when the
                    # slider value is changed.
                    args=[
                        [
                            "frame {}".format(k)
                        ],  # Sets the arguments values to be passed to
                        # the Plotly method set in method on slide
                        dict(
                            mode="immediate",
                            frame=dict(duration=50, redraw=False),
                            transition=dict(duration=0),
                        ),
                    ],
                    label="{:.2f}".format(k * interval * dt),
                )
                for k in range(animation_slice)
            ],
            transition=dict(duration=0),
            x=0,  # slider starting position
            y=0,
            currentvalue=dict(
                font=dict(size=12), prefix="Time: ", visible=True, xanchor="center"
            ),
            len=1.0,
        )  # slider length)
    ]

    x_all = node_coords[..., 0]
    xmax, xmin = x_all.max(), x_all.min()
    xrange = xmax - xmin
    y_all = node_coords[..., 1]
    ymax, ymin = y_all.max(), y_all.min()
    yrange = ymax - ymin
    margin = 0.05

    layout = dict(
        title="Animating a 2D Slinky",
        autosize=True,
        # width=600,
        # height=600,
        showlegend=False,
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
        ),
        xaxis_range=[xmin - margin * xrange, xmax + margin * xrange],
        yaxis_range=[ymin - margin * yrange, ymax + margin * yrange],
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0,
                x=1.15,
                xanchor="right",
                yanchor="top",
                pad=dict(t=0, r=10),
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=20, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    )
                ],
            )
        ],
        sliders=sliders,
    )
    fig = go.Figure(data=data, layout=layout, frames=frames)
    # fig.show(renderer="notebook_connected")
    return fig


@jaxtyped
@typechecked
def cross2d(
    tensor_1: Float[Array, "*batch 2"], tensor_2: Float[Array, "*batch 2"]
) -> Float[Array, "*batch 2"]:
    """Compute the cross product of two (arrays of) 2D vectors."""
    chex.assert_equal_shape([tensor_1, tensor_2])
    chex.assert_shape([tensor_1, tensor_2], (..., 2))
    cross_mat = jnp.array([[0.0, 1.0], [-1.0, 0.0]], dtype=tensor_1.dtype)
    return jnp.einsum("...i,ij,...j", tensor_1, cross_mat, tensor_2)


@jaxtyped
@typechecked
def dot(
    tensor_1: Float[Array, "*batch d"],
    tensor_2: Float[Array, "*batch d"],
    keepdims=False,
) -> Float[Array, "*batch"]:
    """Dot product of two tensors along their last axis."""
    return jnp.sum(tensor_1 * tensor_2, axis=-1, keepdims=keepdims)


@jaxtyped
@typechecked
def signed_angle_2d(
    v1: Float[Array, "*batch 2"], v2: Float[Array, "*batch 2"]
) -> Float[Array, "*batch"]:
    """Compute the signed angle between two (arrays of) 2D vectors."""
    return jnp.arctan2(cross2d(v1, v2), dot(v1, v2, keepdims=False))


@jaxtyped
@typechecked
def transform_cartesian_alpha_to_douglas_single(
    cartesian_alpha_input: Float[Array, "*batch 2 3"],
) -> Float[Array, "*batch 3"]:
    """
    Args:
        cartesian_alpha_input: input tensor of shape (... x 2 x 3). The last two
            dimensions describe pair of adjacent bars, where each bar has the
            coords (center_x, center_z, alpha)
    Returns:
        Douglas model's degrees of freedom of shape (... x 3).
        The last axis is (dxi, dz, dphi)
    """
    alpha_1 = cartesian_alpha_input[..., 0, 2]
    alpha_2 = cartesian_alpha_input[..., 1, 2]

    # l = (cartesian_alpha_input[..., 1, 0:2] - cartesian_alpha_input[..., 0, 0:2]).norm(
    #     dim=-1
    # )
    l = cartesian_alpha_input[..., 1, 0:2] - cartesian_alpha_input[..., 0, 0:2]

    psi = 0.5 * (alpha_1 + alpha_2)

    # dxi = l * torch.cos(psi)
    # dz = l * torch.sin(psi)
    sin = jnp.sin(psi)
    cos = jnp.cos(psi)
    dz = dot(l, jnp.stack([cos, sin], axis=-1))
    dxi = dot(l, jnp.stack([sin, -cos], axis=-1))

    dphi = alpha_2 - alpha_1

    return jnp.stack([dxi, dz, dphi], axis=-1)  # (... x 3)


@jaxtyped
@typechecked
def _convert_cartesian_alpha_to_douglas(
    state: Float[Array, "*batch cycles 6"]
) -> Float[Array, "*batch cycles-1 3"]:
    """_summary_

    Args:
        state (torch.Tensor): TensorType[..., "cycles": n, "state": 6]
    Returns:
        douglas (torch.Tensor): TensorType[..., "cycles": n-1, "douglas_coords": 3]
    """
    coords = state[..., :3]
    coord_pairs = jnp.stack((coords[..., :-1, :], coords[..., 1:, :]), axis=-2)
    return transform_cartesian_alpha_to_douglas_single(coord_pairs)


class SlinkyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_trajectories,
        test_trajectories,
        input_length,
        target_length,
        val_length,
        batch_size=8,
        perturb: float = 0,
    ):
        super().__init__()
        self.train_trajectories = train_trajectories
        self.test_trajectories = test_trajectories
        self.batch_size = batch_size
        self.input_length = input_length
        self.target_length = target_length
        self.test_length = val_length
        self.perturb = perturb

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = trajectory_dataset.TrajectoryDataset(
            self.train_trajectories,
            input_length=self.input_length,
            target_length=self.target_length,
        )
        self.test_dataset = trajectory_dataset.TrajectoryDataset(
            self.test_trajectories,
            input_length=self.input_length,
            target_length=self.test_length,
        )

    def train_dataloader(self):
        return self.train_dataset.to_dataloader(
            self.batch_size, "random", drop_last=True
        )

    def val_dataloader(self):
        return self.test_dataset.to_dataloader(
            self.batch_size, "sequential", drop_last=True
        )

    def test_dataloader(self):
        return self.test_dataset.to_dataloader(
            self.batch_size, "sequential", drop_last=True
        )

    # def transfer_batch_to_device(self, batch, device, dataloader_idx):
    #     [b.to_torch(device=device) for b in batch]
    #     return batch

    # def on_after_batch_transfer(self, batch, dataloader_idx):
    #     result = []
    #     for b in batch:
    #         if self.perturb:
    #             b["state"] += torch.randn_like(b["state"]) * self.perturb
    #         result.append(dict(b))
    #     return result


class DenseBlock(eqx.Module):
    # net: eqx.Module
    dim_per_layer: int
    num_layers: int
    layers: List[eqx.Module]

    def __init__(self, dim_per_layer, num_layers, *, key):
        # super(DenseBlock, self).__init__()
        self.dim_per_layer = dim_per_layer
        self.num_layers = num_layers

        layer_keys = jrandom.split(key, num_layers)
        layers = []
        for i, k in enumerate(layer_keys):
            layers.append(
                eqx.nn.Sequential(
                    [
                        eqx.nn.Linear(
                            dim_per_layer * i + dim_per_layer,
                            dim_per_layer,
                            key=k,
                        ),
                        eqx.nn.Lambda(jax.nn.softplus),
                    ]
                )
            )
        self.layers = layers
        # self.net = eqx.nn.Sequential(layer)

    def __call__(self, x, *, key=None):
        # def f(carry, block):
        #     output = block(carry)
        #     return output, jnp.concatenate((carry, output), axis=-1)
        for block in self.layers:
            y = block(x)
            x = jnp.concatenate((x, y), axis=-1)
        return x


# @jaxtyped
# @typechecked
class ODEFuncJax(eqx.Module):
    dim_per_layer: int
    num_layers: int
    net: eqx.Module
    boundaries: Tuple[int, int]

    def __init__(
        self, NeuronsPerLayer=32, NumLayer=5, Boundaries=(1, 1), *, key
    ):  # for boundary condition, 1 stands for fixed, 0 stands for free
        super(ODEFuncJax, self).__init__()
        self.dim_per_layer = NeuronsPerLayer
        self.num_layers = NumLayer
        # self.net = nn.Sequential(
        #     # nn.Linear(6 + 4, self.neuronsPerLayer),
        #     nn.Linear(6, self.neuronsPerLayer),
        #     snn.DenseBlock(self.neuronsPerLayer, self.numLayers),
        #     snn.Square(),
        #     nn.Linear(int(self.neuronsPerLayer * (self.numLayers + 1)), 1),
        # )
        l_k_1, d_k, l_k_2 = jrandom.split(key, 3)
        net = eqx.nn.Sequential(
            [
                eqx.nn.Linear(6, self.dim_per_layer, key=l_k_1),
                DenseBlock(self.dim_per_layer, self.num_layers, key=d_k),
                eqx.nn.Lambda(jnp.square),
                eqx.nn.Linear(
                    int(self.dim_per_layer * (self.num_layers + 1)), 1, key=l_k_2
                ),
            ]
        )
        self.net = eqx.filter_jit(net)

        self.boundaries = Boundaries

    @staticmethod
    def _make_triplet_catesian_alpha(
        x: Float[Array, "*batch cycles 3"]
    ) -> Float[Array, "*batch cycles-2 9"]:
        """Convert slinky coords to overlapping triplet coords, differentiably

        Args:
                x (torch.Tensor): slinky coords (n_batch x n_cycles x 3)
        Returns:
                torch.Tensor: Shape of (n_batch x (n_cycles-2) x 9). The last axis is
                (x_1, y_1, alpha_1, x_2, y_2, alpha_2, x_3, y_3, alpha_3)
        """
        x_prev = x[..., :-2, :]
        x_mid = x[..., 1:-1, :]
        x_next = x[..., 2:, :]
        return jnp.concatenate((x_prev, x_mid, x_next), axis=-1)

    @staticmethod
    def chiral_transformation_x_douglas(
        data: Float[Array, "*batchcycles 6"]
    ) -> Float[Array, "*batchcycles 6"]:
        # data is a 6-dim input vector
        # the output is a 6-dim vector, as the mirror of input data with respect to the x-axis
        new_data = data * jnp.array([1, -1, -1, 1, -1, -1])
        return new_data

    @staticmethod
    def chiral_transformation_z_douglas(
        data: Float[Array, "*batchcycles 6"]
    ) -> Float[Array, "*batchcycles 6"]:
        # data is a 6-dim input vector
        # the output is a 6-dim vector, as the mirror of input data with respect to the z-axis
        new_data = jnp.stack(
            [
                data[..., 3],
                -data[..., 4],
                data[..., 5],
                data[..., 0],
                -data[..., 1],
                data[..., 2],
            ],
            axis=-1,
        )
        return new_data

    @staticmethod
    def chiral_transformation_xz_douglas(
        data: Float[Array, "*batchcycles 6"]
    ) -> Float[Array, "*batchcycles 6"]:
        # data is a 6-dim input vector
        # the output is a 6-dim vector, as the 180 degrees rotation of input data
        new_data = jnp.stack(
            [
                data[..., 3],
                data[..., 4],
                -data[..., 5],
                data[..., 0],
                data[..., 1],
                -data[..., 2],
            ],
            axis=-1,
        )
        return new_data

    @staticmethod
    def chiral_transformation_x_triplet(
        data: Float[Array, "*batchcycles 6"]
    ) -> Float[Array, "*batchcycles 6"]:
        # data is a 6-dim input vector
        # the output is a 6-dim vector, as the mirror of input data with respect to the x-axis
        new_data = data * jnp.array([1, 1, -1, -1, -1, -1])
        return new_data

    @staticmethod
    def chiral_transformation_z_triplet(
        data: Float[Array, "*batchcycles 6"]
    ) -> Float[Array, "*batchcycles 6"]:
        # data is a 6-dim input vector
        # the output is a 6-dim vector, as the mirror of input data with respect to the z-axis
        new_data = jnp.stack(
            [
                data[..., 1],
                data[..., 0],
                data[..., 2],
                -data[..., 5],
                -data[..., 4],
                -data[..., 3],
            ],
            axis=-1,
        )
        return new_data

    @staticmethod
    def chiral_transformation_xz_triplet(
        data: Float[Array, "*batchcycles 6"]
    ) -> Float[Array, "*batchcycles 6"]:
        # data is a 6-dim input vector
        # the output is a 6-dim vector, as the 180 degrees rotation of input data
        new_data = jnp.stack(
            [
                data[..., 1],
                data[..., 0],
                -data[..., 2],
                data[..., 5],
                data[..., 4],
                data[..., 3],
            ],
            axis=-1,
        )
        return new_data

    @staticmethod
    def transform_cartesian_alpha_to_triplet(
        cartesian_alpha_input: Float[Array, "*batchcycles 9"]
    ) -> Float[Array, "*batchcycles 6"]:
        """
        Args:
            cartesian_alpha_input: input tensor of shape (... x 9). The columns are
                (center_x_1, center_z_1, alpha_1, center_x_2, center_z_2, alpha_2,
                ...)
        Returns:
            The (... x 6) shaped triplet coordinates with columns (l1, l2, theta,
            gamma1, gamma2, gamma3)
        """
        l1 = cartesian_alpha_input[..., 3:5] - cartesian_alpha_input[..., 0:2]
        l2 = cartesian_alpha_input[..., 6:8] - cartesian_alpha_input[..., 3:5]
        offset_angle = lax.stop_gradient(jnp.arctan2(l1[..., 1], l1[..., 0]))

        alpha_1 = cartesian_alpha_input[..., 2] - offset_angle
        alpha_2 = cartesian_alpha_input[..., 5] - offset_angle
        alpha_3 = cartesian_alpha_input[..., 8] - offset_angle

        theta = signed_angle_2d(l2, l1)

        gamma_2 = alpha_2 - 0.5 * (jnp.pi - theta)
        # print(gamma_2)
        gamma_2 = (gamma_2 + jnp.pi / 2) % jnp.pi - jnp.pi / 2
        # print(gamma_2)
        gamma_1 = alpha_2 - alpha_1
        gamma_3 = alpha_2 - alpha_3

        return jnp.stack(
            [
                jnp.linalg.norm(l1, axis=-1),
                jnp.linalg.norm(l2, axis=-1),
                theta,
                gamma_1,
                gamma_2,
                gamma_3,
            ],
            axis=-1,
        )  # (... x 6)

    @staticmethod
    def transform_cartesian_alpha_to_douglas(
        cartesian_alpha_input: Float[Array, "*batchcycles 9"],
    ) -> Float[Array, "*batchcycles 2 3"]:
        """
        Args:
            cartesian_alpha_input: input tensor of shape (... x 9). The columns are
                (center_x_1, center_z_1, alpha_1, center_x_2, center_z_2, alpha_2,
                ...)
        Returns:
            Douglas model's degrees of freedom of shape (... x 2 x 3).
            The last axis is (dxi, dz, dphi)
        """
        alpha_1 = cartesian_alpha_input[..., 2]
        alpha_2 = cartesian_alpha_input[..., 5]
        alpha_3 = cartesian_alpha_input[..., 8]

        l1 = cartesian_alpha_input[..., 3:5] - cartesian_alpha_input[..., 0:2]
        l2 = cartesian_alpha_input[..., 6:8] - cartesian_alpha_input[..., 3:5]

        # theta = signed_angle_2d(l2, l1)

        # l1_norm = l1.norm(dim=-1)
        # l2_norm = l2.norm(dim=-1)

        psi_1 = 0.5 * (alpha_1 + alpha_2)
        psi_2 = 0.5 * (alpha_2 + alpha_3)

        sin_1 = jnp.sin(psi_1)
        cos_1 = jnp.cos(psi_1)
        dz_1 = dot(l1, jnp.stack([cos_1, sin_1], axis=-1))
        dxi_1 = dot(l1, jnp.stack([sin_1, -cos_1], axis=-1))

        sin_2 = jnp.sin(psi_2)
        cos_2 = jnp.cos(psi_2)
        dz_2 = dot(l2, jnp.stack([cos_2, sin_2], axis=-1))
        dxi_2 = dot(l2, jnp.stack([sin_2, -cos_2], axis=-1))

        dphi_1 = alpha_2 - alpha_1
        dphi_2 = alpha_3 - alpha_2

        first_pair = jnp.stack([dxi_1, dz_1, dphi_1], axis=-1)
        second_pair = jnp.stack([dxi_2, dz_2, dphi_2], axis=-1)
        return jnp.stack([first_pair, second_pair], axis=-2)  # (... x 2 x 3)

    def preprocessing(self, coords: jnp.ndarray) -> jnp.ndarray:
        triplet_alpha_node_pos = self._make_triplet_catesian_alpha(coords)
        ############################# CONVERSION ##############################
        # triplet_coords = self.transform_cartesian_alpha_to_triplet(
        #     triplet_alpha_node_pos
        # )  # rigid body motion removal module
        ########################### END CONVERSION ############################

        # ############################## TEMPORARY ##############################
        triplet_coords = self.transform_cartesian_alpha_to_douglas(
            triplet_alpha_node_pos
            # coords_transform.transform_triplet_to_cartesian_alpha(triplet_coords)
        )
        triplet_coords = triplet_coords.reshape(triplet_coords.shape[:-2] + (-1,))
        # ############################ END TEMPORARY ############################

        augmented_x = jnp.stack(
            [
                triplet_coords,
                # ############################## TEMPORARY ##############################
                # trans.chiral_transformation_x(triplet_coords),
                # trans.chiral_transformation_y(triplet_coords),
                # trans.chiral_transformation_xy(triplet_coords),
                # ############################ END TEMPORARY ############################
                self.chiral_transformation_x_douglas(triplet_coords),
                self.chiral_transformation_z_douglas(triplet_coords),
                self.chiral_transformation_xz_douglas(triplet_coords),
            ],
            axis=0,
        )  # chirality module
        # augmented_x = self.augment_with_sin_cos(augmented_x)
        return augmented_x

    def augment_with_sin_cos(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate(
            [x, jnp.sin(x[..., [2, 3, 4, 5]]), jnp.cos(x[..., [2, 3, 4, 5]])],
            axis=-1,
        )

    def cal_energy(self, coords: jnp.ndarray) -> jnp.ndarray:
        # calculating the acceleration of the 2D slinky system
        # the dimensions of y are (num_samples, num_cycles, [x_dis,y_dis,alpha_dis,x_vel,y_vel,alpha_vel])
        # yp, ypp, ypa = self.construct_y(y)
        # yinput = torch.cat((ypp,yp,ypa),-1)
        augmented_x = self.preprocessing(coords)
        # print(augmented_x.shape)
        f = jax.vmap(self.net)
        shape_prefix = augmented_x.shape[:-1]
        out = f(
            augmented_x.reshape(-1, augmented_x.shape[-1])
            # * torch.tensor([1e3, 1e3, 1e1, 1e3, 1e3, 1e1]).to(augmented_x.device)
            # * torch.tensor([1e3, 1e3, 1e1, 1e1, 1e1, 1e1]).type_as(augmented_x)
            # * torch.tensor([1e3, 1e3, 1e1, 1e3, 1e3, 1e1]).type_as(augmented_x)
        )  # adding weights here because the amplitude of deltaX and deltaY is ~0.01 (m), the amplitude of angle is ~pi
        out = out.reshape(shape_prefix + (-1,))
        out = jnp.sum(out, axis=0, keepdims=False)
        return out

    @eqx.filter_jit
    def calDeriv(self, y):
        # with torch.enable_grad():
        # coords = y[..., :3].clone().requires_grad_(True)
        coords = y[..., :3]

        def f(coords):
            return jnp.sum(self.cal_energy(coords))

        # out = jnp.sum(self.cal_energy(coords))
        grad = jax.grad(f)(coords)
        # deriv = torch.autograd.grad(
        #     [out], [coords], retain_graph=True, create_graph=True
        # )  # "backward in forward" feature, deriving the equivariant force from the invariant energy
        # # the dimensions of grad are (..., num_samples, num_cycles, 3)
        # grad = deriv[0]
        if self.boundaries[0] == 1:
            # grad[..., 0, :] = 0
            grad = grad.at[..., 0, :].set(0)
        if self.boundaries[1] == 1:
            # grad[..., -1, :] = 0
            grad = grad.at[..., -1, :].set(0)
        return grad
        # if grad is not None:
        # 	aug = torch.zeros_like(grad)[..., 0:1, :]
        # 	return self.contruct_grad(grad, aug)

    def __call__(self, y):
        grad = self.calDeriv(y)
        # if grad is not None:
        # the dimensions of the return value are (num_samples, num_cycles, 6)
        # print(f"{grad=}")
        return grad  # * torch.tensor([1e2, 1e2, 1]).type_as(grad)


@jaxtyped
@typechecked
class ODEFuncQuadratic(eqx.Module):
    """Douglas force calculator

    Input is cartesian_alpha coordinates of a slinky
    Output is the elastic force w.r.t. the coordinates based on Douglas model
    """

    c_dxi: jnp.ndarray
    c_log_dxi2: jnp.ndarray
    c_dz: jnp.ndarray
    c_log_dz2: jnp.ndarray
    c_dphi: jnp.ndarray
    c_log_dphi2: jnp.ndarray
    c_dxi_dz: jnp.ndarray
    c_dxi_dphi: jnp.ndarray
    c_dz_dphi: jnp.ndarray
    b: jnp.ndarray
    douglas: bool = False
    boundaries: Tuple[int, int] = (1, 1)

    def __init__(self, douglas: bool = False, Boundaries=(1, 1), *, key=None, **kwargs):
        super().__init__()
        self.c_dxi = jnp.array(0.0)
        self.c_log_dxi2 = jnp.array(-5.0)
        self.c_dz = jnp.array(0.0)
        self.c_log_dz2 = jnp.array(-5.0)
        self.c_dphi = jnp.array(0.0)
        self.c_log_dphi2 = jnp.array(-5.0)
        self.c_dxi_dz = jnp.array(0.0)
        self.c_dxi_dphi = jnp.array(0.0)
        self.c_dz_dphi = jnp.array(0.0)
        self.b = jnp.array(0.0)
        self.douglas = douglas
        self.boundaries = Boundaries

    def _make_pair_cartesian_alpha_coords(
        self, x: Float[Array, "*batch cycles 3"]
    ) -> Float[Array, "*batch cycles-1 2 3"]:
        """Convert slinky coords to overlapping triplet coords, differentiably

        Args:
                x (jnp.ndarray): slinky coords (n_batch x n_cycles x 3)
        Returns:
                jnp.ndarray: Shape of (n_batch x (n_cycles-1) x 2 x 3).
        """
        x_prev = x[..., :-1, :]
        # x_mid = x[:, 1:-1, :]
        x_next = x[..., 1:, :]
        return jnp.stack((x_prev, x_next), axis=-2)

    def _preprocess(self, coords: jnp.ndarray) -> jnp.ndarray:
        """
        Convert cartesian_alpha coords to Douglas coords

        Args:
                coords (jnp.ndarray): TensorType[..., "cycles": n, "coords": 3]
        Returns:
                douglas_coords (jnp.ndarray): TensorType[..., "pairs": n-1, "coords": 3]
        """
        bar_pairs = self._make_pair_cartesian_alpha_coords(coords)
        # bar_pairs[..., 2] += jnp.pi / 2  # adjust alpha to be the angle wrt x axis
        bar_pairs = bar_pairs.at[..., 2].add(jnp.pi / 2)
        return transform_cartesian_alpha_to_douglas_single(bar_pairs)

    def cal_energy(self, coords: jnp.ndarray) -> jnp.ndarray:
        """Calculate Slinky energy

        Args:
                coords (jnp.ndarray): TensorType[..., "cycles": n, "coords": 3]

        Returns:
                energy (jnp.ndarray): TensorType[..., "pairs": n-1]
        """
        douglas_coords = self._preprocess(coords)
        dxi = douglas_coords[..., 0]
        dz = douglas_coords[..., 1]
        dphi = douglas_coords[..., 2]
        c_dxi_scale = 0.002
        c_dz_scale = 0.002
        c_dphi_scale = 0.002
        c_dxi_dz = 0.002
        c_dxi_dphi_scale = 0.002
        c_dz_dphi_scale = 0.002
        # if self.douglas:
        #     E = (
        #         self.c_dxi * c_dxi_scale * jnp.tanh(dxi)
        #         + self.c_log_dxi2.exp() * dxi**2
        #         + self.c_log_dz2.exp() * dz**2
        #         + self.c_log_dphi2.exp() * dphi**2
        #     )
        # else:
        #     E = (
        #         self.c_dxi * c_dxi_scale * dxi
        #         + self.c_log_dxi2.exp() * dxi**2
        #         + self.c_dz * c_dz_scale * dz
        #         + self.c_log_dz2.exp() * dz**2
        #         + self.c_dphi * c_dphi_scale * dphi
        #         + self.c_log_dphi2.exp() * dphi**2
        #         + self.c_dxi_dz * c_dxi_dz * dxi * dz
        #         + self.c_dxi_dphi * c_dxi_dphi_scale * dxi * dphi
        #         + self.c_dz_dphi * c_dz_dphi_scale * dz * dphi
        #     )
        E = lax.cond(
            self.douglas,
            lambda model, dxi, dz, dphi: (
                model.c_dxi * c_dxi_scale * dxi
                + jnp.exp(model.c_log_dxi2) * dxi**2
                + jnp.exp(model.c_log_dz2) * dz**2
                + jnp.exp(model.c_log_dphi2) * dphi**2
            ),
            lambda model, dxi, dz, dphi: (
                model.c_dxi * c_dxi_scale * dxi
                + jnp.exp(model.c_log_dxi2) * dxi**2
                + model.c_dz * c_dz_scale * dz
                + jnp.exp(model.c_log_dz2) * dz**2
                + model.c_dphi * c_dphi_scale * dphi
                + jnp.exp(model.c_log_dphi2) * dphi**2
                + model.c_dxi_dz * c_dxi_dz * dxi * dz
                + model.c_dxi_dphi * c_dxi_dphi_scale * dxi * dphi
                + model.c_dz_dphi * c_dz_dphi_scale * dz * dphi
            ),
            self,
            dxi,
            dz,
            dphi,
        )
        return E

    @eqx.filter_jit
    def __call__(self, coords):
        coords = coords[..., :3]

        @jax.grad
        def f(coords):
            return self.cal_energy(coords).sum()

        grad = f(coords)

        if self.boundaries[0] == 1:
            grad = grad.at[..., 0, :].set(0)
        if self.boundaries[1] == 1:
            grad = grad.at[..., -1, :].set(0)

        return grad * jnp.array([1e2, 1e2, 1])


class ODEPhysJax(eqx.Module):
    m: float
    J: float
    g: float
    # vel_copy_mat: jnp.ndarray
    inv_inertia: jnp.ndarray
    # gVec: jnp.ndarray
    ODEFunc: eqx.Module
    boundaries: Tuple[int, int]

    def get_trainable_filter(self):
        filter_spec = jtu.tree_map(eqx.filters.is_inexact_array, self)
        filter_spec = eqx.tree_at(
            lambda tree: tree.inv_inertia,
            filter_spec,
            replace=False,
        )
        return filter_spec

    def __init__(self, ODEFunc: eqx.Module, Boundaries=(1, 1), *, key=None):
        super(ODEPhysJax, self).__init__()
        # the physical parameters
        self.m = 2.5e-3
        self.J = 0.5 * self.m * 0.033**2
        self.g = 9.8
        # register constant matrices
        # self.register_buffer("coeffMatrix1", torch.zeros(6, 6).float())
        # self.coeffMatrix1[3:, :3] = torch.eye(3).float()
        # self.vel_copy_mat = jnp.block(
        #     [[jnp.zeros((3, 3)), jnp.eye(3)], [jnp.zeros((3, 6))]]
        # )
        # self.register_buffer("coeffMatrix2", torch.zeros(6, 6).float())
        # self.coeffMatrix2[0:3, 3:] = -torch.diag(
        #     torch.tensor([1 / self.m, 1 / self.m, 1 / self.J])
        # ).float()
        # self.inv_inertia_mat = jnp.block([[jnp.zeros((3,3)), -jnp.diag(jnp.array([1 / self.m, 1 / self.m, 1 / self.J]))],[jnp.zeros((3,6))]])
        self.inv_inertia = jnp.array([1 / self.m, 1 / self.m, 1 / self.J])
        # self.register_buffer("gVec", torch.tensor([0, 0, 0, 0, -self.g, 0]))
        # self.gVec = jnp.array([0, 0, 0, 0, -self.g, 0])
        self.ODEFunc = ODEFunc
        self.boundaries = Boundaries

    # def construct_g(self, y):
    #     gravityGrad = jnp.zeros_like(y)
    #     if self.boundaries[0] == 1 and self.boundaries[1] == 1:  # both ends fixed
    #         gravityGrad = gravityGrad.at[..., 1:-1, :].set(self.gVec)
    #     elif (
    #         self.boundaries[0] == 0 and self.boundaries[1] == 1
    #     ):  # left end free, right end fixed
    #         gravityGrad[..., 0:-1, :] = self.gVec
    #     elif (
    #         self.boundaries[0] == 1 and self.boundaries[1] == 0
    #     ):  # right end free, left end fixed
    #         gravityGrad[..., 1:, :] = self.gVec
    #     elif self.boundaries[0] == 0 and self.boundaries[1] == 0:  # both ends free
    #         gravityGrad[..., :, :] = self.gVec
    #     else:
    #         raise RuntimeError("boundary conditions not allowed")
    #     return gravityGrad

    def __call__(self, t: jnp.ndarray, y: jnp.ndarray, args=None) -> jnp.ndarray:
        grad = self.ODEFunc(y)
        # gravityGrad = self.construct_g(y)
        # if grad is not None:
        # the dimensions of the return value are (num_samples, num_cycles, 6)
        # dy_dt = jnp.einsum("...i,ij->...j", y, self.vel_copy_mat)
        # a = jnp.einsum("...i,i->...i", -grad, self.inv_inertia)
        a = -grad
        # dy_dt = dy_dt.at[..., -3:].add(a)
        dy_dt = jnp.concatenate([y[..., -3:], a], axis=-1)

        left = right = None
        if self.boundaries[0] == 1:
            left = 1
        if self.boundaries[1] == 1:
            right = -1

        gravity_idx = -2
        dy_dt = dy_dt.at[..., left:right, gravity_idx].add(-self.g)
        return dy_dt


def get_eqx_model_state(model: eqx.Module, filter_spec=eqx.is_array) -> List:
    tree_weights, tree_other = eqx.partition(model, filter_spec)
    flattened_weights, _ = jtu.tree_flatten(tree_weights)
    return flattened_weights


def set_eqx_model_state(
    model: eqx.Module, weights: List, filter_spec=eqx.is_array
) -> eqx.Module:
    tree_weights, tree_other = eqx.partition(model, filter_spec)
    tree_def = jtu.tree_structure(tree_weights)
    loaded_weights = jtu.tree_unflatten(tree_def, weights)
    return eqx.combine(loaded_weights, tree_other)


class SlinkyTrajectoryRegressor(pl.LightningModule):
    model: eqx.Module

    def __init__(
        self,
        dim_per_layer: int,
        n_layers: int,
        key: jrandom.KeyArray,
        kinetic_reg: float = 0,
        n_cycles_loss: int = 20,
        # net_type: str = "ESNN",
        lr: float = 1e-3,
        weight_decay: float = 0,
        odeint_method: str = "dopri5",
        angular_loss_weight: float = 1,
        pretrained_net: Optional[str] = None,
        length_scaling: float = 1,
        loss_coord_frame: str = "global",
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # if net_type == "ESNN":
        #     self.net = node_f.ODEFunc(
        #         NeuronsPerLayer=dim_per_layer, NumLayer=n_layers, Boundaries=(1, 1)
        #     )
        # elif net_type == "ESNN2":
        #     self.net = node_f.ODEFunc2(
        #         NeuronsPerLayer=dim_per_layer, NumLayer=n_layers, Boundaries=(1, 1)
        #     )
        # elif net_type == "EMLP":
        #     self.net = emlp_models.SlinkyForceODEFunc(
        #         num_layers=n_layers, boundaries=(1, 1)
        #     )
        # elif net_type == "quadratic":
        #     self.net = node_f.ODEFuncQuadratic(douglas=True, Boundaries=(1, 1))
        #     # self.net = node_f.ODEFuncQuadratic(douglas=False, boundaries=(1, 1))
        net_key, self.key = jrandom.split(key)

        # self.net = ODEFuncJax(
        #     NeuronsPerLayer=dim_per_layer,
        #     NumLayer=n_layers,
        #     Boundaries=(1, 1),
        #     key=net_key,
        # )
        # self.net = ODEFuncQuadratic(False, (1, 1), key=net_key)
        self.net = ODEFuncJax(32, 5, (1, 1), key=net_key)

        if pretrained_net:
            print("loading " + pretrained_net)
            self._load_checkpoint(self.net, pretrained_net)
            print("loading " + pretrained_net + " done")

        self.model = ODEPhysJax(self.net, Boundaries=(1, 1))
        # self.register_buffer(
        #     "feat_weights",
        #     jnp.array([1e2, 1e2, angular_loss_weight, 1e2, 1e2, angular_loss_weight]),
        # )
        self.feat_weights = jnp.array(
            [1e2, 1e2, angular_loss_weight, 1e2, 1e2, angular_loss_weight]
        )

        self.atol = 1e-4
        self.rtol = 1e-4
        self.n_cycles_loss = n_cycles_loss
        self.kinetic_reg = kinetic_reg
        self.mse = lambda x, y: jnp.mean((x - y) ** 2)
        self.length_scaling = length_scaling
        self.loss_coord_frame = loss_coord_frame

        self.forward_backward: Callable = eqx.filter_value_and_grad(
            self.forward, arg=self.model.get_trainable_filter()
        )

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("SlinkyTrajectoryRegressor")
        parser.add_argument("--dim_per_layer", type=int, default=32)
        parser.add_argument("--n_layers", type=int, default=5)
        parser.add_argument(
            "--kinetic_reg",
            type=float,
            default=0,
            help="The kinetic regularization coefficient",
        )
        parser.add_argument(
            "--n_cycles_loss", type=int, help="Number of cycles to calculate loss for"
        )
        parser.add_argument(
            "--net_type",
            default="ESNN",
            choices=["ESNN", "EMLP", "ESNN2", "quadratic"],
            type=str,
            help="The type of force prediction network",
        )
        parser.add_argument(
            "--odeint_method",
            default="dopri5",
            choices=[
                "dopri8",
                "dopri5",
                "bosh3",
                "fehlberg2",
                "adaptive_heun",
                "euler",
                "midpoint",
                "rk4",
                "explicit_adams",
                "implicit_adams",
                "fixed_adams",
                "scipy_solver",
            ],
            type=str,
            help="The odeint method",
        )
        parser.add_argument(
            "--lr", type=float, help="The learning rate of the meta steps"
        )
        parser.add_argument(
            "--angular_loss_weight",
            type=float,
            default=1,
            help="The weight of angular loss terms",
        )
        parser.add_argument(
            "--weight_decay", type=float, help="Weight decay of optimizer"
        )
        parser.add_argument(
            "--pretrained_net",
            type=str,
            default=None,
            help="Pretrained core network state dict path",
        )
        parser.add_argument(
            "--loss_coord_frame",
            type=str,
            choices=["global", "douglas"],
            default="global",
        )
        parser.add_argument(
            "--grad_clip_norm", type=float, default=None, help="Gradient clipping norm"
        )
        parser.add_argument(
            "--init_lr", type=float, default=0, help="Initial learning rate"
        )
        parser.add_argument(
            "--warmup_steps", type=int, default=50, help="Number of warmup steps"
        )
        parser.add_argument(
            "--peak_lr", type=float, default=1e-2, help="Peak learning rate"
        )
        parser.add_argument(
            "--decay_steps",
            type=int,
            default=20000,
            help="Number of decay steps. If None, use the number of steps in the dataset.",
        )
        parser.add_argument(
            "--end_lr", type=float, default=0.0, help="End learning rate"
        )
        parser.add_argument(
            "--max_timeframes",
            type=int,
            default=10,
            help="Max number of time steps in a sequence. When the "
            "length of input sequences exceeds this value, "
            "downsampling is performed to cap the input length to "
            "the model.",
        )
        return parent_parser

    # def _kinetic_function(self, t, x):
    #     f = self.model(t, x)
    #     return (f**2).sum()
    # return (f**2).sum(dim=tuple(range(1, len(f.shape) + 1)))

    def _regularizer_augmented_model(
        self, t: jnp.ndarray, aug_x: Tuple[jnp.ndarray, jnp.ndarray]
    ):
        x = aug_x[0]
        # e = aug_x[1]
        f = self.model(t, x)
        return (f, (f**2).mean(axis=0).sum())

    def _solve_dynamics(self, model, start, t: jnp.ndarray, method: str):
        t = t[0]  # TODO: this only applies to evenly-spaced trajectories
        if self.kinetic_reg:
            raise NotImplementedError("Kinetic regularization not implemented yet")
            # start = (start, torch.tensor(0).type_as(start))
            # output, regularization = self._odeint(
            #     start,
            #     t,
            #     method,
            #     fun=self._regularizer_augmented_model,
            #     adjoint_params=tuple(self.model.parameters()),
            #     adjoint_options=dict(norm="seminorm"),
            # )
            # duration = len(t)  # .detach_()
            # # print(f"{t=}")
            # # if duration == 0:
            # #     duration = 1
            # regularization = regularization[-1] / duration
        else:
            regularization = 0
            output = self._odeint(start, t, method, fun=model)
        output = output[1:].swapaxes(0, 1)
        return output, regularization

    def inference(self, x0, t, method):
        return self._solve_dynamics(self.model, x0, t, method)[0]

    def criterion(self, x, y):
        if self.loss_coord_frame == "global":
            return jnp.mean(((x - y) * self.feat_weights) ** 2) / (
                self.length_scaling**2
            )
        elif self.loss_coord_frame == "douglas":
            return jnp.mean((x - y) ** 2) / (self.length_scaling**2)

    def _odeint(
        self, start: jnp.ndarray, t: jnp.ndarray, method: str, fun: Callable, **kwargs
    ) -> jnp.ndarray:
        term = diffrax.ODETerm(fun)
        solver = diffrax.Dopri5()
        saveat = diffrax.SaveAt(ts=t)
        stepsize_controller = diffrax.PIDController(rtol=self.rtol, atol=self.atol)
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t[0],
            t1=t[-1],
            dt0=None,
            y0=start,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
        )
        pred_y = sol.ys
        return pred_y

    @staticmethod
    def forward(
        model: eqx.Module,
        dynamics_solver: Callable,
        input: jnp.ndarray,
        t: jnp.ndarray,
        true_output: jnp.ndarray,
        criterion: Callable,
        loss_coord_frame: str,
        n_cycles_loss: int,
        # method: str = "dopri5",
        # method: str = None,
        # regularization: bool = True,
        *,
        key,
    ):
        start = input[:, 0]

        # if method is None:
        #     method = self.hparams.odeint_method

        output, regularization = dynamics_solver(model, start, t, "dopri5")
        # print(f"{output.shape=}")
        # print(f"{true_output.shape=}")

        # mse = self.mse(output, true_output) / (args.length_scaling**2)
        if loss_coord_frame == "douglas":
            # select_cycles = jrandom.permutation(key, output.shape[-2] - 1)[
            #     : self.n_cycles_loss
            # ]
            loss_output = _convert_cartesian_alpha_to_douglas(output)
            loss_true_output = _convert_cartesian_alpha_to_douglas(true_output)
        elif loss_coord_frame == "global":
            select_cycles = jrandom.permutation(key, input.shape[-2])[:n_cycles_loss]
            loss_output = output[..., select_cycles, :]
            loss_true_output = true_output[..., select_cycles, :]
        loss = criterion(loss_output, loss_true_output)
        # if regularization:
        #     loss = loss + self.kinetic_reg * regularization
        return loss
        # return {
        #     "loss": loss,
        #     "output": output,
        #     "mse": mse,
        #     "kinetic_regularization": regularization,
        # }

    @eqx.filter_jit
    def _training_step(
        self,
        # batch: Tuple[batch.Batch, batch.Batch],
        input,
        time,
        true_output,
        key: jrandom.KeyArray,
        opt_state: optax.OptState,
    ) -> Tuple[jnp.ndarray, eqx.Module, optax.OptState]:
        loss, grad = self.forward_backward(
            self.model,
            self._solve_dynamics,
            input,
            time,
            true_output,
            self.criterion,
            self.loss_coord_frame,
            self.n_cycles_loss,
            key=key,
        )
        updates, next_opt_state = self.optimizer.update(grad, opt_state, self.model)
        model = eqx.apply_updates(self.model, updates)
        return loss, model, next_opt_state

    def log_params(self):
        self.log(
            "dxi",
            self.model.ODEFunc.c_dxi.tolist(),
            prog_bar=True,
            logger=True,
        )
        self.log(
            "dxi2",
            jnp.exp(self.model.ODEFunc.c_log_dxi2).tolist(),
            prog_bar=True,
            logger=True,
        )
        self.log(
            "dz2",
            jnp.exp(self.model.ODEFunc.c_log_dz2).tolist(),
            prog_bar=True,
            logger=True,
        )
        self.log(
            "dphi2",
            jnp.exp(self.model.ODEFunc.c_log_dphi2).tolist(),
            prog_bar=True,
            logger=True,
        )

    def visualize_batch(self, batch_gt, batch_pred, dt):
        for i, (gt, pred) in enumerate(zip(batch_gt, batch_pred)):
            anim = animate_2d_slinky_trajectories(
                np.stack([gt, pred], axis=0), dt, animation_slice=min(100, gt.shape[0])
            )
            buffer = io.StringIO()
            anim.write_html(buffer, full_html=False)
            anim_html = buffer.getvalue()
            wandb.log({f"train_anim/{i}": wandb.Html(anim_html, inject=False)})

    @staticmethod
    def _downsample_indices_random(total, n_samples, key):
        return jrandom.permutation(key, total)[:n_samples]

    @staticmethod
    def _downsample_indices_regular(total, n_samples):
        return jnp.arange(total)[:: (total // n_samples)][:n_samples]

    @staticmethod
    def _downsample_indices_even(total, n_samples):
        max_index = total - 1
        n_intervals = n_samples - 1
        interval, remainder = divmod(max_index, n_intervals)
        indices = np.cumsum((np.arange(n_intervals) < remainder).astype(int) + interval)
        return jnp.concatenate((jnp.zeros(1, dtype="int32"), indices))

    def _downsample_indices(self, total, n_samples):
        if total < n_samples:
            return jnp.arange(total)
        else:
            return self._downsample_indices_even(total, n_samples)

    def assemble_input_output(self, batch):
        input, true_output = (
            jnp.asarray(batch[0]["state"]),
            jnp.asarray(batch[1]["state"]),
        )
        # print(f"{true_output.shape=}")
        # print(f"{self.hparams.max_timeframes=}")
        time_inds = self._downsample_indices(true_output.shape[1], self.hparams.max_timeframes)
        true_output = true_output[:, time_inds]
        time = jnp.concatenate([batch[0]["time"], batch[1]["time"][:, time_inds]], axis=1)
        # print(f"{input.shape=}")
        # print(f"{true_output.shape=}")
        # print(f"{time.shape=}")
        return input, true_output, time

    def training_step(self, batch, batch_idx):
        key, self.key = jrandom.split(self.key)

        input, true_output, time = self.assemble_input_output(batch)

        loss, self.model, self.opt_state = self._training_step(
            input, time, true_output, key, self.opt_state
        )
        self.log(
            "train_loss",
            loss.tolist(),
            prog_bar=True,
            logger=True,
            batch_size=batch[0]["time"].shape[0],
        )
        self.log(
            "force_norm",
            jnp.linalg.norm(self.model.ODEFunc(input[:, 0])).tolist(),
            prog_bar=True,
            logger=True,
            batch_size=batch[0]["time"].shape[0],
        )

        # self.log_params()

        self.log(
            "lr",
            self.opt_state.hyperparams["learning_rate"].tolist(),
            prog_bar=True,
            logger=True,
            # batch_size=batch[0]["time"].shape[0],
        )
        # self.log("train_mse", mse, prog_bar=True, logger=True)
        # if self.kinetic_reg:
        #     self.log(
        #         "kinetic_regularization",
        #         kinetic_regularization,
        #         prog_bar=True,
        #         logger=True,
        #     )
        return None

    def fit(self, datamodule, epochs: int):
        self.configure_optimizers()
        self.metrics = {}
        dataloader = datamodule.train_dataloader()
        for epoch in tqdm.tqdm(range(epochs)):
            self.pbar = tqdm.tqdm(enumerate(dataloader))
            self.pbar.set_description(f"Epoch {epoch}")
            for batch_idx, batch in self.pbar:
                self.training_step(batch, batch_idx)
            self.log_params()

    # def log(self, name: str, value, prog_bar: bool, logger: bool, **kwargs):
    #     if logger and self.logger is not None:
    #         self.logger.log(name, value, **kwargs)
    #     self.metrics.update({name: value})
    #     self.pbar.set_postfix(self.metrics)
    #     # if prog_bar and self.progress_bar_callback is not None:
    #     #     self.progress_bar_callback.add_metric(name, value)

    def validation_step(self, batch, batch_idx):
        input, true_output, time = self.assemble_input_output(batch)
        key, self.key = jrandom.split(self.key)
        loss = self.forward(
            self.model,
            self._solve_dynamics,
            input,
            time,
            true_output,
            self.criterion,
            self.loss_coord_frame,
            input.shape[-2],
            key=key,
        )
        # self.eval()
        # result = self.forward(input, time, true_output, regularization=False)
        # loss = result["mse"]
        self.log(
            "val_loss",
            loss.tolist(),
            prog_bar=True,
            logger=True,
            batch_size=time.shape[0],
        )
        self.log(
            "avg_val_loss",
            loss.tolist() / time.shape[1],
            prog_bar=False,
            logger=True,
            batch_size=time.shape[0],
        )
        # for n, p in self.named_parameters():
        # self.log(f"params/{n}", p, prog_bar=False, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input, true_output, time = self.assemble_input_output(batch)
        key, self.key = jrandom.split(self.key)
        loss = self.forward(
            self.model,
            self._solve_dynamics,
            input,
            time,
            true_output,
            self.criterion,
            self.loss_coord_frame,
            input.shape[-2],
            key=key,
        )
        # self.eval()
        # result = self.forward(input, time, true_output, regularization=False)
        # loss = result["mse"]
        self.log(
            "test_loss",
            loss.tolist(),
            prog_bar=True,
            logger=True,
            batch_size=input.shape[0],
        )
        return loss

    def configure_optimizers(self):
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=self.hparams.init_lr,
            peak_value=self.hparams.peak_lr,
            warmup_steps=self.hparams.warmup_steps,
            decay_steps=self.hparams.decay_steps,
            end_value=self.hparams.end_lr,
        )

        def make_optimizer(grad_clip_norm, learning_rate, weight_decay):
            optimizer_pipeline = []
            if grad_clip_norm is not None:
                optimizer_pipeline.append(optax.clip_by_global_norm(grad_clip_norm))
            optimizer_pipeline.append(
                optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
            )
            return optax.chain(*optimizer_pipeline)

        self.optimizer = optax.inject_hyperparams(make_optimizer)(
            self.hparams.grad_clip_norm, schedule, self.hparams.weight_decay
        )

        # self.optimizer = optax.adabelief(learning_rate=self.hparams.lr)
        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))
        return None

    def load_state_dict(self, state_dict: List, strict: bool = True):
        set_eqx_model_state(self.model, state_dict, eqx.is_array)

    def state_dict(self) -> List:
        return get_eqx_model_state(self.model, eqx.is_array)

    def _load_checkpoint(self, net, pretrained_checkpoint: str):
        return eqx.tree_deserialise_leaves(pretrained_checkpoint, net)
        # raise NotImplementedError
        try:
            import pretrain_lightning

            print(f"Loading pretrained model from {pretrained_checkpoint}")
            pretrained_model = (
                pretrain_lightning.DouglasForceRegressor.load_from_checkpoint(
                    pretrained_checkpoint
                )
            )
            net.load_state_dict(pretrained_model.net.state_dict())
        except:
            print(
                "Load pretrained model failed. Trying to load it as a simple state_dict"
            )
            state_dict = torch.load(pretrained_checkpoint, map_location=self.device)
            net.load_state_dict(state_dict)
        return net


class ClipLengthStepper(pl.Callback):
    def __init__(
        self,
        init_clip_len: int,
        final_clip_len: int,
        anneal_clip_length_epochs: int,
        **kwargs,
    ) -> None:
        super().__init__()
        # self.freq: int = freq
        self.init_len: int = init_clip_len
        self.final_len: int = final_clip_len
        self.cur_len: int = init_clip_len
        self.epochs: int = anneal_clip_length_epochs
        self.steps = np.linspace(
            init_clip_len, final_clip_len, anneal_clip_length_epochs
        ).astype(int)
        self.counter = 0

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("ClipLengthStepper")
        parser.add_argument("--init_clip_len", type=int, default=1)
        parser.add_argument("--final_clip_len", type=int)
        parser.add_argument("--anneal_clip_length_epochs", type=int)
        return parent_parser

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        pl_module.log("clip_length", self.cur_len, prog_bar=True, logger=True)
        if self.counter < self.epochs:
            self.cur_len = self.steps[self.counter]
            trainer.train_dataloader.dataset.datasets.target_length = self.cur_len
            trainer.val_dataloaders[0].dataset.target_length = self.cur_len
            self.counter += 1


# class ClipVisualization(pl.Callback):
#     def __init__(
#         self,
#         vis_freq: int,
#         vis_num: int,
#         **kwargs,
#     ) -> None:
#         super().__init__()
#         self.freq: int = vis_freq
#         self.num: int = vis_num
#         self.counter = 0

#     @staticmethod
#     def add_argparse_args(parent_parser):
#         parser = parent_parser.add_argument_group("ClipVisualization")
#         parser.add_argument("--vis_freq", type=int, default=1)
#         parser.add_argument("--vis_num", type=int, default=1)
#         return parent_parser

#     def on_validation_epoch_end(
#         self, trainer: pl.Trainer, pl_module: pl.LightningModule
#     ) -> None:
#         if self.counter == 0:
#             wandb.log({"val_animation": wandb.Html(animation_html, inject=False)})
#         self.counter = (self.counter + 1) % self.freq


# class EquinoxCheckpoint(ModelCheckpoint):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def _save_model(self, filepath: str) -> None:
#         # save model and optimizer state
#         eqx.tree_serialise_leaves(filepath, self._trainer.model)
#         # save other states
#         self._save_function(filepath, self._trainer, self._trainer.lightning_module)

#     def _load_model(self, filepath: str) -> None:
#         # load model and optimizer state
#         eqx.tree_deserialise_leaves(filepath, self._trainer.model)
#         # load other states
#         self._load_function(filepath, self._trainer, self._trainer.lightning_module)


def read_data(
    delta_t: float = 1e-3,
    length_scaling: float = 1,
    down_sampling: int = 10,
    **kwargs,
):
    # folder = "NeuralODE_Share2/SlinkyGroundTruth"
    folder = "slinky-is-sliding/SlinkyGroundTruth"
    num_cycles = 76
    true_y = np.loadtxt(folder + "/helixCoordinate_2D.txt", delimiter=",")
    true_v = np.loadtxt(folder + "/helixVelocity_2D.txt", delimiter=",")
    # true_y = torch.from_numpy(true_y).float()
    # true_v = torch.from_numpy(true_v).float()
    true_y = np.reshape(true_y, (true_y.shape[0], num_cycles, 3))
    true_v = np.reshape(true_v, (true_v.shape[0], num_cycles, 3))

    y_scale = true_y.std(axis=0).std(axis=0)
    v_scale = true_v.std(axis=0).std(axis=0)
    # true_y /= y_scale
    # true_v /= v_scale
    # print(f"{true_y.shape=}")
    # print(f"{true_v.shape=}")
    print(f"{y_scale=}")
    print(f"{v_scale=}")

    true_delta_t = 1e-3
    # delta_t = args.delta_t
    true_v = length_scaling * true_v * true_delta_t / delta_t
    true_y = length_scaling * true_y
    # delta_t = 1.0

    time = np.arange(true_y.shape[0]) * delta_t
    return (
        batch.Batch(
            state=np.concatenate((true_y, true_v), axis=-1)[::down_sampling],
            time=time[::down_sampling],
        ),
        num_cycles,
        y_scale,
        v_scale,
    )


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(
        default_config_files=["node_default_config.yaml"],
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )

    parser.add_argument(
        "--log", action="store_true", default=False, help="Whether to enable the logger"
    )
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--shuffle", type=bool, help="Whether to shuffle the data")
    # parser.add_argument("--grad_clip", type=float, default=0, help="Gradient clipping")
    # parser.add_argument("--min_epochs", type=int, help="Min training epochs")
    parser.add_argument("--seed", type=int, default=0, help="The random seed")
    # parser.add_argument("--devices", action="append", type=int, help="Torch device")
    parser.add_argument(
        "--incr_freq", type=int, help="Clip length increment frequency in epochs"
    )
    parser.add_argument("--max_length", type=int, help="Max clip length")
    parser.add_argument("--val_length", type=int, help="Max clip length")
    parser.add_argument("--init_length", type=int, help="Initial clip length")
    parser.add_argument(
        "--down_sampling", type=int, default=10, help="Data downsampling"
    )
    parser.add_argument(
        "--delta_t", type=float, default=0.001, help="Delta t assumed in data"
    )
    parser.add_argument(
        "--length_scaling",
        type=float,
        default=1,
        help="Scaling of length assumed in data",
    )
    parser.add_argument(
        "--perturb",
        type=float,
        default=0,
        help="Gaussian perturbation augmentation scale",
    )

    parser.add_argument("--name", default=None, type=str, help="Run name")
    parser.add_argument(
        "--project_name", default="NeuralSlinky", type=str, help="Run name"
    )

    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument(
        "--test_only",
        action="store_true",
        default=False,
        help="Only evaluate the model. (Not very meaningful if not loading a trained checkpoint)",
    )

    ClipLengthStepper.add_argparse_args(parser)
    SlinkyTrajectoryRegressor.add_argparse_args(parser)
    pl.Trainer.add_argparse_args(parser)
    args, _ = parser.parse_known_args()

    slinky_data, num_cycles, y_scale, v_scale = read_data(**vars(args))

    pl.seed_everything(args.seed)
    training_cutoff_ratio = 0.8
    training_cutoff = int(len(slinky_data) * training_cutoff_ratio)
    # training_cutoff = 65

    data_module = SlinkyDataModule(
        [slinky_data[:training_cutoff]],
        [slinky_data[training_cutoff:]],
        # [slinky_data[:training_cutoff]],
        input_length=1,
        target_length=args.init_length,
        val_length=args.val_length,
        batch_size=args.batch_size,
        perturb=args.perturb,  # .01
    )
    data_module.setup()
    print(f"{len(data_module.train_dataloader())=}")
    print(f"{len(data_module.val_dataloader())=}")

    if args.log:
        logger = WandbLogger(args.name, project=args.project_name)
    else:
        logger = WandbLogger(args.name, project=args.project_name, mode="disabled")

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join("jax_checkpoints", wandb.run.name),
        # filename="best-checkpoint",
        save_top_k=3,
        save_last=True,
        verbose=True,
        monitor="avg_val_loss",
        mode="min",
    )
    # checkpoint_callback =

    # early_stopping_callback = EarlyStopping(monitor="val_loss", patience=4)

    wandb.config.update(args)

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        enable_checkpointing=True,
        reload_dataloaders_every_n_epochs=1,
        callbacks=[
            checkpoint_callback,
            # early_stopping_callback,
            # ClipLengthStepper(freq=args.incr_freq, max_len=args.max_length),
            ClipLengthStepper(**vars(args)),
        ],
        #     # profiler="advanced",
        #     # fast_dev_run=True
    )

    if args.checkpoint:
        model = SlinkyTrajectoryRegressor.load_from_checkpoint(
            args.checkpoint  # , dim_per_layer=32, n_layers=5
        )
        # model.load_state_dict(checkpoint.state_dict())
    else:
        model = SlinkyTrajectoryRegressor(**vars(args), key=jrandom.PRNGKey(args.seed))

    if not args.test_only:
        trainer.fit(model=model, datamodule=data_module)
        # model.fit(data_module, args.min_epochs)

    start_point = slinky_data[0]["state"][None, None, :]
    # print(start_point)
    # sys.exit()
    model.eval()
    # evaluate_time = torch.from_numpy(slinky_data["time"]).to(get_model_device(model))
    evaluate_time = slinky_data["time"]

    true_trajectory = slinky_data["state"].reshape(-1, num_cycles, 2, 3)[1:, :, 0, :]

    print(f"{start_point.shape=}")
    print(f"{evaluate_time.shape=}")
    output = model.inference(start_point, evaluate_time[None, :], method="dopri5")

    print(jnp.linalg.norm(model.model(None, slinky_data["state"]), axis=-1)[-10:])
    # output = result["output"]
    output = np.squeeze(output)

    pred_trajectory = output.reshape(-1, num_cycles, 2, 3)[..., 0, :]

    evaluate_animation = animate_2d_slinky_trajectories(
        np.stack([true_trajectory, pred_trajectory]), 0.01
    )

    animation_html_file = "animation_tmp.html"
    evaluate_animation.write_html(animation_html_file, full_html=False)
    with open(animation_html_file, "r") as f:
        evaluate_animation_html = "\n".join(f.readlines())

    # evaluate_animation.show()

    # torch.save(pred_trajectory, "pred_trajectory.pth")
    # torch.save(true_trajectory[1:], "true_trajectory.pth")

    wandb.log({"evaluate_animation": wandb.Html(evaluate_animation_html, inject=False)})
    eqx.tree_serialise_leaves("jax_quadratic_model.ckpt", model.model)

# m = ODEFuncJax(32, 5, key=jrandom.PRNGKey(42))
# train_dataloader = data_module.train_dataloader()
# x = next(iter(train_dataloader))[0].state[:, 0]
# neg_force = m(x)
# print(f"{neg_force.shape=}")

# dense_block = DenseBlock(32, 5, key=jrandom.PRNGKey(42))
# # dense_block = jax.jit(dense_block)
# dense_block = eqx.filter_jit(dense_block)
# output = dense_block(jnp.arange(32))
# print(f"{output.shape=}")
