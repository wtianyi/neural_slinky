from typing import Callable, Dict, Iterator, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jrandom

from jaxtyping import Array, Float, PyTree, jaxtyped
from typeguard import typechecked

import emlp.reps as reps
from emlp.reps import Rep
from emlp.reps import bilinear_weights
from emlp.nn import gated, gate_indices, uniform_rep

from emlp.groups import Group
from emlp.groups import SO, O

import equinox as eqx
import logging

import numpy as np

so2_rep = reps.V(SO(2))
node_rep = so2_rep**0 + so2_rep
triplet_rep = 3 * node_rep


def Sequential(*args):
    """Wrapped to mimic pytorch syntax"""
    return eqx.nn.Sequential(args)


def Linear(repin, repout):
    rep_W = repout << repin
    logging.info(f"Linear W components:{rep_W.size()} rep:{rep_W}")
    rep_bias = repout
    Pw = rep_W.equivariant_projector()
    Pb = rep_bias.equivariant_projector()
    return eqxLinear(Pw, Pb, (repout.size(), repin.size()), key=jax.random.PRNGKey(0))


class eqxLinear(eqx.nn.Linear):
    """Basic equivariant Linear layer from repin to repout."""

    Pw: jnp.ndarray
    Pb: jnp.ndarray
    shape: Tuple[int, int]

    def __init__(self, Pw, Pb, shape, key):
        # The actual in_features and out_features are shape[0] and shape[1]
        # Here is to accommodate the calculation in __call__
        super().__init__(in_features=shape[1], out_features=shape[0], key=key)
        self.Pw = Pw
        self.Pb = Pb
        self.shape = shape

    def __call__(self, x, *, key=None):  # (cin) -> (cout)
        # i, j = self.shape
        # w_init = hk.initializers.TruncatedNormal(1. / np.sqrt(i))
        # w = hk.get_parameter("w", shape=self.shape, dtype=x.dtype, init=w_init)
        # b = hk.get_parameter("b", shape=[i], dtype=x.dtype, init=w_init)
        W = (self.Pw @ self.weight.reshape(-1)).reshape(*self.shape)
        b = self.Pb @ self.bias
        return x @ W.T + b


def BiLinear(repin, repout):
    """Cheap bilinear layer (adds parameters for each part of the input which can be
    interpreted as a linear map from a part of the input to the output representation)."""
    Wdim, weight_proj = bilinear_weights(repout, repin)
    return eqxBiLinear(weight_proj, Wdim, jax.random.PRNGKey(0))


class eqxBiLinear(eqx.Module):
    w: jnp.ndarray
    Wdim: int
    weight_proj: Callable

    def __init__(self, weight_proj: Callable, Wdim: int, key):
        super().__init__()
        self.weight_proj = weight_proj
        self.Wdim = Wdim
        self.w = jax.random.normal(key, shape=[Wdim])

    def __call__(self, x, *, key=None):
        # compatible with non sumreps? need to check
        # w_init = hk.initializers.TruncatedNormal(1.)
        # w = hk.get_parameter("w", shape=[self.Wdim], dtype=x.dtype, init=w_init)
        W = self.weight_proj(self.w, x)
        return 0.1 * (W @ x[..., None])[..., 0]


class GatedNonlinearity(
    eqx.Module
):  # TODO: add support for mixed tensors and non sumreps
    """Gated nonlinearity. Requires input to have the additional gate scalars
    for every non regular and non scalar rep. Applies swish to regular and
    scalar reps. (Right now assumes rep is a SumRep)"""

    rep: Rep

    def __init__(self, rep, *, key=None):
        super().__init__()
        self.rep = rep

    def __call__(self, values, *, key=None):
        gate_scalars = values[..., gate_indices(self.rep)]
        activations = jax.nn.sigmoid(gate_scalars) * values[..., : self.rep.size()]
        return activations


class EMLPBlock(eqx.Module):
    """Basic building block of EMLP consisting of G-Linear, biLinear,
    and gated nonlinearity."""

    linear: eqxLinear
    bilinear: eqxBiLinear
    nonlinearity: GatedNonlinearity

    def __init__(self, repin, repout, *, key=None):
        self.linear = Linear(repin, gated(repout))
        self.bilinear = BiLinear(gated(repout), gated(repout))
        self.nonlinearity = GatedNonlinearity(repout)

    def __call__(self, x, *, key=None):
        lin = self.linear(x, key=key)
        preact = self.bilinear(lin, key=key) + lin
        return self.nonlinearity(preact, key=key)

    # def block(x, *, key=None):
    #     lin = linear(x)
    #     preact = bilinear(lin) + lin
    #     return nonlinearity(preact)

    # return block


def EMLP(
    rep_in: Rep,
    rep_out: Rep,
    group: Group,
    ch: Union[int, Rep, Sequence[int], Sequence[Rep]] = 384,
    num_layers: int = 3,
):
    """Equivariant MultiLayer Perceptron.
    If the input ch argument is an int, uses the hands off uniform_rep heuristic.
    If the ch argument is a representation, uses this representation for the hidden layers.
    Individual layer representations can be set explicitly by using a list of ints or a list of
    representations, rather than use the same for each hidden layer.
    Args:
        rep_in (Rep): input representation
        rep_out (Rep): output representation
        group (Group): symmetry group
        ch (int or list[int] or Rep or list[Rep]): number of channels in the hidden layers
        num_layers (int): number of hidden layers
    Returns:
        Module: the EMLP equinox module."""
    logging.info("Initing EMLP (Haiku)")
    rep_in = rep_in(group)
    rep_out = rep_out(group)
    # Parse ch as a single int, a sequence of ints, a single Rep, a sequence of Reps
    if isinstance(ch, int):
        middle_layers = num_layers * [uniform_rep(ch, group)]
    elif isinstance(ch, Rep):
        middle_layers = num_layers * [ch(group)]
    else:
        middle_layers = [
            (c(group) if isinstance(c, Rep) else uniform_rep(c, group)) for c in ch
        ]
    # assert all((not rep.G is None) for rep in middle_layers[0].reps)
    reps = [rep_in] + middle_layers
    # logging.info(f"Reps: {reps}")
    network = Sequential(
        *[EMLPBlock(rin, rout) for rin, rout in zip(reps, reps[1:])],
        Linear(reps[-1], rep_out),
    )
    return network


def MLP(rep_in, rep_out, group, ch=384, num_layers=3):
    """Standard baseline MLP. Representations and group are used for shapes only."""
    cin = rep_in(group).size()
    cout = rep_out(group).size()
    mlp = eqx.nn.MLP(
        in_size=cin,
        out_size=cout,
        width_size=ch,
        depth=num_layers,
        activation=jax.nn.swish,
        key=jax.random.PRNGKey(0),
    )
    return mlp


class Flip(Group):
    """The alternating group in n dimensions"""

    def __init__(self):
        flip_first_bar = np.eye(9)
        flip_first_bar[:3, :3] = np.fliplr(np.eye(3))
        flip_middle_bar = np.eye(9)
        flip_middle_bar[3:6, 3:6] = np.fliplr(np.eye(3))
        flip_overall = np.empty((9, 9))
        identity = np.eye(9)
        flip_overall[:3] = identity[-3:]
        flip_overall[-3:] = identity[:3]
        self.discrete_generators = np.stack(
            [flip_first_bar, flip_middle_bar, flip_overall], axis=0
        )
        super().__init__(9)


def make_triplet_cartesian_alpha(
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
    # x_prev = jax.lax.stop_gradient(x[..., :-2, :])
    x_mid = x[..., 1:-1, :]
    x_next = x[..., 2:, :]
    # x_next = jax.lax.stop_gradient(x[..., 2:, :])
    return jnp.concatenate((x_prev, x_mid, x_next), axis=-1)


def transform_cartesian_alpha_to_cartesian(
    cartesian_alpha_input: Float[Array, "*batchcycles 9"], bar_length: float
) -> Float[Array, "*batchcycles 18"]:
    """
    Args:
        cartesian_alpha_input: input tensor of shape (... x 9). The columns are
            (center_x_1, center_z_1, alpha_1, center_x_2, center_z_2, alpha_2,
            ...)
        bar_length: A list of three numbers of the bar lengths
    Returns:
        Cartesian coordinates of all nodes in the 2D representation. Use the
        center node of the first cycle in the triplet as the origin, and the
        first link as the x axis.
        In total there are 9 nodes, so the output shape is (... x 18), in
        the order of (top_1, center_1, bottom_1, top_2, center_2, ...)
    """
    alpha_1 = cartesian_alpha_input[..., 2]
    alpha_2 = cartesian_alpha_input[..., 5]
    alpha_3 = cartesian_alpha_input[..., 8]

    center_1 = cartesian_alpha_input[..., jnp.array([0, 1])]
    center_2 = cartesian_alpha_input[..., jnp.array([3, 4])]
    center_3 = cartesian_alpha_input[..., jnp.array([6, 7])]

    def cal_top_bottom(center, alpha, bar_length):
        dx = 0.5 * bar_length * jnp.cos(alpha)
        dy = 0.5 * bar_length * jnp.sin(alpha)
        top = center.at[..., 0].add(dx)
        top = top.at[..., 1].add(dy)
        bottom = center.at[..., 0].add(-dx)
        bottom = bottom.at[..., 1].add(-dy)
        return top, bottom

    top_1, bottom_1 = cal_top_bottom(center_1, alpha_1, bar_length)
    top_2, bottom_2 = cal_top_bottom(center_2, alpha_2, bar_length)
    top_3, bottom_3 = cal_top_bottom(center_3, alpha_3, bar_length)
    return jnp.concatenate(
        [
            top_1,
            center_1,
            bottom_1,
            top_2,
            center_2,
            bottom_2,
            top_3,
            center_3,
            bottom_3,
        ],
        axis=-1,
    )


class SlinkyEnergyPredictorEMLP(eqx.Module):
    input_group: Group
    input_rep: Rep
    output_rep: Rep
    emlp: eqx.Module
    vemlp: Callable
    bar_length: float

    def __init__(self, dim_per_layer: int, num_layers: int, *, key=None):
        input_group = Flip() * O(2)
        reps.V.rho_dense(input_group.sample())
        self.input_group = Flip() * O(2)
        self.input_rep = reps.V(input_group)
        self.output_rep = reps.Scalar(input_group)
        # self.emlp = jax.vmap(
        self.emlp = EMLP(
            self.input_rep,
            self.output_rep,
            self.input_group,
            ch=dim_per_layer,
            num_layers=num_layers,
        )
        self.vemlp = eqx.filter_vmap(self.emlp, kwargs=dict(key=None))
        self.bar_length = 0.01

    @staticmethod
    def recenter(x: Float[Array, "cycles 18"]):
        reshaped_x = x.reshape(x.shape[:1] + (9, 2))
        mean = jnp.mean(reshaped_x, axis=-2, keepdims=True)
        return (reshaped_x - jax.lax.stop_gradient(mean)).reshape(x.shape)

    def __call__(
        self, cartesian_alpha_coords: Float[Array, "cycles 3"], *, key=None
    ) -> Float[Array, "1"]:
        triple_coords = make_triplet_cartesian_alpha(cartesian_alpha_coords)
        cartesian_coords = transform_cartesian_alpha_to_cartesian(
            triple_coords, self.bar_length
        )
        cartesian_coords = self.recenter(cartesian_coords)
        return self.vemlp(cartesian_coords, key=key).sum()


class SlinkyForceODEFunc(eqx.Module):
    energy_fun: eqx.Module
    cal_energy: Callable
    force_fun: Callable
    boundaries: Tuple[int, int]

    def __init__(self, dim_per_layer: int, num_layers: int, boundaries=(1, 1), *, key):
        # l_k_1, d_k, l_k_2 = jrandom.split(key, 3)
        self.boundaries = boundaries
        self.energy_fun = SlinkyEnergyPredictorEMLP(dim_per_layer, num_layers, key=key)
        batch_energy_fun = jax.vmap(self.energy_fun)

        def cal_energy(x):
            return batch_energy_fun(x).sum()

        self.cal_energy = cal_energy
        self.force_fun = jax.grad(cal_energy)

    def __call__(
        self, cartesian_alpha_pos_vel: Float[Array, "*batch cycles 6"], *, key=None
    ) -> Float[Array, "*batch cycles 3"]:
        cartesian_alpha_coords = cartesian_alpha_pos_vel[..., :3]
        force_pred = self.force_fun(cartesian_alpha_coords)
        if self.boundaries[0] == 1:
            force_pred = force_pred.at[..., 0, :].set(0)
        if self.boundaries[1] == 1:
            force_pred = force_pred.at[..., -1, :].set(0)
        return force_pred


# class SlinkyForcePredictorEMLP(torch.nn.Module):
#     def __init__(
#         self,
#         num_layers: int,
#     ) -> None:
#         super().__init__()
#         self.emlp = enn.EMLP(triplet_rep, triplet_rep, SO(2), num_layers=num_layers)

#     def forward(self, bar_alpha_node_pos) -> torch.Tensor:
#         force_pred = self.emlp(bar_alpha_node_pos)

#         return force_pred


# class SlinkyForceODEFunc(torch.nn.Module):
#     def __init__(self, num_layers: int, boundaries: Tuple[int, int]) -> None:
#         super().__init__()
#         self.slinky_force_predictor = SlinkyForcePredictorEMLP(num_layers)
#         self.boundaries = boundaries

#     @staticmethod
#     def _make_triplet_alpha_node_pos(x: torch.Tensor) -> torch.Tensor:
#         """Convert slinky coords to overlapping triplet coords, differentiably

#         Args:
#             x (torch.Tensor): slinky coords (n_batch x n_cycles x 3)
#         Returns:
#             torch.Tensor: Shape of (n_batch x (n_cycles-2) x 9). The last axis is
#             (alpha_1, alpha_2, alpha_3, x_1, y_1, x_2, y_2, x_3, y_3)
#         """
#         x_prev = x[:, :-2, :]
#         x_mid = x[:, 1:-1, :]
#         x_next = x[:, 2:, :]
#         offset = torch.atan2(
#             x_mid[..., 1] - x_prev[..., 1], x_mid[..., 0] - x_prev[..., 0]
#         )
#         return torch.stack(
#             (
#                 x_prev[..., 2] - offset,
#                 x_mid[..., 2] - offset,
#                 x_next[..., 2] - offset,
#                 x_prev[..., 0],
#                 x_prev[..., 1],
#                 x_mid[..., 0],
#                 x_mid[..., 1],
#                 x_next[..., 0],
#                 x_next[..., 1],
#             ),
#             dim=-1,
#         )

#     def forward(self, y):
#         # if self.boundaries != (1,1): # not both ends fixed
#         #     raise ValueError(f"Boundary condition {self.boundaries} is not implemented for")
#         coords = y[..., :3]
#         triplet_alpha_node_pos = self._make_triplet_alpha_node_pos(coords)
#         forces = self.slinky_force_predictor(triplet_alpha_node_pos)
#         result = torch.zeros_like(coords)  # (n_batch x n_cycles x 3)
#         result[..., :-2, 0] += forces[..., 3]
#         result[..., :-2, 1] += forces[..., 4]
#         result[..., :-2, 2] += forces[..., 0]
#         result[..., 1:-1, 0] += forces[..., 5]
#         result[..., 1:-1, 1] += forces[..., 6]
#         result[..., 1:-1, 2] += forces[..., 1]
#         result[..., 2:, 0] += forces[..., 7]
#         result[..., 2:, 1] += forces[..., 8]
#         result[..., 2:, 2] += forces[..., 2]
#         if self.boundaries[0] == 1:
#             result[..., 0, :] = 0
#         if self.boundaries[1] == 1:
#             result[..., -1, :] = 0
#         # if torch.any(torch.isnan(result)):
#         #     print(f"{y=}")
#         print(f"{torch.abs(result).max()=}")
#         return result
