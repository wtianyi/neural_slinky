#!/usr/bin/env python3
from typing import List
import torch
import numpy as np

"""For Douglas representation, the xi is the component of a link that's normal
to the angular bisector, and z is component that's along the angular bisector
"""

# * Cartesian
def transform_triplet_to_cartesian(
    cycle_triplet_coord: torch.Tensor, bar_lengths: List[float]
):
    """Transform triplet coordinates to 2d cartesian coordinates of the nodes

    Args:
        cycle_triplet_coord: (... x 6) shape. The names of the last axis are
            (l1, l2, theta, gamma1, gamma2, gamma3)
        bar_length: A list of three numbers of the bar lengths
    Returns:
        Cartesian coordinates of all nodes in the 2D representation. Use the
        center node of the first cycle in the triplet as the origin, and the
        first link as the x axis.
        In total there are 9 nodes, so the output shape is (... x 9 x 2), in
        the order of (top_1, center_1, bottom_1, top_2, center_2, ...)
    """
    # shape = cycle_triplet_coord.shape
    # device = cycle_triplet_coord.device

    l1 = cycle_triplet_coord[..., 0]
    l2 = cycle_triplet_coord[..., 1]
    theta = cycle_triplet_coord[..., 2]
    gamma1 = cycle_triplet_coord[..., 3]
    gamma2 = cycle_triplet_coord[..., 4]
    gamma3 = cycle_triplet_coord[..., 5]

    alpha2 = 0.5 * (np.pi - theta) + gamma2
    alpha1 = alpha2 - gamma1
    alpha3 = alpha2 - gamma3

    center_1 = torch.stack([l1.detach() - l1, torch.zeros_like(l1)], dim=-1)
    center_2 = torch.stack([l1, torch.zeros_like(l1)], dim=-1)
    center_3 = center_2.clone()
    center_3[..., 0] += l2 * torch.cos(theta)
    center_3[..., 1] -= l2 * torch.sin(theta)

    def cal_top_bottom(center, alpha, bar_length):
        top = center.clone()
        top[..., 0] += 0.5 * bar_length * torch.cos(alpha)
        top[..., 1] += 0.5 * bar_length * torch.sin(alpha)
        bottom = center.clone()
        bottom[..., 0] -= 0.5 * bar_length * torch.cos(alpha)
        bottom[..., 1] -= 0.5 * bar_length * torch.sin(alpha)
        return top, bottom

    top_1, bottom_1 = cal_top_bottom(center_1, alpha1, bar_lengths[0])
    top_2, bottom_2 = cal_top_bottom(center_2, alpha2, bar_lengths[1])
    top_3, bottom_3 = cal_top_bottom(center_3, alpha3, bar_lengths[2])
    return torch.stack(
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
        dim=-2,
    )


def cross2d(tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
    assert tensor_1.shape[-1] == 2
    assert tensor_2.shape[-1] == 2
    device = tensor_1.device
    dtype = tensor_1.dtype
    cross_mat = torch.tensor([[0, 1], [-1, 0]], device=device, dtype=dtype)
    return torch.einsum("...i,ij,...j", tensor_1, cross_mat, tensor_2)


def dot(tensor_1: torch.Tensor, tensor_2: torch.Tensor, keepdim=False):
    return torch.sum(tensor_1 * tensor_2, dim=-1, keepdim=keepdim)


def signed_angle_2d(v1: torch.Tensor, v2: torch.Tensor):
    return torch.atan2(cross2d(v1, v2), dot(v1, v2, keepdim=False))


def compute_bisector(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    r"""Compute the (un-normalized) angle bisector between two 2d vectors.

    Let angle $\theta$ be the angle s.t. a counter-clockwise rotation of v1 by
    $\theta$ will align v1 with v2. The returned bisector will fall within
    $\theta$. Note that this means `compute_bisector(v1, v2)` is not the same
    as `compute_bisector(v2, v1)`.

    Args:
        v1: The first vector(s). Shape should be (..., 2)
        v2: The second vector(s). Shape should be the same as `v1`

    Returns:
        The angle bisector vectors of shape (..., 2)
    """

    def _compute_bisector(v1: torch.Tensor, v2: torch.Tensor):
        norm_1 = v1.norm(dim=-1, keepdim=True)
        norm_2 = v2.norm(dim=-1, keepdim=True)
        norm = (norm_1 * norm_2).detach()

        return norm_1 * v2 / norm + norm_2 * v1 / norm

    def _rotate90(v: torch.Tensor):
        return torch.stack([-v[..., 1], v[..., 0]], dim=-1)

    bisec_1 = _compute_bisector(v1, v2)
    bisec_2 = _rotate90(_compute_bisector(v1, -v2))
    bisec_3 = -bisec_1

    case = (dot(v1, v2) < 0).long() + (
        (dot(v1, v2) >= 0) & (cross2d(v1, v2) < 0)
    ) * 2  # 0, 1 or 2
    #     print(case)
    case = case[..., None]
    case = case.expand_as(bisec_1)
    case = case[None, :]

    return torch.gather(
        torch.stack([bisec_1, bisec_2, bisec_3], dim=0), 0, case
    ).squeeze()


def transform_cartesian_to_triplet(cartesian_input: torch.Tensor):
    """
    Args:
        cartesian_input: (... x 9 x 2) shape. The names of the last axis are
            (top_1, center_1, bottom_1, top_2, center_2, ...)
    Returns:
        (... x 6) shape. The names of the last axis are
            (l1, l2, theta, gamma1, gamma2, gamma3)
    """
    top_1 = cartesian_input[..., 0, :]
    center_1 = cartesian_input[..., 1, :]
    bottom_1 = cartesian_input[..., 2, :]
    top_2 = cartesian_input[..., 3, :]
    center_2 = cartesian_input[..., 4, :]
    bottom_2 = cartesian_input[..., 5, :]
    top_3 = cartesian_input[..., 6, :]
    center_3 = cartesian_input[..., 7, :]
    bottom_3 = cartesian_input[..., 8, :]

    l1 = center_2 - center_1
    l2 = center_3 - center_2
    theta = signed_angle_2d(l2, l1)

    # TODO: make use of the bottom for real data?
    bar_1 = top_1 - center_1
    bar_2 = top_2 - center_2
    bar_3 = top_3 - center_3

    gamma_1 = signed_angle_2d(bar_1, bar_2)
    gamma_3 = signed_angle_2d(bar_3, bar_2)

    #     print(compute_bisector(l1, -l2))

    gamma_2 = signed_angle_2d(compute_bisector(l1, -l2), bar_2)
    return torch.stack(
        [l1.norm(dim=-1), l2.norm(dim=-1), theta, gamma_1, gamma_2, gamma_3], dim=-1
    )


def transform_cartesian_to_douglas(cartesian_input: torch.Tensor):
    """
    Args:
        cartesian_input: (... x 9 x 2) shape. The names of the last axis are
            (top_1, center_1, bottom_1, top_2, center_2, ...)
    Returns:
        (... x 2 x 3) shape. The last axis is (dxi, dz, dphi)
    """
    top_1 = cartesian_input[..., 0, :]
    center_1 = cartesian_input[..., 1, :]
    bottom_1 = cartesian_input[..., 2, :]
    top_2 = cartesian_input[..., 3, :]
    center_2 = cartesian_input[..., 4, :]
    bottom_2 = cartesian_input[..., 5, :]
    top_3 = cartesian_input[..., 6, :]
    center_3 = cartesian_input[..., 7, :]
    bottom_3 = cartesian_input[..., 8, :]

    l1 = center_2 - center_1
    l2 = center_3 - center_2
    #     theta = signed_angle_2d(l2, l1)

    # TODO: make use of the bottom for real data?
    bar_1 = top_1 - center_1
    bar_2 = top_2 - center_2
    bar_3 = top_3 - center_3

    alpha_1 = signed_angle_2d(l1, bar_1)
    alpha_2 = signed_angle_2d(l1, bar_2)
    alpha_2_2 = signed_angle_2d(l2, bar_2)
    alpha_3 = signed_angle_2d(l2, bar_3)

    gamma_1 = signed_angle_2d(bar_1, bar_2)
    gamma_3 = signed_angle_2d(bar_3, bar_2)

    l1_len = l1.norm(dim=-1)
    l2_len = l2.norm(dim=-1)
    psi_1 = 0.5 * (alpha_1 + alpha_2)
    psi_2 = 0.5 * (alpha_2_2 + alpha_3)
    dxi_1 = l1_len * torch.cos(psi_1)
    dz_1 = l1_len * torch.sin(psi_1)
    dxi_2 = l2_len * torch.cos(psi_2)
    dz_2 = l2_len * torch.sin(psi_2)
    #     print(compute_bisector(l1, -l2))

    first_pair_dof = torch.stack([dxi_1, dz_1, gamma_1], dim=-1)  # (... x 3)
    second_pair_dof = torch.stack([dxi_2, dz_2, -gamma_3], dim=-1)  # (... x 3)
    return torch.stack([first_pair_dof, second_pair_dof], dim=-2)  # (... x 2 x 3)


# * Center-line Cartesian + Alpha
def transform_triplet_to_cartesian_alpha(cycle_triplet_coord: torch.Tensor):
    """
    Args:
        cycle_triplet_coord: (... x 6) shape. The names of the last axis are
            (l1, l2, theta, gamma1, gamma2, gamma3)
    Returns:
        Cartesian coordinates of centerline nodes in the 2D representation and
        the three alpha angles describing the rotational position of the bars in
        the triplet.

        The output shape is (... x 9), in the order of (center_x_1, center_z_1,
        alpha_1, center_x_2, center_z_2, alpha_2, ...)
    """
    shape = cycle_triplet_coord.shape
    device = cycle_triplet_coord.device

    l1 = cycle_triplet_coord[..., 0]
    l2 = cycle_triplet_coord[..., 1]
    theta = cycle_triplet_coord[..., 2]
    gamma1 = cycle_triplet_coord[..., 3]
    gamma2 = cycle_triplet_coord[..., 4]
    gamma3 = cycle_triplet_coord[..., 5]

    alpha2 = 0.5 * (np.pi - theta) + gamma2
    alpha1 = alpha2 - gamma1
    alpha3 = alpha2 - gamma3

    center_1 = torch.stack([l1.detach() - l1, torch.zeros_like(l1)], dim=-1)
    center_2 = torch.stack([l1, torch.zeros_like(l1)], dim=-1)
    center_3 = center_2.clone()
    center_3[..., 0] += l2 * torch.cos(theta)
    center_3[..., 1] -= l2 * torch.sin(theta)

    return torch.cat(
        [
            center_1,
            alpha1[..., None],
            center_2,
            alpha2[..., None],
            center_3,
            alpha3[..., None],
        ],
        dim=-1,
    )


def transform_cartesian_alpha_to_douglas(cartesian_alpha_input: torch.Tensor):
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

    theta = signed_angle_2d(l2, l1)

    l1_norm = l1.norm(dim=-1)
    l2_norm = l2.norm(dim=-1)

    psi_1 = 0.5 * (alpha_1 + alpha_2)
    psi_2 = 0.5 * (alpha_2 + alpha_3) + theta

    dxi_1 = l1_norm * torch.cos(psi_1)
    dz_1 = l1_norm * torch.sin(psi_1)

    dxi_2 = l2_norm * torch.cos(psi_2)
    dz_2 = l2_norm * torch.sin(psi_2)

    dphi_1 = alpha_2 - alpha_1
    dphi_2 = alpha_3 - alpha_2

    first_pair = torch.stack([dxi_1, dz_1, dphi_1], dim=-1)
    second_pair = torch.stack([dxi_2, dz_2, dphi_2], dim=-1)
    return torch.stack([first_pair, second_pair], dim=-2)  # (... x 2 x 3)


def transform_cartesian_alpha_to_douglas_v2(cartesian_alpha_input: torch.Tensor):
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

    sin_1 = torch.sin(psi_1)
    cos_1 = torch.cos(psi_1)
    dxi_1 = dot(l1, torch.stack([cos_1, sin_1], dim=-1))
    dz_1 = dot(l1, torch.stack([sin_1, -cos_1], dim=-1))

    sin_2 = torch.sin(psi_2)
    cos_2 = torch.cos(psi_2)
    dxi_2 = dot(l2, torch.stack([cos_2, sin_2], dim=-1))
    dz_2 = dot(l2, torch.stack([sin_2, -cos_2], dim=-1))

    dphi_1 = alpha_2 - alpha_1
    dphi_2 = alpha_3 - alpha_2

    first_pair = torch.stack([dxi_1, dz_1, dphi_1], dim=-1)
    second_pair = torch.stack([dxi_2, dz_2, dphi_2], dim=-1)
    return torch.stack([first_pair, second_pair], dim=-2)  # (... x 2 x 3)


def transform_cartesian_alpha_to_triplet(cartesian_alpha_input: torch.Tensor):
    """
    Args:
        cartesian_alpha_input: input tensor of shape (... x 9). The columns are
            (center_x_1, center_z_1, alpha_1, center_x_2, center_z_2, alpha_2,
            ...)
    Returns:
        The (... x 6) shaped triplet coordinates with columns (l1, l2, theta,
        gamma1, gamma2, gamma3)
    """
    alpha_1 = cartesian_alpha_input[..., 2]
    alpha_2 = cartesian_alpha_input[..., 5]
    alpha_3 = cartesian_alpha_input[..., 8]

    l1 = cartesian_alpha_input[..., 3:5] - cartesian_alpha_input[..., 0:2]
    l2 = cartesian_alpha_input[..., 6:8] - cartesian_alpha_input[..., 3:5]

    theta = signed_angle_2d(l2, l1)

    gamma_2 = alpha_2 - 0.5 * (np.pi - theta)
    gamma_1 = alpha_2 - alpha_1
    gamma_3 = alpha_2 - alpha_3

    return torch.stack(
        [l1.norm(dim=-1), l2.norm(dim=-1), theta, gamma_1, gamma_2, gamma_3], dim=-1
    )  # (... x 2 x 3)


def transform_cartesian_alpha_to_cartesian(
    cartesian_alpha_input: torch.Tensor, bar_lengths: List[float]
):
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
        In total there are 9 nodes, so the output shape is (... x 9 x 2), in
        the order of (top_1, center_1, bottom_1, top_2, center_2, ...)
    """
    alpha_1 = cartesian_alpha_input[..., 2]
    alpha_2 = cartesian_alpha_input[..., 5]
    alpha_3 = cartesian_alpha_input[..., 8]

    l1 = cartesian_alpha_input[..., 3:5] - cartesian_alpha_input[..., 0:2]
    l2 = cartesian_alpha_input[..., 6:8] - cartesian_alpha_input[..., 3:5]

    theta = signed_angle_2d(l2, l1)

    center_1 = torch.stack([l1.detach() - l1, torch.zeros_like(l1)], dim=-1)
    center_2 = torch.stack([l1, torch.zeros_like(l1)], dim=-1)
    center_3 = center_2.clone()
    center_3[..., 0] += l2 * torch.cos(theta)
    center_3[..., 1] -= l2 * torch.sin(theta)

    def cal_top_bottom(center, alpha, bar_length):
        top = center.clone()
        top[..., 0] += 0.5 * bar_length * torch.cos(alpha)
        top[..., 1] += 0.5 * bar_length * torch.sin(alpha)
        bottom = center.clone()
        bottom[..., 0] -= 0.5 * bar_length * torch.cos(alpha)
        bottom[..., 1] -= 0.5 * bar_length * torch.sin(alpha)
        return top, bottom

    top_1, bottom_1 = cal_top_bottom(center_1, alpha_1, bar_lengths[0])
    top_2, bottom_2 = cal_top_bottom(center_2, alpha_2, bar_lengths[1])
    top_3, bottom_3 = cal_top_bottom(center_3, alpha_3, bar_lengths[2])
    return torch.stack(
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
        dim=-2,
    )


def transform_cartesian_alpha_to_cartesian_v2(
    cartesian_alpha_input: torch.Tensor, bar_lengths: List[float]
):
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
        In total there are 9 nodes, so the output shape is (... x 9 x 2), in
        the order of (top_1, center_1, bottom_1, top_2, center_2, ...)
    """
    alpha_1 = cartesian_alpha_input[..., 2]
    alpha_2 = cartesian_alpha_input[..., 5]
    alpha_3 = cartesian_alpha_input[..., 8]

    center_1 = cartesian_alpha_input[..., [0, 1]]
    center_2 = cartesian_alpha_input[..., [3, 4]]
    center_3 = cartesian_alpha_input[..., [6, 7]]

    def cal_top_bottom(center, alpha, bar_length):
        top = center.clone()
        top[..., 0] += 0.5 * bar_length * torch.cos(alpha)
        top[..., 1] += 0.5 * bar_length * torch.sin(alpha)
        bottom = center.clone()
        bottom[..., 0] -= 0.5 * bar_length * torch.cos(alpha)
        bottom[..., 1] -= 0.5 * bar_length * torch.sin(alpha)
        return top, bottom

    top_1, bottom_1 = cal_top_bottom(center_1, alpha_1, bar_lengths[0])
    top_2, bottom_2 = cal_top_bottom(center_2, alpha_2, bar_lengths[1])
    top_3, bottom_3 = cal_top_bottom(center_3, alpha_3, bar_lengths[2])
    return torch.stack(
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
        dim=-2,
    )


def transform_triplet_to_cartesian_alpha_single(cycle_triplet_coord: torch.Tensor):
    """
    Args:
        cycle_triplet_coord: (... x 6) shape. The names of the last axis are
            (l1, l2, theta, gamma1, gamma2, gamma3)
    Returns:
        Cartesian coordinates of centerline nodes in the 2D representation and
        the three alpha angles describing the rotational position of the bars in
        the triplet.

        The output shape is (... x 2 x 2 x 3). Each triplet corresponds to two
        pairs of adjacent bars, where each bar has the coords (center_x,
        center_z, alpha)
    """
    shape = cycle_triplet_coord.shape
    device = cycle_triplet_coord.device

    l1 = cycle_triplet_coord[..., 0]
    l2 = cycle_triplet_coord[..., 1]
    theta = cycle_triplet_coord[..., 2]
    gamma1 = cycle_triplet_coord[..., 3]
    gamma2 = cycle_triplet_coord[..., 4]
    gamma3 = cycle_triplet_coord[..., 5]

    alpha2 = 0.5 * (np.pi - theta) + gamma2
    alpha1 = alpha2 - gamma1
    alpha22 = alpha2 + theta
    alpha3 = alpha22 - gamma3

    center_1 = torch.stack([l1.detach() - l1, torch.zeros_like(l1)], dim=-1)
    center_2 = torch.stack([l1, torch.zeros_like(l1)], dim=-1)
    center_3 = center_2.clone()
    center_3[..., 0] += l2 * torch.cos(theta)
    center_3[..., 1] -= l2 * torch.sin(theta)

    first_pair = torch.stack(
        [
            torch.cat([center_1, alpha1[..., None]], dim=-1),
            torch.cat([center_2, alpha2[..., None]], dim=-1),
        ],
        dim=-2,
    )

    second_pair = torch.stack(
        [
            torch.cat([center_2, alpha22[..., None]], dim=-1),
            torch.cat([center_3, alpha3[..., None]], dim=-1),
        ],
        dim=-2,
    )

    return torch.stack(
        [first_pair, second_pair],
        dim=-3,
    )


def transform_cartesian_alpha_to_douglas_single(cartesian_alpha_input: torch.Tensor):
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

    l = (cartesian_alpha_input[..., 1, 0:2] - cartesian_alpha_input[..., 0, 0:2]).norm(
        dim=-1
    )

    psi = 0.5 * (alpha_1 + alpha_2)

    dxi = l * torch.cos(psi)
    dz = l * torch.sin(psi)

    dphi = alpha_2 - alpha_1

    return torch.stack([dxi, dz, dphi], dim=-1)  # (... x 3)


# def transform_cartesian_alpha_to_triplet(cartesian_alpha_input: torch.Tensor):
#     """
#     Args:
#         cartesian_alpha_input: input tensor of shape (... x 9). The columns are
#             (center_x_1, center_z_1, alpha_1, center_x_2, center_z_2, alpha_2,
#             ...)
#     Returns:
#         The (... x 6) shaped triplet coordinates with columns (l1, l2, theta,
#         gamma1, gamma2, gamma3)
#     """
#     alpha_1 = cartesian_alpha_input[..., 2]
#     alpha_2 = cartesian_alpha_input[..., 5]
#     alpha_3 = cartesian_alpha_input[..., 8]

#     l1 = cartesian_alpha_input[..., 3:5] - cartesian_alpha_input[..., 0:2]
#     l2 = cartesian_alpha_input[..., 6:8] - cartesian_alpha_input[..., 3:5]

#     theta = signed_angle_2d(l2, l1)

#     gamma_2 = alpha_2 - 0.5 * (np.pi - theta)
#     gamma_1 = alpha_2 - alpha_1
#     gamma_3 = alpha_2 - alpha_3

#     return torch.stack(
#         [l1.norm(dim=-1), l2.norm(dim=-1), theta, gamma_1, gamma_2, gamma_3], dim=-1
#     )  # (... x 2 x 3)

# * Misc
def group_into_triplets(slinky: torch.Tensor) -> torch.Tensor:
    """Group (batches of) whole slinky coordinates to groups of triplets.

    Args:
        slinky (torch.Tensor): (... x $L$ x $M$) where $L$ is the number of cycles in a slinky, $M$ is the number of coordinates for each cycle

    Returns:
        torch.Tensor: (... x $L-2$ x 3 x $M$)
    """
    first = slinky[..., 0:-2, :]
    second = slinky[..., 1:-1, :]
    third = slinky[..., 2:, :]
    return torch.stack([first, second, third], dim=-2)
