# ---
# jupyter:
#   jupytext:
#     formats: ipynb,md,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: 'Python 3.7.4 64-bit (''py37'': conda)'
#     language: python
#     name: python37464bitpy37conda606da32f2b124e7b8f99ee90fe706c33
# ---

# %%
import torch
import numpy as np

# %%
import scipy.io as scio
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

# %%
from glob import glob
import os

# %%
from neural_slinky import utils


# %%
def preprocess(df, inplace=True):
    if not inplace:
        df = df.copy()
    df.loc[:, "gamma2"] = (df.loc[:, "gamma2"] + np.pi/2) % (np.pi) - np.pi/2
    return df


# %%
def transform_z_xi(df):
    result = pd.DataFrame(columns=["l1", "l2", "theta", "d_z_1", "d_xi_1", "d_z_2", "d_xi_2", "gamma1", "gamma2", "gamma3"])
    result["l1"] = df["l1"]
    result["l2"] = df["l2"]
    result["theta"] = df["theta"]
    result["gamma1"] = df["gamma1"]
    result["gamma2"] = df["gamma2"]
    result["gamma3"] = df["gamma3"]
    
    
    phi_1 = 0.5 * (np.pi - df["theta"]) + df["gamma2"] - 0.5 * df["gamma1"]
    phi_2 = 0.5 * (np.pi + df["theta"]) + df["gamma2"] - 0.5 * df["gamma3"]
    
    result["phi_1"] = phi_1
    result["phi_2"] = phi_2
    
    result["d_z_1"] = df["l1"] * np.cos(phi_1)
    result["d_xi_1"] = df["l1"] * np.sin(phi_1)
    
    result["d_z_2"] = df["l2"] * np.cos(phi_2)
    result["d_xi_2"] = df["l2"] * np.sin(phi_2)
    return result

# %%
mat_data["helixCoordinates"].shape

# %%
mat_data["centerlineNodes"].shape

# %%
mat_data["centerlineCoordinates"].shape

# %%
mat_data["centerlineCoordinates"][-1]

# %%
mat_data

# %%
df = pd.DataFrame(mat, columns=["l1", "l2", "theta", "gamma1", "gamma2", "gamma3"])
# feature_df = make_feature_df(transform_z_xi(preprocess(df, inplace=False)))
df = transform_z_xi(preprocess(df))
df


# %%
def make_feature_df(douglas_df):
    feature_df = pd.DataFrame(columns=["d_xi_1", "d_xi_1^2", "d_xi_2", "d_xi_2^2", "d_z_1^2", "d_z_2^2", "gamma1^2", "gamma3^2"])
    feature_df["d_xi_1"] = douglas_df["d_xi_1"]
    feature_df["d_xi_1^2"] = douglas_df["d_xi_1"] ** 2
    feature_df["d_xi_2"] = douglas_df["d_xi_2"]
    feature_df["d_xi_2^2"] = douglas_df["d_xi_2"] ** 2
    feature_df["d_z_1^2"] = douglas_df["d_z_1"] ** 2
    feature_df["d_z_2^2"] = douglas_df["d_z_2"] ** 2
    feature_df["gamma1^2"] = douglas_df["gamma1"] ** 2
    feature_df["gamma3^2"] = douglas_df["gamma3"] ** 2
    return feature_df


# %%
df_list = []
slinky_idx = 0
for mat_file in glob("data/RAWDATA#ANGLENEW#CHANGE#RELATIVE/*.mat"):
    mat_data = scio.loadmat(mat_file)
    for mat, energy, total_energy in zip(
        mat_data["NNInput_All"].transpose(2,1,0),
        np.squeeze(mat_data["NNOutput_All"].transpose(2,1,0)),
        mat_data["circleEnergy"].sum(axis=1)
      ):
        df = pd.DataFrame(mat, columns=["l1", "l2", "theta", "gamma1", "gamma2", "gamma3"])
        # feature_df = make_feature_df(transform_z_xi(preprocess(df, inplace=False)))
        df = transform_z_xi(preprocess(df))
        df["energy"] = energy
        df["total_energy"] = total_energy
        df["slinky_idx"] = slinky_idx
        df["triplet_idx"] = np.arange(mat.shape[0])
        slinky_idx += 1
        df["comment"] = os.path.basename(mat_file)
        df_list.append(df)
total_df = pd.concat(df_list)

# %%
total_df.reset_index().to_feather("data/aggregated_slinky_triplet_energy.feather")

# %%
sns.pairplot(data=total_df.drop(["comment"], axis=1), diag_kind="hist", diag_kws={"bins": 100}, plot_kws={"s": 1, "edgecolor": "none"})

# %%
SlinkyData_Angle_6_new_Large_Gravity = scio.loadmat("data/RAWDATA#ANGLENEW#CHANGE#RELATIVE/SlinkyData_Angle_6_new_Large_Gravity.mat")

# %%
np.all(SlinkyData_Angle_6_new_Large_Gravity["circleEnergy"][:,1:-1].reshape(-1,1) == energy)

# %%
NNInput_All_reshape = SlinkyData_Angle_6_new_Large_Gravity["NNInput_All_reshape"].T

# %%
NNInput_All_reshape.shape

# %%
energy = SlinkyData_Angle_6_new_Large_Gravity["NNOutput_All_reshape"].T

# %%
df = pd.DataFrame(NNInput_All_reshape, columns=["l1", "l2", "theta", "gamma1", "gamma2", "gamma3"])
sns.pairplot(df, diag_kind="hist")

# %%
sns.pairplot(preprocess(df, inplace=False), diag_kind="hist")

# %%
sns.pairplot(douglas_df, diag_kind="hist", plot_kws={"edgecolor": "none", "size": 1})

# %%
sns.pairplot(douglas_df, diag_kind="hist", plot_kws={"edgecolor": "none", "size": 1})

# %%
sns.pairplot(douglas_df[["theta", "phi_1", "phi_2"]], diag_kind="hist", plot_kws={"edgecolor": "none", "size": 1})

# %%
sns.pairplot(douglas_df[["theta", "phi_1", "phi_2"]], diag_kind="hist", plot_kws={"edgecolor": "none", "size": 1})

# %%
df = pd.DataFrame(NNInput_All_reshape, columns=["l1", "l2", "theta", "gamma1", "gamma2", "gamma3"])
feature_df = make_feature_df(transform_z_xi(preprocess(df, inplace=False)))

# %%
SlinkyData_Angle_6_new_Large_Gravity["NNInput_All"].transpose(2,1,0).shape

# %%
feature_df.columns

# %%
feature_df = make_feature_df(total_df)
douglas_df = pd.DataFrame(np.concatenate([
    feature_df[["d_xi_1", "d_xi_1^2", "d_z_1^2", "gamma1^2"]].to_numpy(),
    feature_df[["d_xi_2", "d_xi_2^2", "d_z_2^2", "gamma1^2"]].to_numpy(),
]), columns = ["d_xi", "d_xi^2", "d_z^2", "gamma^2"]).drop_duplicates()

# %%
douglas_df["energy"] = lr.predict(douglas_df)

# %%
lr.coef_

# %%
list(lr.coef_)

# %%
import torch


# %%
class DouglasModelTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cycle_triplet_coord):
        """
        Args:
            cycle_triplet_coord: input tensor of shape (N x 6). The columns are
                (l1, l2, theta, gamma1, gamma2, gamma3)
        Returns:
            Douglas model's degrees of freedom of shape (N x 2 x 3).
            The last axis is (dxi, dz, dphi) where dphi is gamma1 or gamma3
        """
        l1 = cycle_triplet_coord[..., 0]
        l2 = cycle_triplet_coord[..., 1]
        theta = cycle_triplet_coord[..., 2]
        gamma1 = cycle_triplet_coord[..., 3]
        gamma2 = cycle_triplet_coord[..., 4]
        gamma3 = cycle_triplet_coord[..., 5]

#         psi_1 = 0.5 * (np.pi - theta) - gamma2 - 0.5 * gamma1
#         psi_2 = 0.5 * (np.pi + theta) - gamma2 + 0.5 * gamma3
        psi_1 = 0.5 * (np.pi - theta) + gamma2 - 0.5 * gamma1
        psi_2 = 0.5 * (np.pi + theta) + gamma2 - 0.5 * gamma3

        dz_1 = l1 * torch.cos(psi_1)
        dxi_1 = l1 * torch.sin(psi_1)

        dz_2 = l2 * torch.cos(psi_2)
        dxi_2 = l2 * torch.sin(psi_2)
        
        first_pair_dof = torch.stack([dxi_1, dz_1, gamma1], dim=-1) # (N x 3)
        second_pair_dof = torch.stack([dxi_2, dz_2, -gamma3], dim=-1) # (N x 3)
        return torch.stack([first_pair_dof, second_pair_dof], dim=1) # (N x 2 x 3)

class DouglasModel(torch.nn.Module):
    def __init__(self, c_dxi, c_dxi_sq, c_dz_sq, c_dphi_sq):
        super().__init__()
        self.c_dxi = c_dxi
        self.c_dxi_sq = c_dxi_sq
        self.c_dz_sq = c_dz_sq
        self.c_dphi_sq = c_dphi_sq

    def forward(self, douglas_dof):
        """
        Args:
            douglas_dof: (N x 3) tensor. The columns are (dxi, dz, dphi)
        """
        dxi = douglas_dof[..., 0]
        dz = douglas_dof[..., 1]
        dphi = douglas_dof[..., 2]
        
        return self.c_dxi * dxi + self.c_dxi_sq * dxi ** 2 + self.c_dz_sq * dz ** 2 + self.c_dphi_sq * dphi ** 2


# %%
triplet_input = torch.from_numpy(total_df[['l1', 'l2', 'theta', 'gamma1', 'gamma2', 'gamma3']].to_numpy()).requires_grad_()

douglas_dof = DouglasModelTransform()(triplet_input)
douglas_model = DouglasModel(*lr.coef_)
douglas_energy = douglas_model(douglas_dof)

douglas_force = torch.autograd.grad(douglas_energy, triplet_input, torch.ones_like(douglas_energy))[0]

# torch.save({
#     "triplet_coords": triplet_input,
#     "douglas_dof": 
#     "douglas_energy": doug
# })

# %%
feature_df = make_feature_df(total_df)

# %%
feature_df["slinky_idx"] = total_df["slinky_idx"]
feature_df["triplet_idx"] = total_df["triplet_idx"]

# %%
total_df

# %%
feature_df

# %% [markdown]
# # triplet coord to `xz` (or `xy` if you like)

# %% [markdown]
# ![image.png](attachment:image.png)

# %%
signed_angle_2d(torch.tensor([1,0]), torch.tensor([1,1]))

# %%
from typing import List
def transform_triplet_to_cartesian(cycle_triplet_coord: torch.Tensor, bar_lengths: List[float]):
    """
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
    
    center_1 = torch.stack([l1.detach()-l1, torch.zeros_like(l1)], dim=-1)
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
    return torch.stack([
        top_1, center_1, bottom_1,
        top_2, center_2, bottom_2,
        top_3, center_3, bottom_3
    ], dim=-2)


# %%
def cross2d(tensor_1: torch.Tensor, tensor_2: torch.Tensor):
    assert tensor_1.shape[-1] == 2
    assert tensor_2.shape[-1] == 2
    device = tensor_1.device
    dtype = tensor_1.dtype
    cross_mat = torch.tensor([[0, 1], [-1, 0]], device=device, dtype=dtype)
    return torch.einsum("...i,ij,...j", tensor_1, cross_mat, tensor_2)

def dot(tensor_1: torch.Tensor, tensor_2: torch.Tensor, keepdim=False):
    return torch.sum(tensor_1 * tensor_2, dim = -1, keepdim=keepdim)

def signed_angle_2d(v1: torch.Tensor, v2: torch.Tensor):
    return torch.atan2(cross2d(v1, v2), dot(v1,v2, keepdim=False))

def compute_bisector(v1: torch.Tensor, v2: torch.Tensor):
    def _compute_bisector(v1: torch.Tensor, v2: torch.Tensor):
        norm_1 = v1.norm(dim=-1, keepdim=True)
        norm_2 = v2.norm(dim=-1, keepdim=True)
        norm = (norm_1 * norm_2).detach()
        
        return norm_1 * v2 / norm  + norm_2 * v1 / norm

    def _rotate90(v: torch.Tensor):
        return torch.stack([-v[...,1], v[...,0]], dim=-1)

    bisec_1 = _compute_bisector(v1, v2)
    bisec_2 = _rotate90(_compute_bisector(v1, -v2))
    bisec_3 = -bisec_1

    case = (dot(v1, v2) < 0).long() + ((dot(v1, v2) >= 0) & (cross2d(v1, v2) < 0))*2 # 0, 1 or 2
#     print(case)
    case = case[..., None]
    case = case.expand_as(bisec_1)
    case = case[None, :]

    return torch.gather(torch.stack([bisec_1, bisec_2, bisec_3], dim=0), 0, case).squeeze()

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

    l1 = (center_2 - center_1)
    l2 = (center_3 - center_2)
    theta = signed_angle_2d(l2, l1)

    # TODO: make use of the bottom for real data?
    bar_1 = top_1 - center_1
    bar_2 = top_2 - center_2
    bar_3 = top_3 - center_3

    gamma_1 = signed_angle_2d(bar_1, bar_2)
    gamma_3 = signed_angle_2d(bar_3, bar_2)
    
#     print(compute_bisector(l1, -l2))
    
    gamma_2 = signed_angle_2d(compute_bisector(l1, -l2), bar_2)
    return torch.stack([l1.norm(dim=-1), l2.norm(dim=-1), theta, gamma_1, gamma_2, gamma_3], dim=-1)

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

    l1 = (center_2 - center_1)
    l2 = (center_3 - center_2)
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
    
    first_pair_dof = torch.stack([dxi_1, dz_1, gamma_1], dim=-1) # (... x 3)
    second_pair_dof = torch.stack([dxi_2, dz_2, -gamma_3], dim=-1) # (... x 3)
    return torch.stack([first_pair_dof, second_pair_dof], dim=-2) # (... x 2 x 3)


# %%
from typing import List
def transform_triplet_to_cartesian_alpha(cycle_triplet_coord: torch.Tensor):
    """
    Args:
        cycle_triplet_coord: (... x 6) shape. The names of the last axis are
            (l1, l2, theta, gamma1, gamma2, gamma3)
    Returns:
        Cartesian coordinates of all nodes in the 2D representation. Use the
        center node of the first cycle in the triplet as the origin, and the
        first link as the x axis.
        In total there are 9 nodes, so the output shape is (... x 2 x 2 x 3),
        in the order of (center_x, center_z, alpha)
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

    center_1 = torch.stack([l1.detach()-l1, torch.zeros_like(l1)], dim=-1)
    center_2 = torch.stack([l1, torch.zeros_like(l1)], dim=-1)
    center_3 = center_2.clone()
    center_3[..., 0] += l2 * torch.cos(theta)
    center_3[..., 1] -= l2 * torch.sin(theta)

    first_pair = torch.stack([
        torch.cat([center_1, alpha1[...,None]], dim=-1),
        torch.cat([center_2, alpha2[...,None]], dim=-1)
    ], dim=-2)
    second_pair = torch.stack([
        torch.cat([center_2, (alpha2+theta)[...,None]], dim=-1),
        torch.cat([center_3, (alpha3+theta)[...,None]], dim=-1)
    ], dim=-2)
    
    return torch.stack([first_pair, second_pair], dim=-3)

def transform_cartesian_alpha_to_douglas(cartesian_alpha_input: torch.Tensor):
    """
    Args:
        cartesian_alpha_input: input tensor of shape (... x 2 x 3). The columns are
            (center_x, center_z, alpha)
    Returns:
        Douglas model's degrees of freedom of shape (... x 3).
        The last axis is (dxi, dz, dphi)
    """
    alpha_1 = cartesian_alpha_input[..., 0, 2]
    alpha_2 = cartesian_alpha_input[..., 1, 2]
    l = (cartesian_alpha_input[..., 1, :2] - cartesian_alpha_input[..., 0, :2]).norm(dim=-1)
    psi = 0.5 * (alpha_1 + alpha_2)
    dxi = l * torch.cos(psi)
    dz = l * torch.sin(psi)
    dphi = alpha_2 - alpha_1
    
    return torch.stack([dxi, dz, dphi], dim=-1) # (... x 3)


# %%
douglas_1 = transform_cartesian_alpha_to_douglas(transform_triplet_to_cartesian_alpha(triplet_input))

# %%
douglas_2 = transform_cartesian_to_douglas(transform_triplet_to_cartesian(triplet_input, bar_lengths))

# %%
douglas_1

# %%
douglas_2

# %%
cartesian_coords = transform_triplet_to_cartesian(triplet_input, bar_lengths)

# %%
v1 = torch.zeros(100,2)
v1[:,0] = 1
theta = torch.linspace(0, 2*np.pi, 100)
v2 = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

# %%
bisec_1 = compute_bisector(v1, v2)
bisec_2 = compute_bisector(v2, v1)

# %%
half_theta_1 = signed_angle_2d(v1, bisec_1)
half_theta_2 = signed_angle_2d(v1, bisec_2)

# %%
bisec_1 = bisec_1 / bisec_1.norm(dim=-1, keepdim=True)
plt.scatter(bisec_1[:,0], bisec_1[:,1], c=theta)

# %%
bisec_2 = bisec_2 / bisec_2.norm(dim=-1, keepdim=True)
plt.scatter(bisec_2[:,0], bisec_2[:,1], c=theta)

# %%
plt.plot(theta, half_theta_1)

# %%
plt.plot(theta, half_theta_2)

# %%
triplet_result = transform_cartesian_to_triplet(cartesian_coords)

# %%
triplet_result

# %%
torch.allclose(triplet_input, triplet_result)

# %%
feature_df["energy_1_pytorch"] = douglas_energy[:,0].detach().numpy()
feature_df["energy_2_pytorch"] = douglas_energy[:,1].detach().numpy()

# %%
feature_df["energy_1"] = lr.predict(feature_df[["d_xi_1", "d_xi_1^2", "d_z_1^2", "gamma1^2"]]) - lr.intercept_
feature_df["energy_2"] = lr.predict(feature_df[["d_xi_2", "d_xi_2^2", "d_z_2^2", "gamma3^2"]]) - lr.intercept_

# %%
douglas_dof[:,0,:]**2

# %%
feature_df[["d_xi_1", "d_xi_1^2", "d_z_1^2", "gamma1^2"]]

# %%
feature_df

# %%
douglas_predict = feature_df.groupby("slinky_idx").agg({"energy_1_pytorch": "sum"}).rename({"energy_1_pytorch": "energy_1"}, axis=1)
douglas_predict = douglas_predict + pd.merge(
    feature_df, feature_df.groupby("slinky_idx").agg({"triplet_idx": "max"}).reset_index(),
    on=["slinky_idx", "triplet_idx"])[["energy_2"]].rename({"energy_2": "energy_1"}, axis=1)

# %%
original_df = pd.read_feather("data/aggregated_slinky_triplet_energy.feather")

# %%
group_df.groupby("comment").size().sort_values()[-10:].index

# %%
fig, ax = plt.subplots(figsize=(20,20))
group_df = total_df.groupby(["slinky_idx", "comment"]).agg({"total_energy": "first"}).reset_index()
group_df["douglas_predict"] = douglas_predict["energy_1"]
group_df = group_df[group_df.comment.isin(['SlinkyData_Angle_6_new_Small_Damping.mat',
       'SlinkyData_Angle_6_new_Resample_Small_Gravity.mat',
       'SlinkyData_Angle_6_new_Small_Gravity.mat',
       'SlinkyData_Angle_6_new_Dropping.mat',
       'SlinkyData_Angle_6_new_Resample_Dropping.mat',
       'SlinkyData_Angle_6_new_Large_Damping.mat',
       'SlinkyData_Angle_6_new_Resample_Large_Damping.mat',
       'SlinkyData_Angle_6_new_Stretching_Shear.mat',
       'SlinkyData_Angle_6_new_Shear.mat',
       'SlinkyData_Angle_6_new_Imbalanced_Stretching.mat'])]
plt.plot(np.linspace(0,1,100), np.linspace(0,1,100), linestyle="dotted", color="k")
sp = sns.scatterplot(data=group_df, x="total_energy", y="douglas_predict", hue="comment", edgecolor="none", palette="tab10", s=1, alpha=1)
sp.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# %%
fig, ax = plt.subplots(figsize=(20,20))
group_df = total_df.groupby(["slinky_idx", "comment"]).agg({"total_energy": "first"}).reset_index()
group_df["douglas_predict"] = douglas_predict["energy_1"]
plt.plot(np.linspace(0,1,100), np.linspace(0,1,100), linestyle="dotted", color="k")
sp = sns.scatterplot(data=group_df, x="total_energy", y="douglas_predict", hue="comment", edgecolor="none", palette="deep", alpha=0.05)
plt.xscale("log")
plt.yscale("log")
sp.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# %%
tmp = group_df.copy().query("total_energy > 0.1")
tmp["ratio"] = tmp["douglas_predict"] / tmp["total_energy"]
sns.catplot(data=tmp, y="comment", x="ratio", kind="box", aspect=3)

# %%
(group_df["total_energy"] < 0).any()

# %%
group_df.iloc[(group_df["energy"] > 1).to_numpy() & (douglas_predict < 0.8).to_numpy().ravel()]

# %%
grouped_df_list = []
for single_slinky_data in SlinkyData_Angle_6_new_Large_Gravity["NNInput_All"].transpose(2,1,0):
    df = pd.DataFrame(single_slinky_data, columns=["l1", "l2", "theta", "gamma1", "gamma2", "gamma3"])
    feature_df = make_feature_df(transform_z_xi(preprocess(df, inplace=False)))
    df_ = pd.DataFrame(feature_df.sum(axis=0)).T
    df_ = df_[['d_xi_1', 'd_xi_1^2', 'd_z_1^2', 'gamma1^2']].copy()
#     print(df_)
    df_.loc[0, ['d_xi_1', 'd_xi_1^2', 'd_z_1^2', 'gamma1^2']] += feature_df.iloc[-1][['d_xi_2', 'd_xi_2^2', 'd_z_2^2', 'gamma3^2']].to_numpy()
#     print(feature_df.iloc[-1][['d_xi_2', 'd_xi_2^2', 'd_z_2^2', 'gamma3^2']])
    grouped_df_list.append(df_)

# %%
grouped_df = pd.concat(grouped_df_list)

# %%
from sklearn import linear_model

# %%
lr = linear_model.LinearRegression(positive=True)

# %%
grouped_df

# %%
grouped_energy = SlinkyData_Angle_6_new_Large_Gravity["circleEnergy"].sum(axis=1)
lr.fit(grouped_df, grouped_energy)

# %%
lr.intercept_

# %%
pd.DataFrame(lr.coef_[None,:], columns=list(grouped_df.columns))

# %%

# %%
fig, axs = plt.subplots(1,2,figsize=(12,6))
axs[0].scatter(grouped_energy, lr.predict(grouped_df), s=1)
axs[0].plot(grouped_energy, grouped_energy, linestyle="dotted", color="k")
axs[0].set_xlabel("Single slinky total energy (80 cycles)")
axs[0].set_ylabel("Polynomial prediction")
axs[0].grid()
axs[0].set_title("linear scale")

axs[1].scatter(grouped_energy, lr.predict(grouped_df), s=1)
axs[1].plot(grouped_energy, grouped_energy, linestyle="dotted", color="k")
axs[1].set_xlabel("Single slinky total energy (80 cycles)")
axs[1].set_ylabel("Polynomial prediction")
axs[1].grid()
axs[1].set_xscale("log")
axs[1].set_yscale("log")
axs[1].set_title("log scale")
plt.suptitle("Douglas Polynomial Energy Model")
plt.show()
plt.close()

# %%
lr.fit(feature_df, energy)

# %%
pd.DataFrame(lr.coef_, columns=feature_df.columns)

# %%
df = pd.DataFrame(NNInput_All_reshape, columns=["l1", "l2", "theta", "gamma1", "gamma2", "gamma3"])
feature_df = make_feature_df(transform_z_xi(preprocess(df, inplace=False)))

lr.fit(feature_df, energy)
plt.scatter(energy, lr.predict(feature_df), s=1)
plt.plot(energy, energy, linestyle="dotted", color="k")

# %%
plt.scatter(energy, lr.predict(feature_df), s=1)
# plt.plot(energy, energy, linestyle="dotted", color="k")

# %%
