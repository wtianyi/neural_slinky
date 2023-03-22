# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: 'Python 3.7.4 64-bit (''py37'': conda)'
#     language: python
#     name: python37464bitpy37conda4d64983d62ce41e489a3b689c2654284
# ---

# %%
import torch
import numpy as np
import pandas as pd
from neural_slinky import coords_transform

# %%
from neural_slinky import utils

# %%
from neural_slinky import models

# %%
from neural_slinky import douglas_models

# %%
douglas_models.DouglasModel

# %%
# * Import triplet df
df = pd.read_feather("data/aggregated_slinky_triplet_energy.feather")
print(df.head())

# * Transform to cartesian
triplet_input = torch.from_numpy(
    df[["l1", "l2", "theta", "gamma1", "gamma2", "gamma3"]].to_numpy()
)  # .requires_grad_()

cartesian_alpha_coords = coords_transform.transform_triplet_to_cartesian_alpha(
    triplet_input
)

cartesian_alpha_coords_single = coords_transform.transform_triplet_to_cartesian_alpha_single(
    triplet_input
)

cartesian_coords = coords_transform.transform_triplet_to_cartesian(
    triplet_input, [0.1, 0.1, 0.1]
)

# * Generate force by autograd
douglas_model = douglas_models.DouglasModel(
    c_dxi=0.04111493612707057,
    c_dxi_sq=24.626752530556853,
    c_dz_sq=77.00113656812572,
    c_dphi_sq=0.036598450117485276,
)

# ** For (x, z, alpha) coord
cartesian_alpha_coords.requires_grad_(True)
douglas_coords_alpha = coords_transform.transform_cartesian_alpha_to_douglas(
    cartesian_alpha_coords
)
douglas_energy_alpha = douglas_model(douglas_coords_alpha)
douglas_force_alpha = torch.autograd.grad(
    douglas_energy_alpha,
    cartesian_alpha_coords,
    torch.ones_like(douglas_energy_alpha),
)[0]

# *** For single pairs
cartesian_alpha_coords_single.requires_grad_(True)
douglas_coords_alpha_single = coords_transform.transform_cartesian_alpha_to_douglas_single(
    cartesian_alpha_coords_single
)
douglas_energy_alpha_single = douglas_model(douglas_coords_alpha_single)
douglas_force_alpha_single = torch.autograd.grad(
    douglas_energy_alpha_single,
    cartesian_alpha_coords_single,
    torch.ones_like(douglas_energy_alpha_single),
)[0]

# ** For (x, z) coord
cartesian_coords.requires_grad_(True)
douglas_coords = coords_transform.transform_cartesian_to_douglas(cartesian_coords)
douglas_energy = douglas_model(douglas_coords)
douglas_force = torch.autograd.grad(
    douglas_energy, cartesian_coords, torch.ones_like(douglas_energy)
)[0]

# # * Save douglas data
# douglas_dataset = {
#     "coords": {
#         "cartesian": cartesian_coords,
#         "cartesian_alpha": cartesian_alpha_coords,
#         "cartesian_alpha_single": cartesian_alpha_coords_single.flatten(start_dim=-2),
#     },
#     "force": {
#         "cartesian": douglas_force,
#         "cartesian_alpha": douglas_force_alpha,
#         "cartesian_alpha_single": douglas_force_alpha_single.flatten(start_dim=-2),
#     },
# }

# %%
import seaborn as sns

# %%
sns.pairplot(df[["l1", "l2", "theta", "gamma1", "gamma2", "gamma3"]].sample(frac=0.01), diag_kind="hist", diag_kws={"bins": 50}, plot_kws={"alpha": 0.3, "s": 3}, height=5)
