from typing import Dict
import sys

import torch
from torch import optim
from torch.random import seed
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

import wandb

# from neural_slinky import e3nn_models
from neural_slinky import emlp_models

# from data import douglas_data

device = "cuda"

from neural_slinky.utils import get_model_device, seed_torch, plot_regression_scatter

seed_torch(42)

# key = "cartesian_alpha_single"
key = "cartesian_alpha"
# key = "cartesian"
batch_size = 1024
num_epochs = 100
lr = 1e-2
weight_decay = 0

# * Load Douglas data
douglas_dataset_dict: Dict[str, Dict[str, torch.Tensor]] = torch.load(
    "data/douglas_dataset.pt"
)

column_perm_inds = [2, 5, 8, 0, 1, 3, 4, 6, 7]
input_tensor = (
    douglas_dataset_dict["coords"][key].float().clone().detach().requires_grad_(False)
)[..., column_perm_inds]
output_tensor = (
    douglas_dataset_dict["force"][key].float().clone().detach().requires_grad_(False)
)[..., column_perm_inds]

print("input shape:", input_tensor.shape)
print("output shape:", output_tensor.shape)

train_val_input, test_input, train_val_output, test_output = train_test_split(
    input_tensor, output_tensor, test_size=0.2
)
train_input, val_input, train_output, val_output = train_test_split(
    train_val_input, train_val_output, test_size=0.2
)

# TODO: do random rotation for augmentation

train_dataset = TensorDataset(train_input, train_output)
val_dataset = TensorDataset(val_input, val_output)
test_dataset = TensorDataset(test_input, test_output)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


model = emlp_models.SlinkyForcePredictorEMLP(layers=3)
model.to(device)
print(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def evaluate(model, dataloader):
    df_list = []
    sample_ratio = 0.1
    pos_name_list = ["x_1", "z_1", "x_2", "z_2", "x_3", "z_3"]
    rot_name_list = ["a_1", "a_2", "a_3"]
    with torch.no_grad():
        for (cartesian_alpha, force) in dataloader:
            cartesian_alpha = cartesian_alpha.to(device)
            force = force.to(device)
            pos_force, rot_force = force[..., 3:], force[..., :3]
            force_pred = model(cartesian_alpha)
            pos_force_mse = (pos_force - force_pred[..., 3:]).norm(dim=-1).mean()
            rot_force_mse = (rot_force - force_pred[..., :3]).norm(dim=-1).mean()
            mse = pos_force_mse + rot_force_mse
            force = force.cpu().numpy()
            force_pred = force_pred.cpu().numpy()
            dict_for_df = {}
            for i, name in enumerate(rot_name_list + pos_name_list):
                dict_for_df[f"{name}_truth"] = force[:, i]
                dict_for_df[f"{name}_pred"] = force_pred[:, i]
            df = pd.DataFrame(dict_for_df).sample(frac=sample_ratio)
            df_list.append(df)
    df = pd.concat(df_list)
    wandb.log(
        {
            "test/pos_force_mse": pos_force_mse,
            "test/rot_force_mse": rot_force_mse,
            "test/mse": mse,
        }
    )

    wandb.log(
        {
            f"test/residual_min/{name}": np.min(
                df[f"{name}_pred"] - df[f"{name}_truth"]
            )
            for name in pos_name_list + rot_name_list
        }
    )

    wandb.log(
        {
            f"test/residual_max/{name}": np.max(
                df[f"{name}_pred"] - df[f"{name}_truth"]
            )
            for name in pos_name_list + rot_name_list
        }
    )
    wandb.log(
        {
            f"test/scatters/{name}": plot_regression_scatter(
                df, f"{name}_truth", f"{name}_pred", name
            )
            for name in pos_name_list + rot_name_list
        }
    )


import argparse

parser = argparse.ArgumentParser(
    "Pretraining a force predictor with Douglas synthetic data"
)
parser.add_argument("--log", action="store_true", dest="log")
parser.add_argument("--no-log", action="store_false", dest="log")
args = parser.parse_args()
if args.log:
    wandb.init(project="Slinky-pretrain", config=args)
else:
    wandb.init(mode="disabled")

rot_mse_scale = 10

for epoch in range(num_epochs):
    print(f"epoch {epoch}")
    for (
        cartesian_alpha,
        force,
    ) in train_dataloader:  # cartesian_alpha.shape == (batch_size, 6)
        cartesian_alpha = cartesian_alpha.to(device)
        force = force.to(device)
        pos_force, rot_force = force[..., 3:], force[..., :3]
        # force = force.to(device)

        with torch.enable_grad():
            # output = model(data)
            force_pred = model(cartesian_alpha)
            # mse = (force.view(-1,2) - output[:,0:2]).norm(dim=0)
            rot_force_mse = (rot_force - force_pred[..., :3]).norm(dim=0).mean()
            pos_force_mse = (pos_force - force_pred[..., 3:]).norm(dim=0).mean()
            loss = pos_force_mse + rot_mse_scale * rot_force_mse
            mse = pos_force_mse + rot_force_mse
            wandb.log(
                {
                    "pos_force_mse": pos_force_mse,
                    "rot_force_mse": rot_force_mse,
                    "mse": mse,
                }
            )

        sys.stdout.write(f"\rBatch loss: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    if epoch % 1 == 0:
        evaluate(model, test_dataloader)

    torch.save({"state_dict": model.state_dict(), "epoch": epoch}, "e3nn_checkpoint.pt")
    print()
