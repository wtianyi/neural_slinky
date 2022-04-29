from typing import Dict
import sys
from sklearn import neural_network

import torch
from torch import optim
from torch.random import seed
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from tqdm.auto import tqdm

import wandb

# from neural_slinky import e3nn_models
from neural_slinky import models as ns_models
from neural_slinky import coords_transform

# from data import douglas_data

device = "cuda"

from neural_slinky.utils import (
    AverageMeter,
    get_model_device,
    num_parameters,
    seed_torch,
    plot_regression_scatter,
)

import argparse

parser = argparse.ArgumentParser(
    "Pretraining a force predictor with Douglas synthetic data"
)
parser.add_argument("--log", action="store_true", dest="log")
parser.add_argument("--no-log", action="store_false", dest="log")
args = parser.parse_args()
if args.log:
    wandb.init(project="Slinky-pretrain-ours", config=args)
else:
    wandb.init(mode="disabled")

rot_mse_scale = 10

seed_torch(42)

# key = "cartesian_alpha_single"
key = "cartesian_alpha"
# key = "cartesian"
batch_size = 1024
num_epochs = 100
lr = 1e-3
weight_decay = 0

# * Load Douglas data
douglas_dataset_dict: Dict[str, Dict[str, torch.Tensor]] = torch.load(
    "data/douglas_dataset.pt"
)

# column_perm_inds = [0, 1, 2, 3, 4, 5, 6, 7, 8]
input_tensor = (
    douglas_dataset_dict["coords"][key].float().clone().detach().requires_grad_(False)
)
output_tensor = (
    douglas_dataset_dict["force"][key].float().clone().detach().requires_grad_(False)
)

pos_inds = [0, 1, 3, 4, 6, 7]
rot_inds = [2, 5, 8]

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

neurons_per_layer = 32
num_layers = 5


class BackboneModel(torch.nn.Module):
    def __init__(self):
        super(BackboneModel, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(6, neurons_per_layer),  # snn.Square(),
            ns_models.DenseBlock(neurons_per_layer, num_layers),
            ns_models.Square(),
            # snn.AbsLinear(int(self.neuronsPerLayer*(self.numLayers+1)),1)
            torch.nn.Linear(int(neurons_per_layer * (num_layers + 1)), 1),
        )

    def forward(self, cartesian_alpha):
        douglas_data = coords_transform.transform_cartesian_alpha_to_douglas_v2(
            cartesian_alpha
        ).flatten(start_dim=-2)
        return self.net(douglas_data)


backbone_model = BackboneModel()
model = ns_models.AutogradOutputWrapper(
    ns_models.DouglasChiralInvariantWrapper(backbone_model)
)

print(model)
print("Number of parameters:", num_parameters(model))

model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, factor=0.1, patience=2
)


def evaluate(model, dataloader):
    df_list = []
    sample_ratio = 0.01
    pos_name_list = ["x_1", "z_1", "x_2", "z_2", "x_3", "z_3"]
    rot_name_list = ["a_1", "a_2", "a_3"]
    mse_meter = AverageMeter()
    pos_force_mse_meter = AverageMeter()
    rot_force_mse_meter = AverageMeter()
    with torch.no_grad():
        for (cartesian_alpha, force) in dataloader:
            cartesian_alpha = cartesian_alpha.to(device)
            force = force.to(device)
            pos_force, rot_force = force[..., pos_inds], force[..., rot_inds]
            force_pred = model(cartesian_alpha)
            pos_force_mse = (pos_force - force_pred[..., pos_inds]).norm(dim=-1).mean()
            rot_force_mse = (rot_force - force_pred[..., rot_inds]).norm(dim=-1).mean()
            mse = pos_force_mse + rot_force_mse

            n_data = force.shape[0]
            pos_force_mse_meter.update(pos_force_mse.item(), n=n_data)
            rot_force_mse_meter.update(rot_force_mse.item(), n=n_data)
            mse_meter.update(mse.item(), n=n_data)

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
            "test/pos_force_mse": pos_force_mse_meter.val,
            "test/rot_force_mse": rot_force_mse_meter.val,
            "test/mse": mse_meter.val,
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

from neural_slinky import priority_memory
pr_buffer = priority_memory.PrioritizedReplayBuffer(size=100000, alpha=0.1, beta=0.1)
pr_scheduler = priority_memory.PrioritizedReplayBetaScheduler(pr_buf=pr_buffer, T_max=10)

for epoch in tqdm(range(num_epochs)):
    mse_meter = AverageMeter()
    print(f"epoch {epoch}")
    for (
        cartesian_alpha,
        force,
    ) in tqdm(train_dataloader):  # cartesian_alpha.shape == (batch_size, 6)
        cartesian_alpha_cuda = cartesian_alpha.to(device)
        force_cuda = force.to(device)
        pos_force, rot_force = force_cuda[..., pos_inds], force_cuda[..., rot_inds]
        # force = force.to(device)
        n_data = cartesian_alpha_cuda.shape[0]

        with torch.enable_grad():
            # output = model(data)
            force_pred = model(cartesian_alpha_cuda)
            # mse = (force.view(-1,2) - output[:,0:2]).norm(dim=0)
            rot_force_mse = ((rot_force - force_pred[..., rot_inds]) ** 2).mean(dim=1)
            pos_force_mse = ((pos_force - force_pred[..., pos_inds]) ** 2).mean(dim=1)
            loss = pos_force_mse + rot_mse_scale * rot_force_mse
            mse = pos_force_mse + rot_force_mse
            pr_buffer
            mse_meter.update(mse.item(), n_data)
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        optimizer.step()

    if epoch % 1 == 0:
        evaluate(model, test_dataloader)
    scheduler.step(mse_meter.val)

    torch.save({"state_dict": model.state_dict(), "epoch": epoch}, "ours_checkpoint.pt")
    print()
