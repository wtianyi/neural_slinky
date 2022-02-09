from typing import Dict
import sys

import torch
from torch import optim
from torch.random import seed
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import wandb
from neural_slinky import e3nn_models
from data import douglas_data

device = "cuda"

from neural_slinky.utils import seed_torch, AverageMeter, plot_regression_scatter

seed_torch(42)

# key = "cartesian_alpha_single"
key = "cartesian"
batch_size = 128
num_epochs = 100
lr = 1e-2
weight_decay = 0

# * Load Douglas data
douglas_dataset_dict: Dict[str, Dict[str, torch.Tensor]] = torch.load(
    "data/douglas_dataset.pt"
)

input_tensor = (
    douglas_dataset_dict["coords"][key].float().clone().detach().requires_grad_(False)
)
output_tensor = (
    douglas_dataset_dict["force"][key].float().clone().detach().requires_grad_(False)
)

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


model = e3nn_models.SlinkyForcePredictorCartesian(
    irreps_node_input="0e",
    irreps_node_attr="2x0e",
    irreps_edge_attr="2x0e",
    irreps_node_output="1x1o",
    max_radius=0.06,
    num_neighbors=1,
    num_nodes=2,
    mul=50,
    layers=3,
    lmax=5,
    pool_nodes=False,
)
model.to(device)
print(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def make_data(cartesian: torch.Tensor):
    num_triplets = cartesian.shape[0]
    batch = (
        torch.arange(num_triplets, device=device)
        .view(-1, 1)
        .repeat((1, 9))
        .flatten()
        .long()
    )
    edge_index = torch.tensor(
        [[1, 0], [1, 2], [4, 3], [4, 5], [7, 6], [7, 8], [1, 4], [4, 7]], device=device
    ).repeat((num_triplets, 1)) + torch.arange(num_triplets, device=device).mul_(
        9
    ).repeat_interleave(
        8
    ).view(
        -1, 1
    )
    edge_index = edge_index.long()

    edge_attr = torch.nn.functional.one_hot(
        torch.tensor([1, 1, 1, 1, 1, 1, 0, 0], device=device).repeat(num_triplets),
        num_classes=2,
    ).float()

    node_attr = torch.nn.functional.one_hot(
        torch.tensor([1, 0, 1], device=device).repeat(3 * num_triplets), num_classes=2
    ).float()

    cartesian = cartesian.reshape(-1, 2)
    node_input = cartesian.new_ones(cartesian.shape[0], 1)
    data = {
        "pos": torch.cat(
            [cartesian, cartesian.new_zeros(cartesian.shape[0], 1)], dim=-1
        ),
        "edge_src": edge_index[:, 0],
        "edge_dst": edge_index[:, 1],
        "node_input": node_input,
        "batch": batch,
        "node_attr": node_attr,
        "edge_attr": edge_attr,
    }
    return data


def evaluate(model, dataloader):
    df_list = []
    sample_ratio = 0.01
    dim_names = []

    for i in range(9):
        dim_names.append(f"x_{i}")
        dim_names.append(f"z_{i}")

    mse_meter = AverageMeter()

    with torch.no_grad():
        for (
            cartesian,
            force,
        ) in train_dataloader:  # cartesian_alpha.shape == (batch_size, 6)
            cartesian = cartesian.to(device)
            force = force.to(device)

            data = make_data(cartesian.detach())
            force_pred = model(data)[:, 0:2]
            mse: torch.Tensor = (force.view(-1, 2) - force_pred).norm(dim=1).mean()

            mse_meter.update(mse.item(), n=cartesian.shape[0])

            force = force.cpu().numpy().reshape(-1, 18)
            force_pred = force_pred.cpu().numpy().reshape(-1, 18)

            dict_for_df = {}
            for i, name in enumerate(dim_names):
                dict_for_df[f"{name}_truth"] = force[:, i]
                dict_for_df[f"{name}_pred"] = force_pred[:, i]
            df = pd.DataFrame(dict_for_df).sample(frac=sample_ratio)
            df_list.append(df)

    df = pd.concat(df_list)

    wandb.log(
        {
            "test/mse": mse_meter.val,
        }
    )

    wandb.log(
        {
            f"test/residual_min/{name}": np.min(
                df[f"{name}_pred"] - df[f"{name}_truth"]
            )
            for name in dim_names
        }
    )

    wandb.log(
        {
            f"test/residual_max/{name}": np.max(
                df[f"{name}_pred"] - df[f"{name}_truth"]
            )
            for name in dim_names
        }
    )

    wandb.log(
        {
            f"test/scatters/{name}_scatter": plot_regression_scatter(
                df, f"{name}_truth", f"{name}_pred", name
            )
            for name in dim_names
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
    wandb.init(project="Slinky-pretrain-e3nn", config=args)
else:
    wandb.init(mode="disabled")

for epoch in range(num_epochs):
    print(f"epoch {epoch}")
    for i, (cartesian, force,) in enumerate(
        train_dataloader
    ):  # cartesian_alpha.shape == (batch_size, 6)
        cartesian = cartesian.to(device)
        force = force.to(device)

        data = make_data(cartesian.detach())
        with torch.enable_grad():
            output = model(data)
            mse = (force.view(-1, 2) - output[:, 0:2]).norm(dim=1)
            loss = mse.mean()

        wandb.log({"mse": loss})

        sys.stdout.write(f"\rBatch loss: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if i % 1000 == 0:
            evaluate(model, test_dataloader)

    if epoch % 1 == 0:
        evaluate(model, test_dataloader)

    torch.save({"state_dict": model.state_dict(), "epoch": epoch}, "e3nn_checkpoint.pt")
    print()
