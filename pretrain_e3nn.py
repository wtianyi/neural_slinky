from typing import Dict
import sys

import torch
from torch import optim
from torch.random import seed
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split

import wandb
from neural_slinky import e3nn_models
from data import douglas_data

device = "cuda:2"

from neural_slinky.utils import seed_torch

seed_torch(42)

# key = "cartesian_alpha_single"
key = "cartesian"
batch_size = 64
num_epochs = 100
lr = 1e-2
weight_decay = 0

# * Load Douglas data
douglas_dataset_dict: Dict[str, Dict[str, torch.Tensor]] = torch.load(
    "data/douglas_dataset.pt"
)

input_tensor = (
    douglas_dataset_dict["coords"][key]
    .float()
    .clone()
    .detach()
    .requires_grad_(False)
)
output_tensor = (
    douglas_dataset_dict["force"][key]
    .float()
    .clone()
    .detach()
    .requires_grad_(False)
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
    pool_nodes=False
)
model.to(device)
print(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def make_data(cartesian: torch.Tensor):
    num_triplets = cartesian.shape[0]
    batch = torch.arange(num_triplets, device=device).view(-1, 1).repeat((1, 9)).flatten().long()
    edge_index = torch.tensor(
        [
            [1,0], [1,2],
            [4,3], [4,5],
            [7,6], [7,8],
            [1,4], [4,7]
        ], device=device
    ).repeat((num_triplets,1)) + torch.arange(num_triplets, device=device).mul_(9).repeat_interleave(8).view(-1,1)
    edge_index = edge_index.long()

    edge_attr = torch.nn.functional.one_hot(torch.tensor([1,1,1,1,1,1,0,0], device=device).repeat(num_triplets), num_classes=2).float()

    node_attr = torch.nn.functional.one_hot(torch.tensor([1,0,1], device=device).repeat(3*num_triplets), num_classes=2).float()

    cartesian = cartesian.reshape(-1, 2)
    node_input = cartesian.new_ones(cartesian.shape[0], 1)
    data = {
        "pos": torch.cat([cartesian, cartesian.new_zeros(cartesian.shape[0], 1)], dim=-1),
        "edge_src": edge_index[:,0],
        "edge_dst": edge_index[:,1],
        "node_input": node_input,
        "batch": batch,
        "node_attr": node_attr,
        "edge_attr": edge_attr,
    }
    return data


for epoch in range(num_epochs):
    print(f"epoch {epoch}")
    for (
        cartesian,
        force,
    ) in train_dataloader:  # cartesian_alpha.shape == (batch_size, 6)
        cartesian = cartesian.to(device)
        force = force.to(device)

        # computed_force = douglas_data.calculate_douglas_force_cartesian(cartesian)
        # print("computed_force")
        # print(computed_force)
        # print("force")
        # print(force)
        # assert torch.allclose(computed_force, force)

        data = make_data(cartesian.detach())
        with torch.enable_grad():
            output = model(data)
            mse = (force.view(-1,2) - output[:,0:2]).norm(dim=0)
            loss = mse.sum()

        sys.stdout.write(f"\rBatch loss: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
    torch.save({
        "state_dict": model.state_dict(),
        "epoch": epoch
    }, "e3nn_checkpoint.pt")
    print()