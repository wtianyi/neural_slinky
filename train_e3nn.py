from typing import Dict
import sys

import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split

import wandb
from neural_slinky import e3nn_models

device = "cpu"

key = "cartesian_alpha_single"
batch_size = 1024
num_epochs = 100
lr = 1e-1
weight_decay = 0

# * Load Douglas data
douglas_dataset_dict: Dict[str, Dict[str, torch.Tensor]] = torch.load(
    "data/douglas_dataset.pt"
)

input_tensor = (
    douglas_dataset_dict["coords"][key]
    .flatten(start_dim=0, end_dim=1)
    .float()
    .clone()
    .detach()
    .requires_grad_(False)
)
output_tensor = (
    douglas_dataset_dict["force"][key]
    .flatten(start_dim=0, end_dim=1)
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
    train_val_input, train_val_input, test_size=0.2
)

# TODO: do random rotation for augmentation

train_dataset = TensorDataset(train_input[:batch_size], train_output[:batch_size])
val_dataset = TensorDataset(val_input, val_output)
test_dataset = TensorDataset(test_input, test_output)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


model = e3nn_models.SlinkyForcePredictor(
    irreps_node_input="0o",
    irreps_node_attr="0o",
    irreps_node_output="1x1o+1x0o",
    max_radius=0.06,
    num_neighbors=1,
    num_nodes=2,
    mul=50,
    layers=3,
    lmax=2,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

for epoch in range(num_epochs):
    print(f"epoch {epoch}")
    for (
        cartesian_alpha,
        force,
    ) in train_dataloader:  # cartesian_alpha.shape == (batch_size, 6)
        # cartesian_alpha = cartesian_alpha[0]
        # force = force[0]
        # print(cartesian_alpha)

        xy_1 = torch.cat(
            [cartesian_alpha[..., 0:2], torch.zeros(cartesian_alpha.shape[:-1] + (1,))],
            dim=-1,
        )
        alpha_1 = cartesian_alpha[..., 2:3]
        xy_2 = torch.cat(
            [cartesian_alpha[..., 3:5], torch.zeros(cartesian_alpha.shape[:-1] + (1,))],
            dim=-1,
        )
        alpha_2 = cartesian_alpha[..., 5:6]

        # node_input = torch.cat([xy_1, xy_2, alpha_1, alpha_2], dim=-1).to(device)
        node_pos = torch.stack([xy_1, xy_2], dim=-1).to(device)
        bar_alpha = torch.cat([alpha_1, alpha_2], dim=-1).to(device)

        with torch.enable_grad():
            output = model(node_pos, bar_alpha)
        output = output.reshape(-1, 2, output.shape[-1])
        force_xy_1 = output[..., 0, 0:2]
        force_xy_2 = output[..., 1, 0:2]
        force_alpha_1 = output[..., 0, 3:4]
        force_alpha_2 = output[..., 1, 3:4]
        force_predicted = torch.cat(
            [force_xy_1, force_alpha_1, force_xy_2, force_alpha_2], dim=-1
        )
        mse = (force - force_predicted).norm(dim=0)
        loss = mse.sum()

        sys.stdout.write(f"\rBatch loss: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()

        for n, p in model.named_parameters():
            print(n)
            # print(p)
            print(p.grad)

        optimizer.step()
