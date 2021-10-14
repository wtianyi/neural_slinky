from typing import Dict
import torch
from neural_slinky import models

def construct_model(architecture_name, num_layers, layer_width):
    # net = MLP_Pure_simple(NeuronsPerLayer,5).to(device)
    net = getattr(models, architecture_name)(layer_width, num_layers)
    print("number of parameters: ", sum(param.numel() for param in net.parameters()))
    return net


def evaluate(
    model: torch.nn.Module, data_loader: torch.utils.data.DataLoader
) -> Dict[str, float]:
    """Evaluate a network on a dataset specified by the data_loader.

    Args:
        model: the network to be evaluated
        data_loader: the dataset loader
    Returns:
        A dictionary of evaluation metrics
    """
    model.eval()
    mean_squared_error = 0
    max_absolute_error = -float("inf")
    count = 0
    # TODO: make sure the model's internal autograd parts are under torch.enable_grad()
    # with torch.no_grad():
    for feature, target in data_loader:
        pred = model(feature)
        next_count = count + feature.shape[0]
        err = target - pred
        mean_squared_error = (
            mean_squared_error * (count / next_count) + err.square().sum() / next_count
        )
        max_absolute_error = max(max_absolute_error, torch.max(torch.norm(err, dim=-1)))
        count = next_count

    mean_squared_error, max_absolute_error = (
        mean_squared_error.item(),
        max_absolute_error.item(),
    )
    return {"mse": mean_squared_error, "mae": max_absolute_error}