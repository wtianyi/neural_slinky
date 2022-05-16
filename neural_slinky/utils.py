import os
import random
import shutil

import scipy.io as scio
import numpy as np
import torch

# import wandb
import pandas as pd
from torch import nn
from torch.utils import data

import plotly.graph_objects as go
import plotly.express as px


# defining functions
from typing import Union, List, Dict, Sequence
import itertools


def load_slinky_mat_data(
    datafiles: Union[str, List[str]], verbose=True
) -> Dict[str, torch.Tensor]:
    """Load the datafiles into cpu as `torch.Tensor`s

    Args:
        datafiles: the path(s) of the matlab format data files
    """
    if isinstance(datafiles, str):
        datafiles = [datafiles]
    feature_key = "NNInput_All_reshape"
    target_key = "NNOutput_All_reshape"
    feature_list = []
    target_list = []
    for datafile in datafiles:
        data = scio.loadmat(datafile)
        ft = torch.from_numpy(data[feature_key].T).float()
        feature_list.append(ft)

        tgt = torch.from_numpy(data[target_key].T).float()
        target_list.append(tgt)

        target_dim = tgt.shape[-1] if len(tgt.shape) > 1 else 1
        feature_dim = ft.shape[-1]
        if verbose:
            print(
                f"Loaded {datafile}\nnum_records: {ft.shape[0]}\tfeature_dim: {feature_dim}\ttarget_dim: {target_dim}"
            )
    feature = torch.cat(feature_list, dim=0)
    feature_mean = torch.mean(feature, dim=0)
    feature_std = torch.std(feature, dim=0)

    target = torch.cat(target_list, dim=0)
    target_mean = torch.mean(target, dim=0)
    target_std = torch.std(target, dim=0)

    return {
        "feature": feature,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target": target,
        "target_mean": target_mean,
        "target_std": target_std,
    }


class Normalize(nn.Module):
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (Tensor): 2d Tensor to be normalized.

        Returns:
            Tensor: Normalized 2d Tensor.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                "Input tensor should be a torch tensor. Got {}.".format(type(tensor))
            )

        if not tensor.is_floating_point():
            raise TypeError(
                "Input tensor should be a float tensor. Got {}.".format(tensor.dtype)
            )

        if tensor.ndim != 2:
            raise ValueError(
                "Expected tensor to be a tensor of size (B, C). Got tensor.size() = "
                "{}.".format(tensor.size())
            )

        if not self.inplace:
            tensor = tensor.clone()

        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        if (std == 0).any():
            raise ValueError(
                "std evaluated to zero after conversion to {}, leading to division by zero.".format(
                    dtype
                )
            )
        # if mean.ndim <= 1:
        mean = mean.expand_as(tensor)
        # if std.ndim <= 1:
        std = std.expand_as(tensor)
        tensor.sub_(mean).div_(std)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class Denormalize(nn.Module):
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (Tensor): 2d Tensor to be normalized.

        Returns:
            Tensor: Normalized 2d Tensor.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                "Input tensor should be a torch tensor. Got {}.".format(type(tensor))
            )

        if not tensor.is_floating_point():
            raise TypeError(
                "Input tensor should be a float tensor. Got {}.".format(tensor.dtype)
            )

        if tensor.ndim != 2:
            raise ValueError(
                "Expected tensor to be a tensor of size (B, C). Got tensor.size() = "
                "{}.".format(tensor.size())
            )

        if not self.inplace:
            tensor = tensor.clone()

        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        if (std == 0).any():
            raise ValueError(
                "std evaluated to zero after conversion to {}, leading to division by zero.".format(
                    dtype
                )
            )
        if mean.ndim <= 1:
            mean = mean.view(1, -1)
        if std.ndim <= 1:
            std = std.view(1, -1)
        tensor.mul_(std).add_(mean)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


def make_dataloader(
    data_tensors: Sequence[torch.Tensor],
    batch_size: int,
    device="cpu",
    is_train: bool = True,
):
    """construct a PyTorch data iterator."""
    data_tensors = [data.to(device) for data in data_tensors]
    dataset = data.TensorDataset(*data_tensors)
    return data.DataLoader(
        dataset, batch_size, shuffle=is_train
    )  # , num_workers=6)#, pin_memory=True)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.1)


def features_normalize(data):
    mu = torch.mean(data, dim=1, keepdim=True)
    std = torch.std(data, dim=1, keepdim=True)
    # mu = np.mean(data.cpu().numpy(),axis=1)
    # std = np.std(data.cpu().numpy(),axis=1)

    # print(std.shape)
    # print(std.reshape(3,1).shape)
    # print(std)
    # print(std.reshape(3,1))
    out = (data - mu) / std
    print("mu:", mu)
    print("std:", std)
    return out, mu, std


def features_normalize_withknown(data, mu, std):
    out = (data - mu.reshape(-1, 1)) / (std.reshape(-1, 1))
    return out


def chiral_transformation_x(data):
    # data is a 6-dim input vector
    # the output is a 6-dim vector, as the mirror of input data with respect to the x-axis
    # new_data = torch.zeros_like(data)
    new_data = data.clone()
    new_data[:, :2] = data[:, :2]
    new_data[:, 2] = -data[:, 2]
    new_data[:, 3:] = -data[:, 3:]
    return new_data


def chiral_transformation_z(data):
    # data is a 6-dim input vector
    # the output is a 6-dim vector, as the mirror of input data with respect to the z-axis
    # new_data = torch.zeros_like(data)
    new_data = data.clone()
    new_data[:, 0] = data[:, 1]
    new_data[:, 1] = data[:, 0]
    new_data[:, 2] = data[:, 2]
    new_data[:, 3] = -data[:, 5]
    new_data[:, 4] = -data[:, 4]
    new_data[:, 5] = -data[:, 3]
    return new_data


def chiral_transformation_xz(data):
    # data is a 6-dim input vector
    # the output is a 6-dim vector, as the 180 degrees rotation of input data
    # new_data = torch.zeros_like(data)
    new_data = data.clone()
    new_data = chiral_transformation_x(data)
    new_data = chiral_transformation_z(new_data)
    return new_data


def chiral_transformation_x_douglas(data):
    new_data = torch.empty_like(data)
    new_data[..., 0] = -data[..., 0]
    new_data[..., 2] = -data[..., 2]
    new_data[..., 3] = -data[..., 3]
    new_data[..., 5] = -data[..., 5]
    return new_data


def chiral_transformation_z_douglas(data):
    new_data = torch.empty_like(data)
    new_data[..., 0] = data[..., 3]
    new_data[..., 1] = -data[..., 4]
    new_data[..., 2] = data[..., 5]
    new_data[..., 3] = data[..., 0]
    new_data[..., 4] = -data[..., 1]
    new_data[..., 5] = data[..., 2]
    return new_data


def chiral_transformation_xz_douglas(data):
    new_data = torch.empty_like(data)
    new_data[..., 0] = -data[..., 3]
    new_data[..., 1] = -data[..., 4]
    new_data[..., 2] = -data[..., 5]
    new_data[..., 3] = -data[..., 0]
    new_data[..., 4] = -data[..., 1]
    new_data[..., 5] = -data[..., 2]
    # new_data = chiral_transformation_x_douglas(data)
    # new_data = chiral_transformation_z_douglas(new_data)
    return new_data


def chiral_augment_douglas(data):
    return torch.stack(
        [
            data,
            chiral_transformation_x_douglas(data),
            chiral_transformation_z_douglas(data),
            chiral_transformation_xz_douglas(data),
        ],
        dim=0,
    )


def make_tan(data):
    # data is a 6-dim input vector
    # the output is a 6-dim vector, as the 180 degrees rotation of input data
    # new_data = torch.zeros_like(data)
    new_data = data.clone()
    new_data[:, 0] = data[:, 0]
    new_data[:, 1] = data[:, 1]
    new_data[:, 2] = data[:, 2]
    new_data[:, 3] = torch.tan(new_data[:, 3])
    new_data[:, 4] = torch.tan(new_data[:, 4])
    new_data[:, 5] = torch.tan(new_data[:, 5])
    return new_data


def convert_xyz_to_l_gamma(data: torch.Tensor):
    new_data = torch.empty(data.shape[0], 6, device=data.device)

    new_data[:, 0] = torch.sqrt(
        torch.square(data[:, 0] - data[:, 3]) + torch.square(data[:, 1] - data[:, 4])
    )  # l1
    new_data[:, 1] = torch.sqrt(
        torch.square(data[:, 3] - data[:, 6]) + torch.square(data[:, 4] - data[:, 7])
    )  # l2
    theta_1 = torch.atan2(-(data[:, 3] - data[:, 0]), data[:, 4] - data[:, 1])
    theta_2 = torch.atan2(-(data[:, 6] - data[:, 3]), data[:, 7] - data[:, 4])
    new_data[:, 2] = theta_2 - theta_1  # theta
    new_data[:, 3] = data[:, 2] - data[:, 5]  # gamma1
    theta_3 = torch.atan2(-(data[:, 0] - data[:, 3]), data[:, 1] - data[:, 4])
    theta_4 = torch.atan2(-(data[:, 6] - data[:, 3]), data[:, 7] - data[:, 4])
    new_data[:, 4] = (theta_3 + theta_4) / 2 - data[:, 5]  # gamma2
    new_data[:, 5] = data[:, 8] - data[:, 5]  # gamma3

    # new_data (relative coordinates should be 6 dimensional)
    # new_data = new_data[:, :6]
    return new_data


def region_shifting(data, data_mean, data_std):
    new_data = data.clone()
    new_data[:, 0] = data[:, 0]
    new_data[:, 1] = data[:, 1]
    new_data[:, 2] = data[:, 2]
    new_data[:, 3] = data[:, 3]
    new_data[:, 5] = data[:, 5]
    # int_3 = torch.ceil(-(data_mean[3] + data_std[3] * data[:,3]) / math.pi)
    int_4 = torch.ceil(-(data_mean[4] + data_std[4] * data[:, 4]) / np.pi)
    # int_5 = torch.ceil(-(data_mean[5] + data_std[5] * data[:,5]) / math.pi)
    # new_data[:,3] = data[:,3] + int_3 * math.pi / data_std[3] + data_mean[3] / data_std[3]
    new_data[:, 4] = (
        data[:, 4] + int_4 * np.pi / data_std[4] + data_mean[4] / data_std[4]
    )
    # new_data[:,5] = data[:,5] + int_5 * math.pi / data_std[5] + data_mean[5] / data_std[5]
    # print(sum(new_data[:,3] >= math.pi))
    # print(sum(new_data[:,4] >= math.pi))
    # print(sum(new_data[:,5] >= math.pi))
    # print(sum(new_data[:,3] < 0))
    # print(sum(new_data[:,4] < 0))
    # print(sum(new_data[:,5] < 0))
    # assert(sum(new_data[:,3] >= math.pi) == 0)
    # assert(sum(new_data[:,4] >= math.pi) == 0)
    # assert(sum(new_data[:,5] >= math.pi) == 0)
    # assert(sum(new_data[:,3] < 0) == 0)
    # assert(sum(new_data[:,4] < 0) == 0)
    # assert(sum(new_data[:,5] < 0) == 0)
    return new_data


def region_shifting2(data, data_mean, data_std):
    new_data = data.clone()
    new_data[:, 0] = data[:, 0]
    new_data[:, 1] = data[:, 1]
    new_data[:, 2] = data[:, 2]
    new_data[:, 3] = data[:, 3]
    new_data[:, 5] = data[:, 5]
    # int_3 = torch.ceil(-0.5-(data_mean[3] + data_std[3] * data[:,3]) / math.pi)
    int_4 = torch.ceil(-0.5 - (data_mean[4] + data_std[4] * data[:, 4]) / np.pi)
    # int_5 = torch.ceil(-0.5-(data_mean[5] + data_std[5] * data[:,5]) / math.pi)
    # new_data[:,3] = data[:,3] + int_3 * math.pi / data_std[3] + data_mean[3] / data_std[3]
    new_data[:, 4] = (
        data[:, 4] + int_4 * np.pi / data_std[4] + data_mean[4] / data_std[4]
    )
    # new_data[:,5] = data[:,5] + int_5 * math.pi / data_std[5] + data_mean[5] / data_std[5]
    return new_data


def region_shifting3(data):
    new_data = data.clone()
    new_data[:, 0] = data[:, 0]
    new_data[:, 1] = data[:, 1]
    new_data[:, 2] = data[:, 2]
    new_data[:, 3] = data[:, 3]
    new_data[:, 5] = data[:, 5]
    int_4 = torch.ceil(-0.5 - data[:, 4] / np.pi)
    new_data[:, 4] = data[:, 4] + int_4 * np.pi
    return new_data


# def finding_region(data):
#     return
def make_shear_data(shear_k=1000):
    # a range of l1 and l2
    l1 = np.random.uniform(0.02, 0.04, 1000)
    l2 = np.random.uniform(0.02, 0.04, 1000)
    # theta will remain 0
    theta = np.zeros_like(l1)
    # gamma_1 and gamma_3 will remain 0
    gamma_1 = np.zeros_like(theta)
    gamma_3 = gamma_1
    # a range of gamma_2
    gamma_2 = np.random.uniform(np.pi / 4, np.pi / 2, len(gamma_1))

    # the stiffness k for shear
    # shear_k = 100;

    # making augment dataset
    shear_data_input = np.stack((l1, l2, theta, gamma_1, gamma_2, gamma_3), 1)
    # making output labels
    shear_data_output = np.zeros_like(gamma_2)
    for ii in range(len(gamma_1)):
        shear_data_output[ii] = (
            0.5
            * shear_k
            * (
                (l1[ii] * np.cos(gamma_2[ii] + np.pi / 2)) ** 2
                + (l2[ii] * np.cos(gamma_2[ii] + np.pi / 2)) ** 2
            )
        )

    return (
        shear_data_input.astype("float32"),
        shear_data_output.reshape(-1, 1).astype("float32"),
    )


def get_model_device(model: nn.Module):
    return next(iter(model.parameters())).device


def save_best_model(net: nn.Module, current_time):
    # device = get_model_device(net)
    # example = torch.rand(1, 6)
    traced_script_module = torch.jit.script(net)
    traced_script_module = traced_script_module.to("cpu")
    save_path = "./" + current_time + "/traced_slinky_resnet.pt"
    traced_script_module.save(save_path)
    # traced_script_module.save("./traced_slinky_resnet.pt")
    # shutil.copy(save_path, os.path.join(wandb.run.dir, "traced_slinky_resnet.pt"))
    # wandb.save("traced_slinky_resnet.pt")
    # net.to(device)


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def augment_z_xi(df):
    result = pd.DataFrame(
        columns=[
            "l1",
            "l2",
            "theta",
            "d_z_1",
            "d_xi_1",
            "d_z_2",
            "d_xi_2",
            "gamma1",
            "gamma3",
        ]
    )
    result["l1"] = df["l1"]
    result["l2"] = df["l2"]
    result["theta"] = df["theta"]
    result["gamma1"] = df["gamma1"]
    result["gamma3"] = df["gamma3"]

    phi_1 = 0.5 * (np.pi - df["theta"]) - df["gamma2"] - 0.5 * df["gamma1"]
    phi_2 = 0.5 * (np.pi + df["theta"]) - df["gamma2"] + 0.5 * df["gamma3"]

    result["phi_1"] = phi_1
    result["phi_2"] = phi_2

    result["d_z_1"] = df["l1"] * np.cos(phi_1)
    result["d_xi_1"] = df["l1"] * np.sin(phi_1)

    result["d_z_2"] = df["l2"] * np.cos(phi_2)
    result["d_xi_2"] = df["l2"] * np.sin(phi_2)
    return result


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val: float = 0
        self.avg: float = 0
        self.sum: float = 0
        self.count: int = 0

    def update(self, val: float, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def plot_regression_scatter(
    df: pd.DataFrame, truth_column: str, pred_column: str, title: str
):
    # fig = go.Figure()
    fig = px.scatter(
        df,
        x=truth_column,
        y=pred_column,
        marginal_x="histogram",
        marginal_y="histogram",
        title=title,
    )
    fig.add_trace(
        go.Scatter(
            x=df[truth_column],
            y=df[truth_column],
            mode="lines",
            line={"dash": "dot", "color": "#000"},
        )
    )
    max_value = np.max(np.abs(df[truth_column]))
    fig["layout"]["xaxis"].update(
        range=[-1.2 * max_value, 1.2 * max_value],  # sets the range of xaxis
        # constrain="domain",  # meanwhile compresses the xaxis by decreasing its "domain"
    )
    fig["layout"]["yaxis"].update(
        range=[-1.2 * max_value, 1.2 * max_value],  # sets the range of yaxis
        scaleanchor="x",
        scaleratio=1,
    )
    return fig


def num_parameters(model: nn.Module) -> int:
    return sum([w.numel() for w in model.parameters()])


def animate_2d_slinky_trajectories(trajectories: torch.Tensor, dt, animation_slice=100):
    """Plot a plotly animation of the motion of a DER

    Args:
        trajectory (torch.Tensor): 2D slinky trajectory (n_trajectories x n_frames x num_cycles x 3)
        dt (float): The time step during simulation
    """
    simulation_steps = trajectories.shape[1]
    interval = int(simulation_steps / animation_slice)
    trajectories = trajectories[:, ::interval]  # downsampling
    # node_coords = trajectories
    node_coords = (
        trajectories.clone().detach().cpu().transpose(1, 0).numpy()
    )  # transpose the time axis to the first dim
    # num_frames = len(node_coords)
    # node_coords = [t.clone().cpu().numpy() for t in trajectories]
    bar_len = 0.02

    def calculate_bar_df(x, y, a, l=1):
        a = a.reshape(-1, 1)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        dx = -0.5 * l * np.sin(a)
        dy = 0.5 * l * np.cos(a)
        xx = np.hstack([x + dx, x - dx, np.ones_like(x) * np.nan]).ravel()
        yy = np.hstack([y + dy, y - dy, np.ones_like(y) * np.nan]).ravel()
        return pd.DataFrame({"x": xx, "y": yy})

    def make_frame_data(coords):
        frame_data = []
        if len(coords.shape) == 2:
            coords = coords[None, :]
        if len(coords.shape) != 3:
            raise ValueError(f"`coords` must have 2 or 3 dims. got {coords.shape=}")
        colors = px.colors.qualitative.Plotly
        for coord, color in zip(coords, itertools.cycle(colors)):
            x = coord[:, 0]
            y = coord[:, 1]
            a = coord[:, 2]
            bar_df = calculate_bar_df(x, y, a, bar_len)
            frame_data.extend(
                [
                    {
                        "type": "scatter",
                        "x": x,
                        "y": y,
                        "line": {"color": color, "width": 2},
                        # "marker": {"size": 4, "color": "red"},
                        "mode": "lines",
                    },
                    {
                        "type": "scatter",
                        "x": bar_df["x"],
                        "y": bar_df["y"],
                        "line": {"color": color, "width": 1},
                    },
                ]
            )
        return frame_data

    data = make_frame_data(node_coords[0])
    # axis = dict(
    #     showbackground=True,
    #     backgroundcolor="rgb(230, 230,230)",
    #     gridcolor="rgb(255, 255, 255)",
    #     zerolinecolor="rgb(255, 255, 255)",
    # )
    frames = []
    for k, coords in enumerate(node_coords):
        frame_data = make_frame_data(coords)
        frames.append(
            {
                "data": frame_data,
                # "traces": [0, 1],
                "name": f"frame {k}",
            }
        )

    sliders = [
        dict(
            steps=[
                dict(
                    method="animate",  # Sets the Plotly method to be called when the
                    # slider value is changed.
                    args=[
                        [
                            "frame {}".format(k)
                        ],  # Sets the arguments values to be passed to
                        # the Plotly method set in method on slide
                        dict(
                            mode="immediate",
                            frame=dict(duration=50, redraw=False),
                            transition=dict(duration=0),
                        ),
                    ],
                    label="{:.2f}".format(k * interval * dt),
                )
                for k in range(animation_slice)
            ],
            transition=dict(duration=0),
            x=0,  # slider starting position
            y=0,
            currentvalue=dict(
                font=dict(size=12), prefix="Time: ", visible=True, xanchor="center"
            ),
            len=1.0,
        )  # slider length)
    ]

    x_all = node_coords[..., 0]
    xmax, xmin = x_all.max(), x_all.min()
    xrange = xmax - xmin
    y_all = node_coords[..., 1]
    ymax, ymin = y_all.max(), y_all.min()
    yrange = ymax - ymin
    margin = 0.05

    layout = dict(
        title="Animating a 2D Slinky",
        autosize=True,
        # width=600,
        # height=600,
        showlegend=False,
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
        ),
        xaxis_range=[xmin - margin * xrange, xmax + margin * xrange],
        yaxis_range=[ymin - margin * yrange, ymax + margin * yrange],
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0,
                x=1.15,
                xanchor="right",
                yanchor="top",
                pad=dict(t=0, r=10),
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=20, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    )
                ],
            )
        ],
        sliders=sliders,
    )
    fig = go.Figure(data=data, layout=layout, frames=frames)
    # fig.show(renderer="notebook_connected")
    return fig


def boolean_flag(parser, name: str, default: bool = False, help: str = None) -> None:
    """Add a boolean flag to argparse parser.

    Parameters
    ----------
    parser (:obj:`argparse.ArgumentParser`):
        parser to add the flag to
    name (:obj:`str`):
        --<name> will enable the flag, while --no-<name> will disable it
    default (:obj:`bool`):
        default value of the flag
    help (:obj:`str`):
        help string for the flag
    """
    dest = name.replace("-", "_")
    parser.add_argument(
        "--" + name, action="store_true", default=default, dest=dest, help=help
    )
    parser.add_argument("--no-" + name, action="store_false", dest=dest)
