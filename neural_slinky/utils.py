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


# defining functions
from typing import Union, List, Dict, Sequence


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
    """construct a PyTorch data iterator. """
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
    result = pd.DataFrame(columns=["l1", "l2", "theta", "d_z_1", "d_xi_1", "d_z_2", "d_xi_2", "gamma1", "gamma3"])
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
