# -*- coding: utf-8 -*-
"""Slinky2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xIHTt82SBceEHmHUhLsz2G91Mna6zLTD
"""

import torch
from torch import nn
from torch.utils import data
import torch.nn.functional as F
import scipy.io as scio
import numpy as np
from matplotlib import pyplot as plt

# from IPython import display
import torch.onnx

# from torchsampler import ImbalancedDatasetSampler
import os
import random
import time
import math

# from torch.utils.tensorboard import SummaryWriter
import wandb
import shutil
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import copy

# from torchstat import stat

# from .priority_memory import FastPriorReplayBuffer
from neural_slinky.priority_memory.prio import (
    PrioritizedReplayBuffer,
    PrioritizedReplayBetaScheduler,
    Batch,
)

from neural_slinky.utils import *
from training_functions import evaluate, construct_model
# from neural_slinky import models

import argparse
# import configparser
import toml

torch.set_printoptions(precision=16)
# 1. Start a new run
# wandb.setup(wandb.Settings(program="D:/MatlabWorks/DER/DER_Slinky_Basic/CaseStudy/HORIZONTAL_HANGING/NNTRAINING#ANGLESOFT/slinky_Resnet_0.py"))
# please use your own account here
# wandb.init(project='Slinky', entity='')

# 2. Save model inputs and hyperparameters
# * Configs
argparser = argparse.ArgumentParser()
argparser.add_argument("--device", type=str, help="The computing device")
argparser.add_argument("--datafiles", nargs="+", type=str, help="Path(s) of train data file(s)")
argparser.add_argument("--datafiles_test", nargs="+", type=str, help="Path(s) of test data file(s)")
argparser.add_argument("--lr", type=float, help="Initial learning rate")
argparser.add_argument(
    "--num_epochs", type=int, help="Number of training epochs"
)
argparser.add_argument(
    "--batch_size",
    dest="batch_size_train",
    type=int,
    help="Training batch size",
)
# argparser.add_argument(
#     "--architecture", choices=["MLPPure"], help="Model architecture"
# )
argparser.add_argument(
    "--layer_width", type=int, help="Number of neurons per layer"
)
argparser.add_argument(
    "--num_layers", type=int, help="Number of neurons per layer"
)
argparser.add_argument(
    "--weight_decay",
    # default=0,
    type=float,
    help="Weight decay of the optimizer (similar to l2-regularization)",
)
argparser.add_argument("--seed", type=int, default=1029)
argparser.add_argument("--wandb", dest="wandb", action="store_true", default=False)
argparser.add_argument("--no-wandb", dest="wandb", action="store_false")
argparser.add_argument("--config", type=str, default=None)
args = argparser.parse_args()

if args.config:
    default_config = toml.load(args.config)
    for k, v in args.__dict__.items():
        if not v:
            if k in default_config:
                args.__dict__[k] = default_config[k]
    print(args)

if args.wandb:
    wandb.init(project="Slinky", mode="online")
else:
    wandb.init(project="Slinky", mode="disabled")
config = wandb.config

loss = nn.MSELoss()
# focal_loss_gamma = 3

currentTime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

os.makedirs(currentTime)

# loss_his = np.zeros(num_epochs)

# config.start_learning_rate = lr
# config.architecture = "Subnet_Resnet_like"
# config.dataset = "sample"
# config.epoch = num_epochs
# config.batch_size = batch_size_train
# config.neurons_per_layer = layer_width
# config.noise_amp = noise_amp
# config.miniBatch_maxNum = miniBatch_maxNum
# config.basic_block_layer = 4
# config.loss = "max_loss"
# config.weight_decay = weight_decay
# config.shear_data_k = shear_k
# config.focal_loss_gamma = focal_loss_gamma

config.architecture = "Subnet_Densenet_like"
config.data_augmentation = "no"
config.loss = "mse_loss"
config.description = "training NN to predict directly the forces"
config.update(args)
wandb.run.name = currentTime
# shutil.copy("./slinky_Resnet_0.py", os.path.join(wandb.run.dir, "slinky_Resnet_0.py"))
# wandb.save('slinky_Resnet_0.py')


seed = args.seed
seed_torch(seed)

device = torch.device(args.device)

# * DATA LOADING
training_data_dict = load_slinky_mat_data(args.datafiles)
test_data_dict = load_slinky_mat_data(args.datafiles_test)


validation_ratio = 0.33

features_train: torch.Tensor
features_validate: torch.Tensor
labels_train: torch.Tensor
labels_validate: torch.Tensor
features_train, features_validate, labels_train, labels_validate = train_test_split(
    training_data_dict["feature"],
    training_data_dict["target"],
    test_size=validation_ratio,
)

features_test = test_data_dict["feature"]
labels_test = test_data_dict["target"]

# * DATA PRE-PROCESSING

# ** Normalization modules construction
feature_normalize_layer = Normalize(
    training_data_dict["feature_mean"], training_data_dict["feature_std"]
)

target_denormalize_layer = Denormalize(
    training_data_dict["target_mean"], training_data_dict["target_std"]
)

# construct the input and output for NN training
num_data = features_train.size(0)
num_val_data = features_validate.size(0)
num_data_test = features_test.size(0)
# batch_size_train = int(num_data / 1)  # 1000
batch_size_validate = int(num_val_data / 1)
batch_size_test = int(num_data_test / 1)

data_loader_train = make_dataloader(
    (features_train, labels_train), args.batch_size_train, device=device
)
data_loader_validate = make_dataloader(
    (features_validate, labels_validate), batch_size_validate, device=device
)
data_loader_test = make_dataloader(
    (features_test, labels_test), batch_size_test, device=device
)
print("Dataloader constructed")


net = construct_model("MLPPureSimple", args.num_layers, args.layer_width)
net = nn.Sequential(feature_normalize_layer, net, target_denormalize_layer)
net.to(device)

# 3. Log gradients and model parameters
wandb.watch(net)


net.train()
trainer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    trainer, patience=10, verbose=True
)

best_val_loss = float("inf")
best_model = copy.deepcopy(net)

pr_buffer = PrioritizedReplayBuffer(num_data, alpha=0.1, beta=0.1)
pr_buffer_scheduler = PrioritizedReplayBetaScheduler(pr_buffer, args.num_epochs, 1)

for X, y in data_loader_train:
    pred = net(X)
    err = torch.norm(pred - y, dim=-1)
    for i, (xx, yy) in enumerate(zip(X, y)):
        ind = pr_buffer.add(Batch(feature=xx, target=yy))
        pr_buffer.update_priority(ind, err[i])

buffer_batch_size = args.batch_size_train

for epoch in tqdm(range(1, args.num_epochs + 1)):
    net.train()
    for _ in tqdm(range(int(num_data / buffer_batch_size))):
        data_batch, indices = pr_buffer.sample(batch_size=buffer_batch_size)
        weights = pr_buffer.get_weight(indices)
        nn_input = data_batch["feature"]
        predicted: torch.Tensor = net(nn_input)
        target = data_batch["target"]
        err = predicted - target
        if err.ndim == 2:
            err = torch.norm(err, dim=-1)

        pr_buffer.update_priority(indices, err)

        l = torch.mean(torch.from_numpy(weights).to(device) * err)
        trainer.zero_grad()
        l.backward()
        trainer.step()

    scheduler.step(l)
    pr_buffer_scheduler.step()

    # recording training metrics
    if epoch % 1 == 0:
        train_metrics = evaluate(net, data_loader_train)
        print(f"epoch {epoch}, train mse loss {train_metrics['mse']}")
        print(f"epoch {epoch}, train max abs loss {train_metrics['mae']}")
        wandb.log({"train/mse": train_metrics["mse"]}, step=epoch)
        wandb.log({"train/mae": train_metrics["mae"]}, step=epoch)

        if train_metrics["mae"] < best_val_loss:
            best_val_loss = train_metrics["mae"]
            save_best_model(net, currentTime)
            best_model = copy.deepcopy(net)

        validate_metrics = evaluate(net, data_loader_validate)
        print(f"epoch {epoch}, validation mse loss {validate_metrics['mse']}")
        print(f"epoch {epoch}, validation max abs loss {validate_metrics['mae']}")
        wandb.log({"validation/mse": validate_metrics["mse"]}, step=epoch)
        wandb.log({"validation/mae": validate_metrics["mae"]}, step=epoch)

    if epoch % 100 == 0:
        test_metrics = evaluate(net, data_loader_test)
        print(f"epoch {epoch}, test mse loss {test_metrics['mse']}")
        print(f"epoch {epoch}, test max abs loss {test_metrics['mae']}")
        wandb.log({"test/mse": test_metrics["mse"]}, step=epoch)
        wandb.log({"test/mae": test_metrics["mae"]}, step=epoch)

    if epoch % 20 == 0:
        torch.save(net, "net_Resnet.pkl")


net.eval()

traced_script_module = torch.jit.script(net.to("cpu"))
traced_script_module = traced_script_module.to("cpu")
traced_script_module.save("./" + currentTime + "/traced_slinky_resnet_final.pt")
traced_script_module.save("./traced_slinky_resnet_final.pt")
shutil.copy(
    "./traced_slinky_resnet_final.pt",
    os.path.join(wandb.run.dir, "traced_slinky_resnet_final.pt"),
)
wandb.save("traced_slinky_resnet_final.pt")
net.to(device)

traced_script_module = torch.jit.script(best_model.to("cpu"))
traced_script_module = traced_script_module.to("cpu")
traced_script_module.save("./" + currentTime + "/traced_slinky_resnet_best.pt")
traced_script_module.save("./traced_slinky_resnet_best.pt")
shutil.copy(
    "./traced_slinky_resnet_best.pt",
    os.path.join(wandb.run.dir, "traced_slinky_resnet_best.pt"),
)
wandb.save("traced_slinky_resnet_best.pt")
best_model.to(device)

wandb.finish()