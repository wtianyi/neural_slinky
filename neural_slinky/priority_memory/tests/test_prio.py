#!/usr/bin/env python3

import unittest
from ..batch import Batch
from ..prio import PrioritizedReplayBuffer


import torch
from torch import nn
from torch.utils import data
import scipy.io as scio
import numpy as np
import os
import random
import math

# from torch.utils.tensorboard import SummaryWriter
import shutil
from sklearn.model_selection import train_test_split

device = torch.device("cuda:1")

# defining functions
def load_array(data_arrays, batch_size, is_train=True):
    """construct a PyTorch data iterator. """
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(
        dataset, batch_size, shuffle=is_train
    )  # , num_workers=6)#, pin_memory=True)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.1)


def features_normalize(data):
    mu = np.mean(data.cpu().numpy(), axis=1)
    std = np.std(data.cpu().numpy(), axis=1)
    # print(std.shape)
    # print(std.reshape(3,1).shape)
    # print(std)
    # print(std.reshape(3,1))
    out = (data.cpu().numpy() - mu.reshape(len(mu), 1)) / (std.reshape(len(std), 1))
    print(mu)
    print(std)
    return torch.from_numpy(out).to(device), mu, std


def features_normalize_withknown(data, mu, std):
    out = (data.cpu().numpy() - mu.reshape(len(mu), 1)) / (std.reshape(len(std), 1))
    return torch.from_numpy(out).to(device)


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


def preprocessing(data):
    new_data = data.clone()

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
    new_data = new_data[:, :6]
    return new_data


def region_shifting(data, data_mean, data_std):
    new_data = data.clone()
    new_data[:, 0] = data[:, 0]
    new_data[:, 1] = data[:, 1]
    new_data[:, 2] = data[:, 2]
    new_data[:, 3] = data[:, 3]
    new_data[:, 5] = data[:, 5]
    # int_3 = torch.ceil(-(data_mean[3] + data_std[3] * data[:,3]) / math.pi)
    int_4 = torch.ceil(-(data_mean[4] + data_std[4] * data[:, 4]) / math.pi)
    # int_5 = torch.ceil(-(data_mean[5] + data_std[5] * data[:,5]) / math.pi)
    # new_data[:,3] = data[:,3] + int_3 * math.pi / data_std[3] + data_mean[3] / data_std[3]
    new_data[:, 4] = (
        data[:, 4] + int_4 * math.pi / data_std[4] + data_mean[4] / data_std[4]
    )
    return new_data


def region_shifting2(data, data_mean, data_std):
    new_data = data.clone()
    new_data[:, 0] = data[:, 0]
    new_data[:, 1] = data[:, 1]
    new_data[:, 2] = data[:, 2]
    new_data[:, 3] = data[:, 3]
    new_data[:, 5] = data[:, 5]
    # int_3 = torch.ceil(-0.5-(data_mean[3] + data_std[3] * data[:,3]) / math.pi)
    int_4 = torch.ceil(-0.5 - (data_mean[4] + data_std[4] * data[:, 4]) / math.pi)
    # int_5 = torch.ceil(-0.5-(data_mean[5] + data_std[5] * data[:,5]) / math.pi)
    # new_data[:,3] = data[:,3] + int_3 * math.pi / data_std[3] + data_mean[3] / data_std[3]
    new_data[:, 4] = (
        data[:, 4] + int_4 * math.pi / data_std[4] + data_mean[4] / data_std[4]
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
    int_4 = torch.ceil(-0.5 - data[:, 4] / math.pi)
    new_data[:, 4] = data[:, 4] + int_4 * math.pi
    return new_data


# class MLPBlock(nn.Module):
#     def __init__(self, NeuronsPerLayer, NumLayer):
#         super(MLPBlock, self).__init__()
#         layer = []
#         for i in range(NumLayer):
#             layer.append(
#                 nn.Sequential(
#                 # nn.BatchNorm1d(NeuronsPerLayer * i + NeuronsPerLayer),
#                 nn.Linear(NeuronsPerLayer, NeuronsPerLayer),
#                 nn.Softplus(beta=1e1)#, Square()
#                 # nn.Tanh()
#                 )
#                 )
#         self.net = nn.Sequential(*layer)

#     def forward(self, X):
#         for blk in self.net:
#             Y = blk(X)
#             # Concatenate the input and output of each block on the channel
#             # dimension
#             X = Y
#         return X

# class DenseBlock(nn.Module):
#     def __init__(self, NeuronsPerLayer, NumLayer):
#         super(DenseBlock, self).__init__()
#         layer = []
#         for i in range(NumLayer):
#             layer.append(
#                 nn.Sequential(
#                 # nn.BatchNorm1d(NeuronsPerLayer * i + NeuronsPerLayer),
#                 nn.Linear(NeuronsPerLayer * i + NeuronsPerLayer, NeuronsPerLayer),
#                 nn.Softplus(beta=1e1)#, Square()
#                 # nn.ReLU()
#                 # nn.LeakyReLU(negative_slope=0.1)
#                 )
#                 )
#                 # nn.Linear(NeuronsPerLayer * i + NeuronsPerLayer, NeuronsPerLayer)), nn.Tanh()
#         # layer.append(
#         #     nn.Sequential(
#         #     # nn.BatchNorm1d(NeuronsPerLayer * i + NeuronsPerLayer),
#         #     nn.Linear(NeuronsPerLayer * (NumLayer-1) + NeuronsPerLayer, NeuronsPerLayer),
#         #     nn.Softplus(), Square()
#         #     )
#         #     )
#         self.net = nn.Sequential(*layer)

#     def forward(self, X):
#         for blk in self.net:
#             Y = blk(X)
#             # Concatenate the input and output of each block on the channel
#             # dimension
#             X = torch.cat((X, Y), dim=-1)
#         return X

# class MLP_Pure_simple(nn.Module):
#     def __init__(self, NeuronsPerLayer=32, NumLayer=4):
#         super(MLP_Pure_simple, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Linear(9,NeuronsPerLayer),
#             # RotationInvariantLayer(NeuronsPerLayer),
#             nn.Softplus(beta=1e3)
#             # nn.ReLU()
#             # nn.LeakyReLU(negative_slope=0.1)
#         )
#         self.layer2 = nn.Sequential(
#             # nn.Linear(NeuronsPerLayer,NeuronsPerLayer), nn.Softplus(beta=1e3),
#             # nn.Linear(NeuronsPerLayer,NeuronsPerLayer), nn.Softplus(beta=1e3)
#             # DenseBlock(NeuronsPerLayer,NumLayer)
#             MLPBlock(NeuronsPerLayer,NumLayer)
#         )
#         self.layer3 = nn.Sequential(
#             # nn.Linear(int(NeuronsPerLayer*(NumLayer+1)),3)
#             nn.Linear(int(NeuronsPerLayer),3)
#         )

#     def forward(self, y):
#         # y.requires_grad_(True)
#         # print(y)

#         # 1. do preprocessing (equivalently do rigid body motion invariance)
#         # x = preprocessing(y)

#         # x = (x - features_mu) / features_std

#         # 2. pass through the NN
#         out = self.layer1(y)
#         out = self.layer2(out)
#         out = self.layer3(out)

#         # 3. taking the derivative
#         # deriv = torch.autograd.grad([out.sum()],[y],retain_graph=True,create_graph=True)
#         # deriv = torch.autograd.grad(out,y,retain_graph=True,create_graph=True)
#         # deriv = torch.sum(out).backward(y,retain_graph=True,create_graph=True)

#         # grad = deriv[0]
#         # if grad is not None:
#         #     return grad[:, 3:6]
#         return out

# class MLP_Pure(nn.Module):
#     def __init__(self, NeuronsPerLayer=32, NumLayer=4):
#         super(MLP_Pure, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Linear(6,NeuronsPerLayer),
#             # RotationInvariantLayer(NeuronsPerLayer),
#             nn.Softplus(beta=1e1)
#             # nn.ReLU()
#             # nn.LeakyReLU(negative_slope=0.1)
#         )
#         self.layer2 = nn.Sequential(
#             # DenseBlock(NeuronsPerLayer,NumLayer)
#             MLPBlock(NeuronsPerLayer,NumLayer)
#         )
#         self.layer3 = nn.Sequential(
#             # nn.Linear(int(NeuronsPerLayer/8),1)
#             # nn.Linear(int(NeuronsPerLayer*(NumLayer+1)),1)
#             nn.Linear(int(NeuronsPerLayer),1)
#         )

#     def forward(self, y):
#         y.requires_grad_(True)
#         # print(y)

#         # 1. do preprocessing (equivalently do rigid body motion invariance)
#         x = preprocessing(y)

#         # x = (x-features_mu) / features_std

#         # 2. do periodicity
#         x = region_shifting3(x)
#         # x = region_shifting2(x,features_mu,features_std)

#         # 3. do chirality
#         augmented_x = torch.stack([x, chiral_transformation_x(x), chiral_transformation_z(x), chiral_transformation_xz(x)], dim=0)

#         # 4. pass through the NN
#         out = self.layer1(augmented_x)
#         out = self.layer2(out)
#         out = self.layer3(out)

#         # 5. getting the energy surrogate
#         out = torch.sum(out, dim=0, keepdim=False)

#         # 6. taking the derivative
#         deriv = torch.autograd.grad([out.sum()],[y],retain_graph=True,create_graph=True)
#         # deriv = torch.autograd.grad(out,y,retain_graph=True,create_graph=True)
#         # deriv = torch.sum(out).backward(y,retain_graph=True,create_graph=True)

#         grad = deriv[0]
#         if grad is not None:
#             return grad[:, 3:6]

# def save_best_model():
#     example = torch.rand(1,6)
#     traced_script_module = torch.jit.trace(net, example.to(device))
#     traced_script_module = traced_script_module.to("cpu")
#     traced_script_module.save("./"+currentTime+"/traced_slinky_resnet.pt")
#     traced_script_module.save("./traced_slinky_resnet.pt")
#     shutil.copy("./traced_slinky_resnet.pt", os.path.join(wandb.run.dir, "traced_slinky_resnet.pt"))
#     wandb.save("traced_slinky_resnet.pt")
#     net.to(device)

# def seed_torch(seed=1029):
# 	random.seed(seed)
# 	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
# 	np.random.seed(seed)
# 	torch.manual_seed(seed)
# 	torch.cuda.manual_seed(seed)
# 	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
# 	torch.backends.cudnn.benchmark = False
# 	torch.backends.cudnn.deterministic = True

# seed_torch()

# # loading in data
# # dataFile = '../RAWDATA#ANGLENEW#CHANGE#RELATIVE#DERIV/SlinkyData_Angle_6_new_Douglas.mat'
# dataFile = './SlinkyData_Angle_6_new_Douglas.mat'
# dataFile_Augmentation = []

# # dataFile_Test = '../RAWDATA#ANGLENEW#CHANGE#RELATIVE#DERIV/SlinkyData_Angle_6_new_Douglas.mat'
# dataFile_Test = './SlinkyData_Angle_6_new_Douglas.mat'
# data2 = scio.loadmat(dataFile)
# data_Test = scio.loadmat(dataFile_Test)
# print("Data loaded")
# data_input = data2['NNInput_All_reshape']
# data_output = data2['NNOutput_All_reshape']
# data_input, data_output = data_input.astype('float32'), data_output.astype('float32')
# print(data_input.shape)
# print(data_output.shape)

# for file_name in dataFile_Augmentation:
#     data_augmentation = scio.loadmat(file_name)
#     data_augmentation_input = data_augmentation['NNInput_All_reshape']
#     data_augmentation_output = data_augmentation['NNOutput_All_reshape']
#     data_augmentation_input, data_augmentation_output = data_augmentation_input.astype('float32'), data_augmentation_output.astype('float32')
#     data_input = np.concatenate((data_input,data_augmentation_input),axis=1)
#     data_output = np.concatenate((data_output,data_augmentation_output),axis=1)
# print('Data augmentation loaded')

# data_input_Test = data_Test['NNInput_All_reshape']
# data_output_Test = data_Test['NNOutput_All_reshape']
# data_input_Test, data_output_Test = data_input_Test.astype('float32'), data_output_Test.astype('float32')

# num_samples = data_input.shape[1] - 2
# data_input = data_input[:,-num_samples:-1]
# data_output = data_output[:,-num_samples:-1]

# features_train, features_validate, labels_train, labels_validate = train_test_split(np.transpose(data_input), np.transpose(data_output), test_size=0.33)
# features_train, features_validate, labels_train, labels_validate = torch.from_numpy(np.transpose(features_train)).to(device), torch.from_numpy(np.transpose(features_validate)).to(device), torch.from_numpy(np.transpose(labels_train)).to(device), torch.from_numpy(np.transpose(labels_validate)).to(device)
# features_Test, labels_Test = torch.from_numpy(data_input_Test).to(device), torch.from_numpy(data_output_Test).to(device)

# # # perform data transformation and add new dat into the dataset
# # # mirror with respect to the z axis
# # features_new = features_train.clone()
# # features_new = features_new[torch.tensor([1, 0, 2, 5, 4, 3]),:]
# # features_new[3:6,:] = -features_new[3:6,:]
# # labels_new = labels_train.clone()
# # features_train_all = torch.cat((features_train,features_new),1)
# # labels_train_all = torch.cat((labels_train,labels_new),1)

# # # mirror with respect to the x axis
# # features_new = features_train.clone()
# # features_new[2,:] = -features_new[2,:]
# # features_new[3:6,:] = -features_new[3:6,:]
# # labels_new = labels_train.clone()
# # features_train_all = torch.cat((features_train_all,features_new),1)
# # labels_train_all = torch.cat((labels_train_all,labels_new),1)

# # # rotate by pi
# # features_new = features_train.clone()
# # features_new = features_new[torch.tensor([1, 0, 2, 5, 4, 3]),:]
# # features_new[2,:] = -features_new[2,:]
# # labels_new = labels_train.clone()
# # features_train_all = torch.cat((features_train_all,features_new),1)
# # labels_train_all = torch.cat((labels_train_all,labels_new),1)

# torch.set_printoptions(precision=16)

# # _, features_mu, features_std = features_normalize(features_train_all)
# _, labels_mu, labels_std = features_normalize(labels_train)
# print(labels_mu)
# print(labels_std)

# # # normalization with z-score
# # features_train = features_normalize_withknown(features_train, features_mu, features_std)
# # labels_train = features_normalize_withknown(labels_train, labels_mu, labels_std)
# # features_validate = features_normalize_withknown(features_validate, features_mu, features_std)
# # labels_validate = features_normalize_withknown(labels_validate, labels_mu, labels_std)
# # features_Test = features_normalize_withknown(features_Test, features_mu, features_std)
# # labels_Test = features_normalize_withknown(labels_Test, labels_mu, labels_std)

# # # normalization with min-max
# # # features_train -= features_train.min(1, keepdim=True)[0]
# # # features_train /= features_train.max(1, keepdim=True)[0]
# # # labels_train -= labels_train.min(1, keepdim=True)[0]
# # # labels_train /= labels_train.max(1, keepdim=True)[0]
# # # features_validate -= features_validate.min(1, keepdim=True)[0]
# # # features_validate /= features_validate.max(1, keepdim=True)[0]
# # # labels_validate -= labels_validate.min(1, keepdim=True)[0]
# # # labels_validate /= labels_validate.max(1, keepdim=True)[0]
# # # features_Test -= features_Test.min(1, keepdim=True)[0]
# # # features_Test /= features_Test.max(1, keepdim=True)[0]
# # # labels_Test -= labels_Test.min(1, keepdim=True)[0]
# # # labels_Test /= labels_Test.max(1, keepdim=True)[0]
# print("Data normalized")

# features_train = features_train.permute(1, 0)
# labels_train = labels_train.permute(1, 0)
# features_validate = features_validate.permute(1, 0)
# labels_validate = labels_validate.permute(1, 0)
# features_Test = features_Test.permute(1, 0)
# labels_Test = labels_Test.permute(1, 0)

# print(features_train.size())
# print(labels_train.size())

# plotData = 0
# if plotData == 1:
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')

#     ax.set_xlabel('l_{i-1}')
#     ax.set_ylabel('l_{i}')
#     ax.set_zlabel('theta_i')

#     xs = features_train[:,0].cpu().detach().numpy()
#     ys = features_train[:,1].cpu().detach().numpy()
#     zs = features_train[:,2].cpu().detach().numpy()
#     v = labels_train.cpu().detach().numpy()
#     c = np.abs(v)

#     cmhot = plt.get_cmap("hot")
#     ax.scatter(xs, ys, zs, v, s=5, c=c, cmap=cmhot)
#     plt.show()
#     input("look at the figures")

# # construct the input and output for NN training
# num_data = features_train.size(0)
# batch_size = int(features_train.size()[0]/1) #1000
# #batch_size = int(2)
# train_iter = load_array((features_train, labels_train), batch_size)
# validate_iter = load_array((features_validate, labels_validate), int(features_validate.size()[0]/1))
# Test_iter = load_array((features_Test, labels_Test), int(features_Test.size()[0]/1))
# print("Dataset constructed")


class TestPrio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda:0"
        cls.num_data = 100000
        cls.feature_dim = 6
        cls.target_dim = 3
        cls.data = torch.randn(cls.num_data, cls.feature_dim, device=cls.device)
        cls.target = torch.randn(cls.num_data, cls.target_dim, device=cls.device)
        return

    def test_add_batch(self):
        prio_buf = PrioritizedReplayBuffer(100, 0.1, 0.1)
        for x, y in zip(self.data, self.target):
            prio_buf.add(Batch(x=x, y=y))

        batch_size = 10
        batch, indices = prio_buf.sample(batch_size)

        x_batch, y_batch = batch["x"], batch["y"]

        self.assertEqual(x_batch.shape[0], batch_size)
        priorities = torch.rand(batch_size, device=self.device)
        print(priorities)
        prio_buf.update_priority(indices, priorities)
        print(prio_buf.weight[indices])

        print(prio_buf.get_weight(indices))
        return
