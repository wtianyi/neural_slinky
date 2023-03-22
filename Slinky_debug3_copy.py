import os
import argparse
import time
from matplotlib.pyplot import new_figure_manager
import numpy as np
# import jax.numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from adabelief_pytorch import AdaBelief
import matplotlib
import matplotlib.pyplot as plt
import copy

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

args.adjoint = True
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

# device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
device = torch.device("cuda:3")
# device = torch.device("cpu")
# args.viz = True
args.test_freq = 100
args.save_data = True 
args.save_freq = 10
args.do_test = False
# args.method = 'rk4'
# args.method = 'adams'
# args.method = 'fehlberg2'
args.method = 'dopri5'
args.batch_time = 1 + 1
args.data_size = 60 + 1
args.data_size_test = 60
args.batch_size = 5
args.deltaT_times = 1
args.niters = int(1e4)
args.atol = 1e-4
args.rtol = 1e-4
args.folder = 'NeuralODE_Share2/data_slinky2'

def readinData():
    folder = 'NeuralODE_Share2/SlinkyGroundTruth'
    true_y = np.loadtxt(folder+'/helixCoordinate_2D.txt',delimiter=',')
    true_v = np.loadtxt(folder+'/helixVelocity_2D.txt',delimiter=',')
    true_y = torch.from_numpy(true_y).float()
    true_v = torch.from_numpy(true_v).float()
    true_y = torch.reshape(true_y,(true_y.size()[0],80,3))
    true_v = torch.reshape(true_v,(true_v.size()[0],80,3))
    # print(true_v.shape)
    return torch.cat((true_y, true_v),-1)

true_y = readinData().to(device)
true_y = true_y[::10,...]
true_y_all = true_y.clone()
true_y = true_y[0:]
args.data_size_all = true_y.size()[0] # the length of the entire data

true_y0 = true_y[0,:].to(device)
true_y0_test = true_y[0,:].to(device)
true_y_test = true_y[0:args.data_size_test,...].to(device)
true_y = true_y[0:args.data_size,...].to(device)
deltaT = 0.01
deltaT_times = deltaT / args.deltaT_times
t = torch.linspace(0.,(true_y.size()[0]-1)*deltaT,true_y.size()[0] * args.deltaT_times).to(device)
t_test = torch.linspace(0.,(args.data_size_test-1)*deltaT,args.data_size_test).to(device)

args.data_size = args.data_size_all

def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False)) # M = num_batch_size
    batch_y0 = true_y_all[s]  # (M, D=[num_Cycles,6])
    batch_t = t[:args.batch_time]  # (T)
    # print(true_y_all.shape)
    # print(s)
    # print(args.batch_time)
    batch_y = torch.stack([true_y_all[s + i,...] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def removeRigidBodyMotion(data):
    new_data = data.clone()

    new_data[...,0] = torch.sqrt(torch.square(data[...,0]-data[...,3]) + torch.square(data[...,1]-data[...,4])) # l1
    new_data[...,1] = torch.sqrt(torch.square(data[...,3]-data[...,6]) + torch.square(data[...,4]-data[...,7])) # l2
    theta_1 = torch.atan2(-(data[...,3]-data[...,0]),data[...,4]-data[...,1])
    theta_2 = torch.atan2(-(data[...,6]-data[...,3]),data[...,7]-data[...,4])
    new_data[...,2] = theta_2 - theta_1 # theta
    new_data[...,3] = data[...,2] - data[...,5] # gamma1
    theta_3 = torch.atan2(-(data[...,0]-data[...,3]),data[...,1]-data[...,4])
    new_data[...,4] = (theta_3 + theta_2) / 2 - data[...,5] # gamma2
    new_data[...,5] = data[...,8] - data[...,5] # gamma3

    # new_data (relative coordinates should be 6 dimensional)
    new_data = new_data[...,:6]
    return new_data

def chiral_transformation_x(data):
    new_data = data.clone()
    new_data[...,2:] = -data[...,2:]
    return new_data

def chiral_transformation_z(data):
    new_data = data.clone()
    new_data[...,0] = data[...,1]
    new_data[...,1] = data[...,0]
    new_data[...,3] = -data[...,5]
    new_data[...,4] = -data[...,4]
    new_data[...,5] = -data[...,3]
    return new_data

def chiral_transformation_xz(data):
    new_data = chiral_transformation_x(data.clone())
    new_data = chiral_transformation_z(new_data)
    return new_data

def cal_data_statistics(data):
    yp = data[...,1:79,0:3]
    ypp = data[...,0:78,0:3]
    ypa = data[...,2:,0:3]
    yinput = torch.cat((ypp,yp,ypa),-1) # N x num_cycles x 9

    # change the yinput to N x num_cycles x 6
    yinput = removeRigidBodyMotion(yinput)
    yinput_x = chiral_transformation_x(yinput.clone())
    yinput_z = chiral_transformation_z(yinput.clone())
    yinput_xz = chiral_transformation_xz(yinput.clone())
    yinput = torch.cat((yinput,yinput_x,yinput_z,yinput_xz),-2)
    data_mean, data_std = torch.mean(yinput,(0,1)), torch.std(yinput,(0,1))
    return data_mean, data_std

data_mean_6, data_std_6 = cal_data_statistics(true_y)
# print(data_mean_6)
# print(data_std_6)

if args.viz:
    makedirs('png')
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_traj2 = fig.add_subplot(132, frameon=False)
    ax_phase = fig.add_subplot(133, frameon=False)
    # ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)

if args.save_data:
    makedirs(args.folder)

def visualize(true_y, pred_y, t, odefunc, itr):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories: displacement')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], 'g-')
        ax_traj.plot(t.cpu().numpy(), pred_y.detach().cpu().numpy()[:, 0, 0], 'b--')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.set_ylim(-3, 3)
        ax_traj.legend()

        ax_traj2.cla()
        ax_traj2.set_title('Trajectories: velocity')
        ax_traj2.set_xlabel('t')
        ax_traj2.set_ylabel('x,y')
        ax_traj2.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_traj2.plot(t.cpu().numpy(), pred_y.detach().cpu().numpy()[:, 0, 1], 'b--')
        ax_traj2.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj2.set_ylim(-10, 10)
        ax_traj2.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.detach().cpu().numpy()[:, 0, 0], pred_y.detach().cpu().numpy()[:, 0, 1], 'b--')
        ax_phase.set_xlim(-3, 3)
        ax_phase.set_ylim(-10, 10)

        # ax_vecfield.cla()
        # ax_vecfield.set_title('Learned Vector Field')
        # ax_vecfield.set_xlabel('x')
        # ax_vecfield.set_ylabel('y')

        # y, x = np.mgrid[-2:2:21j, -2:2:21j]
        # dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
        # mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        # dydt = (dydt / mag)
        # dydt = dydt.reshape(21, 21, 2)

        # ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        # ax_vecfield.set_xlim(-2, 2)
        # ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)

def savedata(true_y, pred_y, t, odefunc, itr, append_name):

    if args.save_data:

        f_true_y = './'+ args.folder + '/true_'+str(itr)+'_'+append_name+'.txt'
        f_pred_y = './'+ args.folder + '/pred_'+str(itr)+'_'+append_name+'.txt'
        f_t = './'+ args.folder + '/t_'+str(itr)+'_'+append_name+'.txt'
        true_y_record = true_y.detach().cpu().numpy().transpose(1,2,0)
        pred_y_record = pred_y.detach().cpu().numpy().transpose(1,2,0)
        # print(true_y_record.shape)
        # print(pred_y_record.shape)

        np.savetxt(f_true_y, true_y_record.reshape((-1,true_y_record.shape[2])).transpose(), delimiter=",")
        np.savetxt(f_pred_y, pred_y_record.reshape((-1,pred_y_record.shape[2])).transpose(), delimiter=",")
        np.savetxt(f_t, np.squeeze(t.detach().cpu().numpy()), delimiter=",")

class Square(nn.Module):
    def forward(self,x):
        return torch.square(x)
        # return torch.abs(x)**4

class SquareLinear(nn.Module):
    def __init__(self, InputDim, OutputDim):
        super(SquareLinear, self).__init__()
        self.params = torch.nn.Parameter(0.10*torch.rand((InputDim, OutputDim),requires_grad=True))
        # self.params = torch.nn.Parameter(torch.ones((OutputDim, InputDim),requires_grad=True))
        # print(InputDim)
        # print(OutputDim)

    def forward(self,x):
        return torch.matmul(x,self.params**2)

class AbsLinear(nn.Module):
    def __init__(self, InputDim, OutputDim):
        super(AbsLinear, self).__init__()
        self.params = torch.nn.Parameter(0.10*torch.rand((InputDim, OutputDim),requires_grad=True))
        # self.params = torch.nn.Parameter(torch.ones((OutputDim, InputDim),requires_grad=True))
        # print(InputDim)
        # print(OutputDim)

    def forward(self,x):
        # print(x)
        return torch.matmul(x,torch.abs(self.params))

class SymSoftplus(nn.Module):
    def __init__(self):
        super(SymSoftplus, self).__init__()
        self.softplus = nn.Softplus()

    def forward(self,x):
        return 0.5 * (self.softplus(x) + self.softplus(-x))

class MLPBlock(nn.Module):
    def __init__(self, NeuronsPerLayer, NumLayer):
        super(MLPBlock, self).__init__()
        layer = []
        for i in range(NumLayer):
            layer.append(
                nn.Sequential(
                nn.Linear(NeuronsPerLayer, NeuronsPerLayer), 
                # nn.Softplus(beta=1e1)
                nn.Tanh()
                )
                )
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            X = blk(X)
        return X

class DenseBlock(nn.Module):
    def __init__(self, NeuronsPerLayer, NumLayer):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(NumLayer):
            layer.append(
                nn.Sequential(
                nn.Linear(NeuronsPerLayer * i + NeuronsPerLayer, NeuronsPerLayer), 
                # Square()
                nn.Softplus(beta=1e1)
                # nn.Tanh()
                )
                )
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate the input and output of each block on the channel
            X = torch.cat((X, Y), dim=-1)
        return X

class DenseBlock_SquareLinear(nn.Module):
    def __init__(self, NeuronsPerLayer, NumLayer):
        super(DenseBlock_SquareLinear, self).__init__()
        layer = []
        # self.softplus = nn.Softplus
        for i in range(NumLayer):
            layer.append(
                nn.Sequential(
                # SquareLinear(NeuronsPerLayer, NeuronsPerLayer * i + NeuronsPerLayer), 
                # SquareLinear(NeuronsPerLayer * i + NeuronsPerLayer, NeuronsPerLayer), 
                AbsLinear(NeuronsPerLayer * i + NeuronsPerLayer, NeuronsPerLayer),
                # SymSoftplus()
                # Square()
                nn.Softplus(beta=1e1)
                # nn.Tanh()
                )
                )
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate the input and output of each block on the channel
            X = torch.cat((X, Y), dim=-1)
        return X

class DenseBlock_AbsLinear(nn.Module):
    def __init__(self, NeuronsPerLayer, NumLayer):
        super(DenseBlock_AbsLinear, self).__init__()
        layer = []
        # self.softplus = nn.Softplus
        for i in range(NumLayer):
            layer.append(
                nn.Sequential(
                # SquareLinear(NeuronsPerLayer, NeuronsPerLayer * i + NeuronsPerLayer), 
                # SquareLinear(NeuronsPerLayer * i + NeuronsPerLayer, NeuronsPerLayer), 
                AbsLinear(NeuronsPerLayer * i + NeuronsPerLayer, NeuronsPerLayer),
                # SymSoftplus()
                # Square()
                nn.Softplus(beta=1e1)
                # nn.Tanh()
                )
                )
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            # print(X)
            Y = blk(X)
            # Concatenate the input and output of each block on the channel
            X = torch.cat((X, Y), dim=-1)
        return X

# incorporating invariance version
class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()
        self.neuronsPerLayer = int(16)
        self.numLayers = 1
        self.net = nn.Sequential(
            # SquareLinear(6,self.neuronsPerLayer),
            # DenseBlock_SquareLinear(self.neuronsPerLayer,self.numLayers), Square(),
            # SquareLinear(int(self.neuronsPerLayer*(self.numLayers+1)),1)
            AbsLinear(6,self.neuronsPerLayer),
            DenseBlock_AbsLinear(self.neuronsPerLayer,self.numLayers), 
            Square(),
            # AbsLinear(self.neuronsPerLayer,1)
            AbsLinear(int(self.neuronsPerLayer*(self.numLayers+1)),1)
        )

        # self.net5 = nn.Sequential(
        #     nn.Linear(6, self.neuronsPerLayer), #nn.Softplus(beta=1e1),
        #     DenseBlock(self.neuronsPerLayer,self.numLayers),# Square(),
        #     nn.Linear(int(self.neuronsPerLayer*(self.numLayers+1)),1)
        # )

        self.net5 = nn.Sequential(
            nn.Linear(6, self.neuronsPerLayer), nn.Tanh(),# nn.Softplus(beta=1e1),
            MLPBlock(self.neuronsPerLayer, self.numLayers),
            nn.Linear(self.neuronsPerLayer, 1)#, Square()
        )
        # self.normal_ = nn.BatchNorm1d(78)

        self.net2 = nn.Sequential(
            nn.Linear(9, self.neuronsPerLayer),
            # nn.Softplus(beta=1e1),
            # nn.Linear(64, 64),
            # nn.Softplus(beta=1e1),
            # nn.Linear(64, 64),
            MLPBlock(self.neuronsPerLayer, self.neuronsPerLayer),
            nn.Linear(self.neuronsPerLayer, 3),
        )

        self.net3 = nn.Sequential(
            DenseBlock(6,self.numLayers), nn.Softplus(),
            # nn.Softplus(beta=1e1),
            # nn.Linear(64, 64),
            nn.Linear(int(6*(self.numLayers+1)),1)
        )
        
        self.net4 = nn.Sequential(
            nn.Linear(6,1)
        )

        self.net11 = nn.Sequential(
            nn.Linear(2, self.neuronsPerLayer),
            DenseBlock(self.neuronsPerLayer,self.numLayers), Square(),
            nn.Linear(int(self.neuronsPerLayer*(self.numLayers+1)),1)
        )

        self.net22 = nn.Sequential(
            nn.Linear(2, self.neuronsPerLayer),
            DenseBlock(self.neuronsPerLayer,self.numLayers), Square(),
            nn.Linear(int(self.neuronsPerLayer*(self.numLayers+1)),1)
        )


        self.net33 = nn.Sequential(
            nn.Linear(2, self.neuronsPerLayer),
            DenseBlock(self.neuronsPerLayer,self.numLayers), Square(),
            nn.Linear(int(self.neuronsPerLayer*(self.numLayers+1)),1)
        )


        self.m = 1.69884e-3
        self.J = 1e-6
        self.g = 9.8
        self.register_buffer('coeffMatrix1',torch.zeros(6,6).float())
        self.coeffMatrix1[3:,:3] = torch.eye(3).float()
        self.register_buffer('coeffMatrix2',torch.zeros(6,6).float())
        self.coeffMatrix2[0:3,3:] = -torch.diag(torch.tensor([1/self.m,1/self.m,1/self.J])).float()
        self.register_buffer('gVec',torch.tensor([0,0,0,0,-self.g,0]))

        # self.register_buffer('ka',torch.ones(1).float())
        # self.register_buffer('ks',torch.ones(1).float())
        # self.register_buffer('kr',torch.ones(1).float())

        self.ka = torch.nn.Parameter(torch.ones((1),requires_grad=True))
        self.ks = torch.nn.Parameter(torch.ones((1),requires_grad=True))
        self.kr = torch.nn.Parameter(torch.ones((1),requires_grad=True))
        # self.ka = torch.nn.Parameter(24.4**0.5*torch.ones((1),requires_grad=False)).to(device)
        # self.ks = torch.nn.Parameter(76.6**0.5*torch.ones((1),requires_grad=False)).to(device)
        # self.kr = torch.nn.Parameter(0.03506**0.5*torch.ones((1),requires_grad=False)).to(device)
        self.ka.to(device)
        self.ks.to(device)
        self.kr.to(device)

        # self.weight6 = torch.nn.Parameter(torch.tensor([24.4, 76.6, 0.03506, 24.4, 76.6, 0.03506],requires_grad=True) * 1.0)
        self.weight6 = torch.nn.Parameter(torch.tensor([0, 76.6, 0.03506, 0, 76.6, 0.03506],requires_grad=False) * 1.0)
        # self.weight6 = torch.nn.Parameter(torch.tensor([0.01,0.01,0.01,0.01,0.01,0.01],requires_grad=True))
        self.weight6.to(device)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight, mean=0, std=0.01)
                # nn.init.constant_(m.bias, val=0.01)
                nn.init.kaiming_normal_(m.weight)

        for m in self.net2.modules():
            if isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight, mean=0, std=0.01)
                # nn.init.constant_(m.bias, val=0.01)
                nn.init.kaiming_normal_(m.weight)

        for m in self.net3.modules():
            if isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight, mean=0, std=0.01)
                # nn.init.constant_(m.bias, val=0.01)
                nn.init.kaiming_normal_(m.weight)

        for m in self.net5.modules():
            if isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight, mean=0, std=0.01)
                # nn.init.constant_(m.bias, val=0.01)
                nn.init.xavier_normal_(m.weight)
                # nn.init.kaiming_normal_(m.weight)

    def calDeriv(self, y):
        with torch.enable_grad():
            # calculating the acceleration of the 2D slinky system
            # the dimensions of y are (num_samples, num_cycles, [x_dis,y_dis,alpha_dis,x_vel,y_vel,alpha_vel])
            # y.requires_grad_(True)
            # yp = y[...,1:-1,0:3]
            # ypp = y[...,0:-2,0:3]
            # ypa = y[...,2:,0:3]

            yp = y[...,1:-1,0:3].clone().requires_grad_(True)
            ypp = y[...,0:-2,0:3].clone()
            ypa = y[...,2:,0:3].clone()

            yinput = torch.cat((ypp,yp,ypa),-1)
            x = self.removeRigidBodyMotion(yinput) #* torch.tensor([100,100,1,1,1,1]).view(1,6)
            x = (x-data_mean_6) / data_std_6
            # x = self.region_shifting(x)
            # print(x.shape)
            # x = torch.transpose(self.normal_(torch.transpose(x,0,1)),0,1)

            # incorporate chirality
            augmented_x = torch.stack([x, self.chiral_transformation_x(x), self.chiral_transformation_z(x), self.chiral_transformation_xz(x)], dim=0)
            out = self.net(augmented_x)
            out = torch.sum(out, dim=0, keepdim=False)
            # out = self.net(x)

            deriv = torch.autograd.grad([out.sum()],[yp],retain_graph=True,create_graph=True)
            # the dimensions of grad are (num_samples, num_cycles, 3)
            # grad = deriv[0][...,1:-1,0:3]
            grad = deriv[0]# * 1e2
            # print(torch.norm(grad[...,0,0:3]))
            # grad_partial = grad[...,1:-1,0:3]
            # print(torch.norm(grad))
            # print(grad.shape)
            aug = torch.zeros_like(grad)[...,0:1,:]
            if grad is not None:
                # print(torch.mean(torch.abs(grad[...,0:1])))
                return torch.cat((aug,grad,aug),-2)

    def calDeriv2(self, y):
        # calculating the acceleration of the 2D slinky system
        # the dimensions of y are (num_samples, num_cycles, [x_dis,y_dis,alpha_dis,x_vel,y_vel,alpha_vel])
        yp = y[...,1:-1,0:3]
        ypp = y[...,0:-2,0:3]
        ypa = y[...,2:,0:3]
        yinput = torch.cat((ypp,yp,ypa),-1)
        grad = self.net2(yinput)
        aug = torch.zeros_like(grad)[...,0:1,:]
        if grad is not None:
            # print(torch.mean(torch.abs(grad[...,0:1])))
            return torch.cat((aug,grad,aug),-2)

    def calDeriv3(self, y):
        print("input shape", y.shape)
        with torch.enable_grad():
            # calculating the acceleration of the 2D slinky system
            # the dimensions of y are (num_samples, num_cycles, [x_dis,y_dis,alpha_dis,x_vel,y_vel,alpha_vel])
            yp = y[...,1:-1,0:3].requires_grad_(True)
            ypp = y[...,0:-2,0:3]
            ypa = y[...,2:,0:3]

            yinput = torch.cat((ypp,yp,ypa),-1)
            x = self.Douglas_transformation(yinput) # N x 78 x 6
            # augmented_x = torch.stack([x, self.chiral_transformation_x_Douglas(x), self.chiral_transformation_z_Douglas(x), self.chiral_transformation_xz_Douglas(x)], dim=0)
            # out = self.net(augmented_x) 
            # out = torch.sum(out, dim=0, keepdim=False)
            x_clone = x.clone().detach().requires_grad_(False)
            # x_clone = copy.deepcopy(x)
            # out = self.net(x)


            # out = 24.409942 * (torch.square(x[...,0]+0.001894637848791/2) + torch.square(x[...,3]+0.001894637848791/2)) + 76.641454 * (torch.square(x[...,1]) + torch.square(x[...,4])) + 0.035065 * (torch.square(x[...,2]) + torch.square(x[...,5]))
            # out = 5.3793 * (torch.square(x[...,0]+0.001894637848791/2) + torch.square(x[...,3]+0.001894637848791/2)) + 9.0090 * (torch.square(x[...,1]) + torch.square(x[...,4])) + 0.1785 * (torch.square(x[...,2]) + torch.square(x[...,5]))
            # out = 0.5 * self.ka**2 * (torch.square(x[...,0]+0.001894637848791/2) + torch.square(x[...,3]+0.001894637848791/2)) + 0.5 * self.ks**2 * (torch.square(x[...,1]) + torch.square(x[...,4])) + 0.5 * self.kr**2 * (torch.square(x[...,2]) + torch.square(x[...,5]))
            # out = 0.5 * self.ka**2 * (torch.square(x[...,0]) + torch.square(x[...,3])) + 0.5 * self.ks**2 * (torch.square(x[...,1]) + torch.square(x[...,4])) + 0.5 * self.kr**2 * (torch.square(x[...,2]) + torch.square(x[...,5]))
            # out = 0.5 * self.ka**2 * 100 * (torch.square(x[...,0]) + torch.square(x[...,3])) + 0.5 * self.ks**2 * 100 * (torch.square(x[...,1]) + torch.square(x[...,4])) + 0.5 * self.kr**2 * (torch.square(x[...,2]) + torch.square(x[...,5]))
            out = 0.5 * torch.abs(torch.squeeze(self.net5(x_clone))) * (torch.square(x[...,0]) + torch.square(x[...,3])) + 0.5 * self.ks**2 * 100 * (torch.square(x[...,1]) + torch.square(x[...,4])) + 0.5 * self.kr**2 * (torch.square(x[...,2]) + torch.square(x[...,5]))
            # out = 0.5 * self.ka**2 * (torch.square(x[...,0]) + torch.square(x[...,3])) + 0.5 * self.ks**2 * 100 * (torch.square(x[...,1]) + torch.square(x[...,4])) + 0.5 * torch.abs(torch.squeeze(self.net5(x_clone))) * (torch.square(x[...,2]) + torch.square(x[...,5]))
            out = 0.5 * self.ka**2 * (torch.square(x[...,0]) + torch.square(x[...,3])) + 0.5 * torch.abs(torch.squeeze(self.net5(x_clone))) * (torch.square(x[...,1]) + torch.square(x[...,4])) + 0.5 * self.kr**2 * (torch.square(x[...,2]) + torch.square(x[...,5]))
            # print(torch.mean(torch.abs(x_clone)))
            if False:
                print("the mean of x is: {}".format(torch.mean(torch.abs(x_clone[...,[0,3]]))))
                print("the mean of y is: {}".format(torch.mean(torch.abs(x_clone[...,[1,4]]))))
                print("the mean of theta is: {}".format(torch.mean(torch.abs(x_clone[...,[2,5]]))))
                # print("the value of kr is: {}".format(self.kr**2))
                print(torch.mean(torch.abs(self.net5(x_clone))))
                print(torch.std(torch.abs(self.net5(x_clone))))
            # print(x_clone.requires_grad)
            # print(out.shape)
            # print(torch.squeeze(self.net(x_clone).shape))
            # input()
            # out = out.view(-1,1)
            # out += self.net(x)
            # print(out)
            # print(torch.mean(torch.abs(out)))
            # input()

            deriv = torch.autograd.grad([out.sum()],[yp],retain_graph=True,create_graph=True)
            # the dimensions of grad are (num_samples, num_cycles, 3)
            # grad = deriv[0][...,1:-1,0:3]
            grad = deriv[0]
            # print(grad.shape)
            # print(grad_partial[...,0,:])
            # print(grad_partial[...,1,:])
            # print(grad_partial.shape)
            # print(torch.norm(grad))
            # print(grad.shape)
            aug = torch.zeros_like(grad)[...,0:1,:]
            if grad is not None:
                # print(torch.mean(torch.abs(grad[...,0:1])))
                print("grad shape", grad.shape)
                print("result shape", torch.cat((aug,grad,aug),-2).shape)
                return torch.cat((aug,grad,aug),-2)

    def calDeriv4(self, y):
        with torch.enable_grad():
            # calculating the acceleration of the 2D slinky system
            # the dimensions of y are (num_samples, num_cycles, [x_dis,y_dis,alpha_dis,x_vel,y_vel,alpha_vel])
            # y.requires_grad_(True)
            # yp = y[...,1:-1,0:3].clone().requires_grad_(True)
            # ypp = y[...,0:-2,0:3].clone().requires_grad_(True)
            # ypa = y[...,2:,0:3].clone().requires_grad_(True)

            yp = y[...,1:-1,0:3].clone().requires_grad_(True)
            ypp = y[...,0:-2,0:3].clone()
            ypa = y[...,2:,0:3].clone()

            yinput = torch.cat((ypp,yp,ypa),-1)
            x = self.Douglas_transformation(yinput)
            # x = x * torch.tensor([1e3,1e3,3e1,1e3,1e3,3e1]).to(device)
            # print(yinput[2,2,:])
            # print(x[2,2,:])
            # x = torch.cat((x,x**2),-1)
            # x = (x-data_mean_6) / data_std_6
            # x = self.region_shifting(x)
            # augmented_x = torch.stack([x, self.chiral_transformation_x_Douglas(x), self.chiral_transformation_z_Douglas(x), self.chiral_transformation_xz_Douglas(x)], dim=0)
            # out = self.net(augmented_x)
            # out = torch.sum(out, dim=0, keepdim=False)
            # out = self.net5(x*1e0) * 0.000874
            out = self.net5(x)

            # print(out[2,2,:])
            # input()
            # out = self.net4(torch.cat((x,x**2),-1))
            # out = self.net4(x**2)
            # out = x**2 * self.weight6 + self.net(x[...,[0,3]]*1e3)
            # out = self.net11(x[...,[0,3]]) + self.net22(x[...,[1,4]]) + self.net33(x[...,[2,5]])

            deriv = torch.autograd.grad([out.sum()],[yp],retain_graph=True,create_graph=True)
            # the dimensions of grad are (num_samples, num_cycles, 3)
            # grad = deriv[0][...,1:-1,0:3]
            grad = deriv[0]#*1e3
            # print(grad[...,0,:])
            # print(grad[...,1,:])
            # print(torch.norm(grad))
            # print(grad.shape)
            aug = torch.zeros_like(grad)[...,0:1,:]
            if grad is not None:
                # print(torch.mean(torch.abs(grad[...,0:1])))
                return torch.cat((aug,grad,aug),-2)

    def removeRigidBodyMotion(self, data):
        new_data = data.clone()

        new_data[...,0] = torch.sqrt(torch.square(data[...,0]-data[...,3]) + torch.square(data[...,1]-data[...,4])) # l1
        new_data[...,1] = torch.sqrt(torch.square(data[...,3]-data[...,6]) + torch.square(data[...,4]-data[...,7])) # l2
        theta_1 = torch.atan2(-(data[...,3]-data[...,0]),data[...,4]-data[...,1])
        theta_2 = torch.atan2(-(data[...,6]-data[...,3]),data[...,7]-data[...,4])
        new_data[...,2] = theta_2 - theta_1 # theta
        new_data[...,3] = data[...,2] - data[...,5] # gamma1
        theta_3 = torch.atan2(-(data[...,0]-data[...,3]),data[...,1]-data[...,4])
        new_data[...,4] = (theta_3 + theta_2) / 2 - data[...,5] # gamma2
        new_data[...,5] = data[...,8] - data[...,5] # gamma3

        # new_data (relative coordinates should be 6 dimensional)
        new_data = new_data[...,:6]
        return new_data

    def chiral_transformation_x(self, data):
        new_data = data.clone()
        new_data[...,2:] = -data[...,2:]
        return new_data

    def chiral_transformation_z(self, data):
        new_data = data.clone()
        new_data[...,0] = data[...,1]
        new_data[...,1] = data[...,0]
        new_data[...,3] = -data[...,5]
        new_data[...,4] = -data[...,4]
        new_data[...,5] = -data[...,3]
        return new_data

    def chiral_transformation_xz(self, data):
        new_data = self.chiral_transformation_x(data.clone())
        new_data = self.chiral_transformation_z(new_data)
        return new_data

    def region_shifting(self, data):
        new_data = data.clone()
        int_4 = torch.ceil(-0.5-data[...,4] / math.pi)
        new_data[...,4] = data[...,4] + int_4 * math.pi 
        return new_data

    def Douglas_transformation(self, data):
        BA1 = (data[...,2] + data[...,5])/2
        BA2 = (data[...,5] + data[...,8])/2
        
        col1 = torch.cos(BA1) * (data[...,3]-data[...,0]) + torch.sin(BA1) * (data[...,4]-data[...,1]) # xi_1
        col2 = -torch.sin(BA1) * (data[...,3]-data[...,0]) + torch.cos(BA1) * (data[...,4]-data[...,1]) # eta_1
        col3 = data[...,5] - data[...,2] # theta_1
        col4 = torch.cos(BA2) * (data[...,6]-data[...,3]) + torch.sin(BA2) * (data[...,7]-data[...,4]) # xi_2
        col5 = -torch.sin(BA2) * (data[...,6]-data[...,3]) + torch.cos(BA2) * (data[...,7]-data[...,4]) # eta_2
        col6 = data[...,8] - data[...,5] # theta_2
        return torch.stack((col1,col2,col3,col4,col5,col6),dim=-1)

    def chiral_transformation_x_Douglas(self, data):
        new_data = data.clone()
        new_data[...,0] = -data[...,0]
        new_data[...,2] = -data[...,2]
        new_data[...,3] = -data[...,3]
        new_data[...,5] = -data[...,5]
        return new_data

    def chiral_transformation_z_Douglas(self, data):
        new_data = data.clone()
        new_data[...,0] = data[...,3]
        new_data[...,1] = -data[...,4]
        new_data[...,2] = data[...,5]
        new_data[...,3] = data[...,0]
        new_data[...,4] = -data[...,1]
        new_data[...,5] = data[...,2]
        return new_data

    def chiral_transformation_xz_Douglas(self, data):
        new_data = data.clone()
        new_data = self.chiral_transformation_x_Douglas(data)
        new_data = self.chiral_transformation_z_Douglas(new_data)
        return new_data

    def forward(self, y):
        # grad = self.calDeriv(y)
        # grad = self.calDeriv2(y)
        grad = self.calDeriv3(y)
        # grad = self.calDeriv4(y)
        # print(torch.norm(grad))
        # print(grad.shape)
        # print("after calculating grad")
        # gravityGrad = torch.zeros_like(y)
        # gravityGrad[...,1:-1,:] += self.gVec
        if grad is not None:
            # the dimensions of the return value are (num_samples, num_cycles, 6)
            return grad
            # return torch.matmul(y,self.coeffMatrix1) + torch.matmul(torch.cat((grad,torch.zeros_like(grad)),-1),self.coeffMatrix2) + gravityGrad

from neural_slinky import coords_transform
from neural_slinky import e3nn_models
class E3nnForce(nn.Module):
    def __init__(self):
        super(E3nnForce, self).__init__()
        self.e3nn = e3nn_models.SlinkyForcePredictorCartesian(
        irreps_node_input="0e",
        irreps_node_attr="2x0e",
        irreps_edge_attr="2x0e",
        irreps_node_output="1x1o",
        max_radius=0.06,
        num_neighbors=1,
        num_nodes=2,
        mul=50,
        layers=6,
        lmax=2,
        pool_nodes=False
    )

    def forward(self, y):
        if len(y.shape) < 3:
            y = y[None, ...]
        B, L, _ = y.shape
        print("y.shape", y.shape)
        cartesian = coords_transform.transform_triplet_to_cartesian(
            coords_transform.transform_cartesian_alpha_to_triplet(
                coords_transform.group_coords(y[..., 0:3]).flatten(start_dim=-2) # (B x (L-2) x 9)
            ), # (B x (L-2) x 6)
            [0.1, 0.1, 0.1]
        ).flatten(end_dim=1) # (N x 9 x 2), N = B x (L-2)
        # print("cartesian.shape", cartesian.shape)

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
        output = self.e3nn(data) # (9N x 3)
        # print("output.shape", output.shape)
        output = output.reshape(B, (L-2), 3, 3, 3)
        cartesian = cartesian.reshape(B, (L-2), 3, 3, 2)
        result = y.new_zeros(B, L, 3)
        result[:, 1:-1, 0:2] = output.sum(dim=-2)[..., 1, 0:2]
        tmp = coords_transform.cross2d(cartesian[..., 0, :] - cartesian[..., 1, :], output[..., 0, 0:2]) + coords_transform.cross2d(cartesian[..., 2, :] - cartesian[..., 1, :], output[..., 2, 0:2])
        # print("tmp.shape", tmp.shape)
        result[:, 1:-1, 2] = tmp[..., 1]
        return result.squeeze(0)


class ODEPhys(nn.Module):

    def __init__(self, ODEFunc):
        super(ODEPhys, self).__init__()
        self.m = 1.69884e-3
        self.J = 1e-6
        self.g = 9.8
        self.damping = 0.01
        self.register_buffer('coeffMatrix1',torch.zeros(6,6).float())
        self.coeffMatrix1[3:,:3] = torch.eye(3).float()
        self.register_buffer('coeffMatrix2',torch.zeros(6,6).float())
        self.coeffMatrix2[0:3,3:] = -torch.diag(torch.tensor([1/self.m,1/self.m,1/self.J])).float()
        # self.register_buffer('coeffMatrix3',torch.zeros(6,6).float())
        # self.coeffMatrix3[3:,3:] = -self.damping * torch.diag(torch.tensor([1/self.m,1/self.m,1/self.J])).float()
        self.register_buffer('gVec',torch.tensor([0,0,0,0,-self.g,0]))
        self.ODEFunc = ODEFunc

    def forward(self, t, y):
        grad = self.ODEFunc(y)
        # print(grad[...,1,1] / self.m)
        gravityGrad = torch.zeros_like(y)
        gravityGrad[...,1:-1,:] = self.gVec * 1
        if grad is not None:
            # the dimensions of the return value are (num_samples, num_cycles, 6)
            return torch.matmul(y,self.coeffMatrix1) + torch.matmul(torch.cat((grad,torch.zeros_like(grad)),-1),self.coeffMatrix2) + gravityGrad

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    seed_torch()

    ii = 0

    # func_orig = ODEFunc().to(device)
    func_orig = E3nnForce().to(device)
    func = ODEPhys(func_orig).to(device)
    
    # optimizer = optim.RMSprop(func.parameters(), lr=1e-4)
    # optimizer = optim.Adam(func.parameters(), lr=1e-1, weight_decay = 1e-2)
    optimizer = optim.Adam(func.parameters(), lr=1e-3)
    # optimizer = AdaBelief(func.parameters(), lr=1e-2, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=50,verbose=True)

    for itr in range(1, args.niters + 1):
        # for name, params in func.ODEFunc.net.named_parameters():
            # print(params)
        #     params.clamp_(min=0)

        # print(str(itr) + ": before the iteration")
        optimizer.zero_grad()
        # print(func_orig.ka,func_orig.ks,func_orig.kr) 

        batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0, batch_t, atol=args.atol, rtol=args.rtol).to(device)
        select_cycles = torch.from_numpy(np.random.choice(np.arange(80, dtype=np.int64), 10, replace=False))
        weight6 = torch.zeros(6).to(device)
        # for kk in range(6):
            # weight6[kk] = torch.norm(pred_y[...,-1,1,kk] - batch_y[...,-1,1,kk],p=2)
        weight6 = torch.tensor([1e2,1e2,1,1e2,1e2,1]) ** 2
        weight6 = weight6.to(device)
        loss = torch.mean(torch.abs(pred_y[...,select_cycles,:] - batch_y[...,select_cycles,:]) **2 * weight6)
        # loss = torch.mean(torch.abs(pred_y[...,-1,select_cycles,:] - batch_y[...,-1,select_cycles,:])**2)
        # loss = torch.mean(torch.abs(pred_y[...,-1,1,:] - batch_y[...,-1,1,:])**2 * weight6)

        # loss = torch.mean(torch.abs(pred_y - batch_y))
        # loss = torch.mean(torch.abs(pred_y[::args.deltaT_times] - true_y))

        # train on testset
        # tic = time.perf_counter()
        # pred_y = odeint(func, true_y0, t, atol=args.atol, rtol=args.rtol)
        # toc = time.perf_counter()
        # print(f" forward propagation takes {toc - tic:0.4f} seconds")
        # loss = torch.mean(torch.abs(pred_y[::args.deltaT_times,0:3] - true_y[...,0:3]))

        # regularization_loss = 0
        # for param in func.parameters():
            # regularization_loss += torch.sum(abs(param))

        # loss += 1e-3 * regularization_loss

        print('Iter {:04d} | Training Batch Loss {:.6f}'.format(itr, loss.item()))
        # print(func.weight6)
        # print(func.ka,func.ks,func.kr)
        # loss = torch.max(torch.square(pred_y[...,0:2] - batch_y[...,0:2]))

        tic = time.perf_counter()
        loss.backward()
        toc = time.perf_counter()
        # print(f" backward propagation takes {toc - tic:0.4f} seconds")

        optimizer.step()
        # scheduler.step(loss)

        if True and itr % 10 == 0 and args.batch_time < 50:
            # true_y_train = true_y[:,0:args.data_size+int(itr/10),...].to(device)
            # t = torch.linspace(0.,(true_y_train.size()[1]-1)*deltaT,true_y_train.size()[1] * args.deltaT_times).to(device)
            # print('increased the train sequence length')
            # print('The current lenght is {:04d}'.format(true_y_train.size()[1]))
            args.batch_time += 1#int(itr/10)
            print('increased the batch sequence length')
            print('The current lenght is {:04d}'.format(args.batch_time))

        if itr % args.test_freq == 0:
            if args.do_test:
                # with torch.no_grad():
                #     pred_y = odeint(func, true_y0, t)
                #     loss = torch.mean(torch.abs(pred_y - true_y))
                #     print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                #     visualize(true_y, pred_y, func, ii)
                #     ii += 1

                # with torch.no_grad():
                func.eval()
                pred_y = odeint(func, true_y0, t, atol=args.atol, rtol=args.rtol)
                # loss = torch.mean(torch.abs(pred_y[::args.deltaT_times] - true_y))
                print('Iter {:04d} | Training total Loss \033[1;32;43m {:.6f} \033[0m'.format(itr, loss.item()))
                # visualize(true_y, pred_y, t, func, ii)

                # pred_y_test = odeint(func, true_y0_test, t_test)
                # loss = torch.mean(torch.abs(pred_y_test - true_y_test))
                # print('Iter {:04d} | Test total Loss {:.6f}'.format(itr, loss.item()))
                # visualize(true_y_test, pred_y_test, t_test, func, ii)

                func.train()
                ii += 1

        if itr % args.save_freq == 0 and args.batch_time > 1:
            if args.save_data:
                # print('inside save_data')
                func.eval()
                pred_y = odeint(func, true_y0, t, atol=args.atol, rtol=args.rtol)
                loss = torch.mean(torch.abs(pred_y[::args.deltaT_times] - true_y))
                print('Iter {:04d} | Training total Loss \033[1;32;43m{:.6f}\033[0m'.format(itr, loss.item()))
                savedata(true_y, pred_y, t, func, itr, 'e3nn_train')

                # pred_y_test = odeint(func, true_y0_test, t_test, atol=args.atol, rtol=args.rtol)
                # loss = torch.mean(torch.abs(pred_y_test - true_y_test))
                # print('Iter {:04d} | Test total Loss {:.6f}'.format(itr, loss.item()))
                # savedata(true_y_test, pred_y_test, t_test, func, itr, 'test')
                func.train()
        # print(str(itr) + "after test")

    fig = plt.figure()
    x_test = torch.zeros(1000,3,6).to(device)
    for ii in range(1000):
        x_test[ii,1,0] = 0.004 
        x_test[ii,1,1] = -0.001 * ii
        x_test[ii,2,0] = 0.004 * 2
        x_test[ii,2,1] = -0.001 * ii
    predicted_force = func_orig(x_test) / 1.69884e-3
    plt.plot(x_test[...,1,1].cpu().detach().numpy(), predicted_force[...,1,1].cpu().detach().numpy())

    # traced_script_module = torch.jit.script(func.to("cpu"))
    # traced_script_module = traced_script_module.to("cpu")
    # traced_script_module.save("./traced_massspring.pt")
    # func.to(device)
