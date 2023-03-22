import torch
import numpy as np


class ODEPhysWrapper(torch.nn.Module):
    def __init__(self, force_func):
        super(ODEPhysWrapper, self).__init__()
        self.m = 1.69884e-3
        self.J = 1e-6
        self.g = 9.8
        self.damping = 0.01
        self.register_buffer("coeffMatrix1", torch.zeros(6, 6).float())
        self.coeffMatrix1[3:, :3] = torch.eye(3).float()
        self.register_buffer("coeffMatrix2", torch.zeros(6, 6).float())
        self.coeffMatrix2[0:3, 3:] = -torch.diag(
            torch.tensor([1 / self.m, 1 / self.m, 1 / self.J])
        ).float()
        # self.register_buffer('coeffMatrix3',torch.zeros(6,6).float())
        # self.coeffMatrix3[3:,3:] = -self.damping * torch.diag(torch.tensor([1/self.m,1/self.m,1/self.J])).float()
        self.register_buffer("gVec", torch.tensor([0, 0, 0, 0, -self.g, 0]))
        self.ODEFunc = force_func

    def forward(self, t, y):
        grad = self.ODEFunc(y)
        # print(grad[...,1,1] / self.m)
        gravityGrad = torch.zeros_like(y)
        gravityGrad[..., 1:-1, :] = self.gVec * 1
        if grad is not None:
            # the dimensions of the return value are (num_samples, num_cycles, 6)
            return (
                torch.matmul(y, self.coeffMatrix1)
                + torch.matmul(
                    torch.cat((grad, torch.zeros_like(grad)), -1), self.coeffMatrix2
                )
                + gravityGrad
            )
