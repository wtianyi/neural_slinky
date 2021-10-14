import torch
from torch import nn
import torch.nn.functional as F
from .utils import *

class RotationInvariantLayer(nn.Module):
    def __init__(self,NeuronsPerLayer=32):
        #super(RotationInvariantLayer, self).__init__()
        #self.sigma = 0.01.to(device)
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(NeuronsPerLayer,3))
        self.w2 = nn.Parameter(torch.randn(NeuronsPerLayer,1))
        self.w3 = nn.Parameter(torch.randn(NeuronsPerLayer,1))
        self.bias = nn.Parameter(torch.randn(NeuronsPerLayer))
        #self.weight = torch.cat([self.w1,self.w2,self.w3,-self.w2-self.w3],1)
        
    def get_weight_(self):
        weight = torch.cat([self.w1,self.w2,self.w3,-self.w2-self.w3],1)
        bias = self.bias
        return weight, bias

    def forward(self,x):
        #formal_weight, formal_bias = self.get_weight_()
        weight = torch.cat([self.w1,self.w2,self.w3,-self.w2-self.w3],1)
        #return F.linear(x,self.weight,self.bias).to(device)
        out = F.linear(x,weight,self.bias)
        #out = x * torch.transpose(weight,0,1) + self.bias
        return out

class BasicBlock(nn.Module):
    def __init__(self,NeuronsPerLayer):
        super().__init__()
        self.layer1 = nn.Linear(NeuronsPerLayer, NeuronsPerLayer)
        self.layer2 = nn.Linear(NeuronsPerLayer, NeuronsPerLayer)
        self.relu = nn.ReLU()
        self.soft = nn.Softplus()
    
    def forward(self,x):
        identity = x

        out = self.layer1(x)
        out = self.soft(out)
        out = self.layer2(out)
        out = self.soft(out)

        out2 = out + identity
        return out2

class MLPBlock(nn.Module):
    def __init__(self, NeuronsPerLayer, NumLayer):
        super(MLPBlock, self).__init__()
        layer = []
        for i in range(NumLayer):
            layer.append(
                nn.Sequential(
                # nn.BatchNorm1d(NeuronsPerLayer * i + NeuronsPerLayer),
                nn.Linear(NeuronsPerLayer, NeuronsPerLayer),
                nn.Softplus(beta=1e1)#, Square()
                # nn.Tanh()
                )
                )
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate the input and output of each block on the channel
            # dimension
            X = Y
        return X

class DenseBlock(nn.Module):
    def __init__(self, NeuronsPerLayer, NumLayer):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(NumLayer):
            layer.append(
                nn.Sequential(
                        # nn.BatchNorm1d(NeuronsPerLayer * i + NeuronsPerLayer),
                        nn.Linear(NeuronsPerLayer * i + NeuronsPerLayer, NeuronsPerLayer), 
                        nn.Softplus()#, Square()
                    )
                )
                # nn.Linear(NeuronsPerLayer * i + NeuronsPerLayer, NeuronsPerLayer)), nn.Tanh()
        # layer.append(
        #     nn.Sequential(
        #     # nn.BatchNorm1d(NeuronsPerLayer * i + NeuronsPerLayer),
        #     nn.Linear(NeuronsPerLayer * (NumLayer-1) + NeuronsPerLayer, NeuronsPerLayer), 
        #     nn.Softplus(), Square()
        #     )
        #     )
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate the input and output of each block on the channel
            # dimension
            X = torch.cat((X, Y), dim=-1)
            # X = torch.cat((X, Y), dim=1)
        # print("inside dense block")
        # print(X.size())
        return X

class MLPPureSimple(nn.Module):
    def __init__(self, NeuronsPerLayer=32, NumLayer=4):
        super(MLPPureSimple, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(9,NeuronsPerLayer),
            # RotationInvariantLayer(NeuronsPerLayer),
            nn.Softplus(beta=1e3)
            # nn.ReLU()
            # nn.LeakyReLU(negative_slope=0.1)
        )
        self.layer2 = nn.Sequential(
            # nn.Linear(NeuronsPerLayer,NeuronsPerLayer), nn.Softplus(beta=1e3),
            # nn.Linear(NeuronsPerLayer,NeuronsPerLayer), nn.Softplus(beta=1e3)
            # DenseBlock(NeuronsPerLayer,NumLayer)
            MLPBlock(NeuronsPerLayer,NumLayer)
        )
        self.layer3 = nn.Sequential(
            # nn.Linear(int(NeuronsPerLayer*(NumLayer+1)),3)
            nn.Linear(int(NeuronsPerLayer),3)
        )

    def forward(self, y):
        # y.requires_grad_(True)
        # print(y)

        # 1. do preprocessing (equivalently do rigid body motion invariance)
        # x = preprocessing(y)

        # x = (x - features_mu) / features_std

        # 2. pass through the NN
        out = self.layer1(y)
        out = self.layer2(out)
        out = self.layer3(out)

        # 3. taking the derivative
        # deriv = torch.autograd.grad([out.sum()],[y],retain_graph=True,create_graph=True)
        # deriv = torch.autograd.grad(out,y,retain_graph=True,create_graph=True)
        # deriv = torch.sum(out).backward(y,retain_graph=True,create_graph=True)

        # grad = deriv[0]
        # if grad is not None:
        #     return grad[:, 3:6]
        return out

class MLPPolyResnet(nn.Module):
    def __init__(self, NeuronsPerLayer=32):
        super(MLPPolyResnet, self).__init__()
        self.layer1 = nn.Sequential(
            #nn.Linear(6,NeuronsPerLayer)
            RotationInvariantLayer(NeuronsPerLayer)
        )
        self.layer2 = nn.Sequential(
            #nn.BatchNorm1d(3*NeuronsPerLayer),
            nn.Linear(3*NeuronsPerLayer,NeuronsPerLayer),
            nn.Softplus()
        )
        self.layer3 = nn.Sequential(
            #Basic_Block(NeuronsPerLayer),
            #Basic_Block(NeuronsPerLayer),
            #Basic_Block(NeuronsPerLayer),
            BasicBlock(NeuronsPerLayer),
            BasicBlock(NeuronsPerLayer),
            BasicBlock(NeuronsPerLayer),
            BasicBlock(NeuronsPerLayer)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(NeuronsPerLayer,1)
        )
        
    def forward(self, x):
        # with torch.no_grad():
        # augmented_x = torch.stack([x.cpu(), chiral_transformation_x(x.cpu()), chiral_transformation_z(x.cpu()), chiral_transformation_xz(x.cpu())]).to(device)
        augmented_x = torch.stack([x, chiral_transformation_x(x), chiral_transformation_z(x), chiral_transformation_xz(x)])
        #print(augmented_x.shape)
        out = self.layer1(augmented_x)
        out = torch.cat([out, out**2, out**3], dim=-1)
        #print(out.shape)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.sum(out, dim=0, keepdim=False)
        return out

class Softplus2(nn.Module):
    # def __init__(self):
        # super().__init__()
        # self.slope = slope
        # self.threshold = threshold

    def forward(self, x):
        # print(x.size())
        # temp = torch.zeros_like(x)

        # temp = torch.abs(x) * x
        # out = torch.zeros_like(x)
        # mask = x.ge(4.4721)
        # out[mask] = torch.square(x[mask])
        # out[~mask] = torch.log(1+torch.exp(torch.abs(x[~mask]) * x[~mask]))
        # return out

        # for ii in range(len(x)):
        #     if torch.abs(x[ii]) * x[ii] > self.threshold:
        #         temp[ii] = x[ii] ** 2
        #     else:
        #         temp[ii] = torch.log(1+torch.exp(self.slope * torch.abs(x[ii]) * x[ii]))/self.slope
        # if torch.abs(x) * x > self.threshold:
        #     return x**2
        # else:
        # return torch.log(1+torch.exp(self.slope * torch.abs(x) * x))/self.slope
        # return (x>4.4721) * torch.abs(x) * x + (x<=4.4721) * torch.log(1+torch.exp(torch.abs(x) * x))
        return (x>4.4721) * torch.abs(x) * x + (x<=4.4721) * torch.log(1+torch.exp(torch.abs(torch.clamp(x,max=4.4721)) * torch.clamp(x,max=4.4721)))
        # return torch.square(nn.Softplus(x))

class Square(nn.Module):
    def forward(self,x):
        return torch.square(x)

class MLPPure(nn.Module):
    def __init__(self, NeuronsPerLayer=32, NumLayer=4):
        super(MLPPure, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(6,NeuronsPerLayer),
            #RotationInvariantLayer(NeuronsPerLayer),
            nn.Softplus()
        )
        self.layer2 = nn.Sequential(
            DenseBlock(NeuronsPerLayer,NumLayer)
        )
        self.layer3 = nn.Sequential(
            # nn.Linear(int(NeuronsPerLayer/8),1)
            nn.Linear(int(NeuronsPerLayer*(NumLayer+1)),1)
        )
        
    def forward(self, y):

        # augmented_x = torch.stack([x, chiral_transformation_x(x), chiral_transformation_z(x), chiral_transformation_xz(x)], dim=0)
        # augmented_x = torch.stack([make_tan(x), make_tan(chiral_transformation_x(x)), make_tan(chiral_transformation_z(x)), make_tan(chiral_transformation_xz(x))], dim=0)

        # x = region_shifting2(y,features_mu,features_std)
        x = y
        augmented_x = torch.stack([x, chiral_transformation_x(x), chiral_transformation_z(x), chiral_transformation_xz(x)], dim=0)
        out = self.layer1(augmented_x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.sum(out, dim=0, keepdim=False)
        return out

        # out = self.layer1(x)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # return out

        # x1 = x
        # x2 = chiral_transformation_x(x)
        # x3 = chiral_transformation_z(x)
        # x4 = chiral_transformation_xz(x)

        # out1 = self.layer1(x1)
        # out1 = self.layer2(out1)
        # out1 = self.layer3(out1)
        # result = out1

        # out2 = self.layer1(x2)
        # out2 = self.layer2(out2)
        # out2 = self.layer3(out2)
        # result += out2

        # out3 = self.layer1(x3)
        # out3 = self.layer2(out3)
        # out3 = self.layer3(out3)
        # result += out3

        # out4 = self.layer1(x4)
        # out4 = self.layer2(out4)
        # out4 = self.layer3(out4)
        # result += out4

        # return result

        # y = region_shifting(x,features_mu,features_std)
        # augmented_x = torch.stack([y, chiral_transformation_x(y), chiral_transformation_z(y), chiral_transformation_xz(y)], dim=0)
        # out = self.layer1(augmented_x)
        # # out = self.layer1(x)
        # # out = torch.cat([out, out**2, out**3], dim=-1)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = torch.sum(out, dim=0, keepdim=False)

        # y2 = region_shifting2(x,features_mu,features_std)
        # augmented_x2 = torch.stack([y2, chiral_transformation_x(y2), chiral_transformation_z(y2), chiral_transformation_xz(y2)], dim=0)
        # out2 = self.layer1(augmented_x2)
        # # out = self.layer1(x)
        # # out = torch.cat([out, out**2, out**3], dim=-1)
        # out2 = self.layer2(out2)
        # out2 = self.layer3(out2)
        # out2 = torch.sum(out2, dim=0, keepdim=False)
        # return 0.5*(out + out2)
        # return out2

class MLPPureWithBatchnorm(nn.Module):
    def __init__(self, neurons_per_layer=32):
        super(MLPPureWithBatchnorm, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(6,neurons_per_layer), nn.Softplus()
        )
        self.layer2 = nn.Sequential(
            nn.BatchNorm1d(neurons_per_layer),
            nn.Linear(neurons_per_layer,neurons_per_layer), nn.Softplus(),
            # nn.BatchNorm1d(NeuronsPerLayer),
            nn.Linear(neurons_per_layer,neurons_per_layer), nn.Softplus(),
            # nn.BatchNorm1d(NeuronsPerLayer),
            nn.Linear(neurons_per_layer,neurons_per_layer), nn.Softplus()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(int(neurons_per_layer),1)
        )
        
    def forward(self, x):
        x1 = x
        x2 = chiral_transformation_x(x)
        x3 = chiral_transformation_z(x)
        x4 = chiral_transformation_xz(x)

        out1 = self.layer1(x1)
        out1 = self.layer2(out1)
        out1 = self.layer3(out1)

        out2 = self.layer1(x2)
        out2 = self.layer2(out2)
        out2 = self.layer3(out2)

        out3 = self.layer1(x3)
        out3 = self.layer2(out3)
        out3 = self.layer3(out3)

        out4 = self.layer1(x4)
        out4 = self.layer2(out4)
        out4 = self.layer3(out4)
        return out1 + out2 + out3 + out4