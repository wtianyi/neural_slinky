from ast import Slice
import torch
from torch import nn
import torch.nn.functional as F
from .utils import *
from typing import Tuple


class RotationInvariantLayer(nn.Module):
    def __init__(self, NeuronsPerLayer=32):
        # super(RotationInvariantLayer, self).__init__()
        # self.sigma = 0.01.to(device)
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(NeuronsPerLayer, 3))
        self.w2 = nn.Parameter(torch.randn(NeuronsPerLayer, 1))
        self.w3 = nn.Parameter(torch.randn(NeuronsPerLayer, 1))
        self.bias = nn.Parameter(torch.randn(NeuronsPerLayer))
        # self.weight = torch.cat([self.w1,self.w2,self.w3,-self.w2-self.w3],1)

    def get_weight_(self):
        weight = torch.cat([self.w1, self.w2, self.w3, -self.w2 - self.w3], 1)
        bias = self.bias
        return weight, bias

    def forward(self, x):
        # formal_weight, formal_bias = self.get_weight_()
        weight = torch.cat([self.w1, self.w2, self.w3, -self.w2 - self.w3], 1)
        # return F.linear(x,self.weight,self.bias).to(device)
        out = F.linear(x, weight, self.bias)
        # out = x * torch.transpose(weight,0,1) + self.bias
        return out


class BasicBlock(nn.Module):
    def __init__(self, NeuronsPerLayer):
        super().__init__()
        self.layer1 = nn.Linear(NeuronsPerLayer, NeuronsPerLayer)
        self.layer2 = nn.Linear(NeuronsPerLayer, NeuronsPerLayer)
        self.relu = nn.ReLU()
        self.soft = nn.Softplus()

    def forward(self, x):
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
                    nn.Softplus(beta=1e1)  # , Square()
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
                nn.Linear(NeuronsPerLayer * i + NeuronsPerLayer, NeuronsPerLayer), 
                nn.Softplus(beta=1e1)
                )
            )
        self.net = nn.Sequential(*layer)
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=-1)
        return Y


class MLPPureSimple(nn.Module):
    def __init__(self, NeuronsPerLayer=32, NumLayer=4):
        super(MLPPureSimple, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(9, NeuronsPerLayer),
            # RotationInvariantLayer(NeuronsPerLayer),
            nn.Softplus(beta=1e3)
            # nn.ReLU()
            # nn.LeakyReLU(negative_slope=0.1)
        )
        self.layer2 = nn.Sequential(
            # nn.Linear(NeuronsPerLayer,NeuronsPerLayer), nn.Softplus(beta=1e3),
            # nn.Linear(NeuronsPerLayer,NeuronsPerLayer), nn.Softplus(beta=1e3)
            # DenseBlock(NeuronsPerLayer,NumLayer)
            MLPBlock(NeuronsPerLayer, NumLayer)
        )
        self.layer3 = nn.Sequential(
            # nn.Linear(int(NeuronsPerLayer*(NumLayer+1)),3)
            nn.Linear(int(NeuronsPerLayer), 3)
        )

    def forward(self, y):
        out = self.layer1(y)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


class MLPPolyResnet(nn.Module):
    def __init__(self, NeuronsPerLayer=32):
        super(MLPPolyResnet, self).__init__()
        self.layer1 = nn.Sequential(
            # nn.Linear(6,NeuronsPerLayer)
            RotationInvariantLayer(NeuronsPerLayer)
        )
        self.layer2 = nn.Sequential(
            # nn.BatchNorm1d(3*NeuronsPerLayer),
            nn.Linear(3 * NeuronsPerLayer, NeuronsPerLayer),
            nn.Softplus(),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(NeuronsPerLayer),
            BasicBlock(NeuronsPerLayer),
            BasicBlock(NeuronsPerLayer),
            BasicBlock(NeuronsPerLayer),
        )
        self.layer4 = nn.Sequential(nn.Linear(NeuronsPerLayer, 1))

    def forward(self, x):
        # with torch.no_grad():
        # augmented_x = torch.stack([x.cpu(), chiral_transformation_x(x.cpu()), chiral_transformation_z(x.cpu()), chiral_transformation_xz(x.cpu())]).to(device)
        augmented_x = torch.stack(
            [
                x,
                chiral_transformation_x(x),
                chiral_transformation_z(x),
                chiral_transformation_xz(x),
            ]
        )
        # print(augmented_x.shape)
        out = self.layer1(augmented_x)
        out = torch.cat([out, out ** 2, out ** 3], dim=-1)
        # print(out.shape)
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
        return (x > 4.4721) * torch.abs(x) * x + (x <= 4.4721) * torch.log(
            1
            + torch.exp(
                torch.abs(torch.clamp(x, max=4.4721)) * torch.clamp(x, max=4.4721)
            )
        )


class Square(nn.Module):
    def forward(self, x):
        return torch.square(x)


class MLPPure(nn.Module):
    def __init__(self, input_dim=6, layer_width=32, num_layers=4):
        super(MLPPure, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, layer_width),
            # RotationInvariantLayer(NeuronsPerLayer),
            nn.Softplus(),
        )
        self.layer2 = nn.Sequential(DenseBlock(layer_width, num_layers))
        self.layer3 = nn.Sequential(
            # nn.Linear(int(NeuronsPerLayer/8),1)
            nn.Linear(int(layer_width * (num_layers + 1)), 1)
        )

    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.sum(out, dim=0, keepdim=False)
        return out


class MLPPureWithBatchnorm(nn.Module):
    def __init__(self, neurons_per_layer=32):
        super(MLPPureWithBatchnorm, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(6, neurons_per_layer), nn.Softplus())
        self.layer2 = nn.Sequential(
            nn.BatchNorm1d(neurons_per_layer),
            nn.Linear(neurons_per_layer, neurons_per_layer),
            nn.Softplus(),
            # nn.BatchNorm1d(NeuronsPerLayer),
            nn.Linear(neurons_per_layer, neurons_per_layer),
            nn.Softplus(),
            # nn.BatchNorm1d(NeuronsPerLayer),
            nn.Linear(neurons_per_layer, neurons_per_layer),
            nn.Softplus(),
        )
        self.layer3 = nn.Sequential(nn.Linear(int(neurons_per_layer), 1))

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


class TripletChiralInvariantWrapper(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, input_tensor: torch.Tensor, **kwargs):
        output = self.backbone(input_tensor, **kwargs)
        for transform in [
            chiral_transformation_x,
            chiral_transformation_z,
            chiral_transformation_xz,
        ]:
            output = output + self.backbone(transform(input_tensor), **kwargs)
        return output


class DouglasChiralInvariantWrapper(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, input_tensor: torch.Tensor, **kwargs):
        output = self.backbone(input_tensor, **kwargs)
        for transform in [
            chiral_transformation_x_douglas,
            chiral_transformation_z_douglas,
            chiral_transformation_xz_douglas,
        ]:
            output = output + self.backbone(transform(input_tensor), **kwargs)
        return output


class AutogradOutputWrapper(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        output_mask: Union[int, slice, torch.Tensor, List, Tuple] = slice(None),
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.output_mask = output_mask

    def forward(self, input_tensor: torch.Tensor, **kwargs):
        with torch.enable_grad():
            input_tensor.requires_grad_(True)
            backbone_output = self.backbone(input_tensor, **kwargs)
            grad = torch.autograd.grad(
                backbone_output,
                input_tensor,
                grad_outputs=torch.ones_like(backbone_output),
                create_graph=True,
                retain_graph=True,
            )[0]
        return grad[self.output_mask]


# TODO: maybe directly use e3nn's `e3nn.o3.FullyConnectedTensorProduct`
# see https://github.com/e3nn/e3nn/blob/3afd6b07a19a2daa94864c50c5baeda0ff1fc6a9/e3nn/o3/_tensor_product/_tensor_product.py#L674
class SimplePolynomial(nn.Module):
    def __init__(self, order=2) -> None:
        super().__init__()
        pass


class SphericalPolynomial(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pass
