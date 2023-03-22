# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: 'Python 3.7.4 64-bit (''py37'': conda)'
#     language: python
#     name: python37464bitpy37conda606da32f2b124e7b8f99ee90fe706c33
# ---

# %%
import torch_scatter

# %%
import torch_cluster

# %%
from e3nn import o3
import e3nn
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.util.test import assert_equivariant

# %%
# ?o3.spherical_harmonics

# %%
irreps_sh = o3.Irreps.spherical_harmonics(3)

# %%
import torch

# %%
o3.spherical_harmonics

# %%
o3.Irreps("1o").D_from_angles(torch.ones(1)*2, torch.ones(1)*3, torch.ones(1)*4)

# %%
o3.angles_to_matrix(torch.ones(1)*2, torch.ones(1)*3, torch.ones(1)*4)

# %%
irreps_sh

# %%
irreps_sh = o3.Irreps.spherical_harmonics(3)
irreps_mid = o3.Irreps("64x0e + 24x1e + 24x1o + 16x2e + 16x2o")
irreps_out = o3.Irreps("1o")

tp1 = FullyConnectedTensorProduct(
            irreps_in1=irreps_sh,
            irreps_in2=irreps_sh,
            irreps_out=irreps_mid,
        )
tp2 = FullyConnectedTensorProduct(
            irreps_in1=irreps_mid,
            irreps_in2=irreps_sh,
            irreps_out=irreps_out,
        )
def f(x):
#     sh = o3.spherical_harmonics(
#         l=irreps_sh,
#         x=x,
#         normalize=False,  # here we don't normalize otherwise it would not be a polynomial
#         normalization='component'
#     )
#     features = tp1(sh, sh)
#     output = tp2(features, sh)
    
    output = o3.spherical_harmonics(
        l=irreps_out,
        x=x,
        normalize=False,  # here we don't normalize otherwise it would not be a polynomial
        normalization='component'
    )
    return output


# %%
tp = o3.FullyConnectedTensorProduct("2x0e + 3x1o", "2x0e + 3x1o", "2x1o")

assert_equivariant(
    tp,
    args_in=[tp.irreps_in1.randn(1, -1), tp.irreps_in2.randn(1, -1)],
    irreps_in=[tp.irreps_in1, tp.irreps_in2],
    irreps_out=[tp.irreps_out]
)

# %%
import pandas as pd

# %%
df = pd.read_feather("data/aggregated_slinky_triplet_energy.feather")

# %%
from data import coords_transform

# %%
import numpy as np
import torch

# %%
triplet_input = torch.from_numpy(
    df[["l1", "l2", "theta", "gamma1", "gamma2", "gamma3"]].to_numpy()
)#.requires_grad_()
cartesian_coords = coords_transform.transform_triplet_to_cartesian(triplet_input, [0.1, 0.1, 0.1])


# %%
douglas_1 = coords_transform.transform_cartesian_to_douglas(cartesian_coords)

# %%
douglas_1.shape

# %%
douglas_2 = coords_transform.transform_cartesian_alpha_to_douglas(
    coords_transform.transform_triplet_to_cartesian_alpha(triplet_input)
)

# %%
douglas_2.shape

# %%
torch.allclose(douglas_1, douglas_2)

# %%
triplet_recover = coords_transform.transform_cartesian_alpha_to_triplet(
    coords_transform.transform_triplet_to_cartesian_alpha(triplet_input)
)

# %%
torch.allclose(triplet_input, triplet_recover)

# %%
douglas_3 = coords_transform.transform_cartesian_alpha_to_douglas_single(coords_transform.transform_triplet_to_cartesian_alpha_single(triplet_input))

# %%
torch.allclose(douglas_1, douglas_3)

# %%
df

# %%
cartesian_coords.shape

# %%
torch_cluster.radius_graph()

# %%
g = e3nn.Gate("16x0o", [torch.tanh], "32x0o", [torch.tanh], "16x1e+16x1o")
g.irreps_out

# %%

# %%
R = o3.rand_matrix()
print(R)
print(o3.Irreps("1o").D_from_matrix(R))

# %%
x = torch.randn(10,3)

# %%
o3.spherical_harmonics(l="1o", x=x @ R, normalize=False)

# %%
o3.spherical_harmonics(l="1o", x=x, normalize=False) @ R

# %%
assert_equivariant(lambda x: o3.spherical_harmonics(
        l="1o",
        x=x[...,[0,1,2]],
        normalize=False,  # here we don't normalize otherwise it would not be a polynomial
        normalization='integral'
), args_in=[torch.randn(100,3)], irreps_in=['cartesian_points'], irreps_out=["1o"], do_translation=False)

# %%
[(a, b) for a, b in irreps_mid]

# %%
assert_equivariant(f, args_in=[torch.randn(100,3)], irreps_in=['cartesian_points'], irreps_out=["1o"], do_translation=False)

# %%
o3.Irreps("2x1o").D_from_angles(torch.ones(1)*2, torch.ones(1)*3, torch.ones(1)*4)

# %%
irreps_sh.D_from_angles(torch.ones(1)*2, torch.ones(1)*3, torch.ones(1)*4)

# %%
# ?o3.spherical_harmonics

# %%
# ?equivariance_error

# %%
x = torch.randn(10, 3)
# print(o3.spherical_harmonics("0e+1e+2e", x, normalize=False, normalization="component"))
# print(o3.spherical_harmonics("2o", -x, normalize=False, normalization="component"))
print(o3.spherical_harmonics("2e", x, normalize=False, normalization="component"))
print(o3.spherical_harmonics("2e", -x, normalize=False, normalization="component"))

# %%

# %%
import torch
from torch_cluster import radius_graph
from torch_scatter import scatter
from e3nn import o3, nn
from e3nn.math import soft_one_hot_linspace
import matplotlib.pyplot as plt

# %%
irreps_input = o3.Irreps("10x0e + 10x1e")
irreps_output = o3.Irreps("20x0e + 10x1e")

# %%
# create node positions
num_nodes = 100
pos = torch.randn(num_nodes, 3)  # random node positions

# create edges
max_radius = 1.8
edge_src, edge_dst = radius_graph(pos, max_radius, max_num_neighbors=num_nodes - 1)

print(edge_src.shape)

edge_vec = pos[edge_dst] - pos[edge_src]

# compute z
num_neighbors = len(edge_src) / num_nodes
num_neighbors

# %%
f_in = irreps_input.randn(num_nodes, -1)

# %%
irreps_sh = o3.Irreps.spherical_harmonics(lmax=2)

# %%
sh = o3.spherical_harmonics(irreps_sh, edge_vec, normalize=True, normalization='component')

# %%
sh.pow(2).mean(dim=-1)

# %%
tp = o3.FullyConnectedTensorProduct(irreps_input, irreps_sh, irreps_output, shared_weights=False)

print(f"{tp} needs {tp.weight_numel} weights")

tp.visualize();

# %%
num_basis = 10

x = torch.linspace(0.0, 2.0, 1000)
y = soft_one_hot_linspace(
    x,
    start=0.0,
    end=max_radius,
    number=num_basis,
    basis='smooth_finite',
    cutoff=True,
)

plt.plot(x, y);

# %%
edge_length_embedding = soft_one_hot_linspace(
    edge_vec.norm(dim=1),
    start=0.0,
    end=max_radius,
    number=num_basis,
    basis='smooth_finite',
    cutoff=True,
)
edge_length_embedding = edge_length_embedding.mul(num_basis**0.5)

print(edge_length_embedding.shape)
edge_length_embedding.pow(2).mean()  # the second moment

# %%
fc = nn.FullyConnectedNet([num_basis, 16, tp.weight_numel], torch.relu)
weight = fc(edge_length_embedding)

print(weight.shape)
print(len(edge_src), tp.weight_numel)

# For a proper normalization, the weights also need to be mean 0
print(weight.mean(), weight.std())  # should close to 0 and 1

# %%
summand = tp(f_in[edge_src], sh, weight)

print(summand.shape)
print(summand.pow(2).mean())  # should be close to 1

# %%
f_out = scatter(summand, edge_dst, dim=0, dim_size=num_nodes)

# %%
f_out.shape

# %%
f_out = f_out.div(num_neighbors**0.5)

f_out.pow(2).mean()  # should be close to 1

# %%

# %%
douglas_dataset = torch.load("data/douglas_dataset.pt")

# %%
douglas_dataset.keys()

# %%
douglas_dataset['coords'].keys()

# %%
for key in douglas_dataset["coords"]:
    print(key, douglas_dataset["coords"][key].shape)

# %%
for key in douglas_dataset["coords"]:
    print(key, douglas_dataset["force"][key].shape)

# %%

# %%
from neural_slinky import e3nn_models

# %%
e3nn_models.SlinkyForcePredictor(
    irreps_node_input="1x0e",
    irreps_node_attr="1x0e+1x1o",
    irreps_node_output="1x0e+1x1o",
    max_radius=0.06,
    num_neighbors=1,
    num_nodes=2,
    mul=50,
    layers=3,
    lmax=2,
)

# %%
import e3nn.io

# %%
e3nn.io.SphericalTensor(4,-1,-1)

# %%
from neural_slinky import coords_transform

# %%
# ?coords_transform.transform_cartesian_alpha_to_triplet

# %%
import numpy as np
import torch
def readinData():
    folder = './NeuralODE_Share2/SlinkyGroundTruth'
    true_y = np.loadtxt(folder+'/helixCoordinate_2D.txt',delimiter=',')
    true_v = np.loadtxt(folder+'/helixVelocity_2D.txt',delimiter=',')
    true_y = torch.from_numpy(true_y).float()
    true_v = torch.from_numpy(true_v).float()
    true_y = torch.reshape(true_y,(true_y.size()[0],80,3))
    true_v = torch.reshape(true_v,(true_v.size()[0],80,3))
    # print(true_v.shape)
    return torch.cat((true_y, true_v),-1)


# %%
slinky_sequence_data = readinData()

# %%
coords_transform.group_into_triplets(slinky_sequence_data[...,:3]).flatten(start_dim=0, end_dim=-1).shape

# %%
coords_transform.transform_triplet_to_cartesian(
    coords_transform.transform_cartesian_alpha_to_triplet(
        coords_transform.group_into_triplets(slinky_sequence_data[..., :3]).flatten(-2)
    ), [0.1, 0.1, 0.1]
).shape

# %%

# %%
coords_transform.transform_cartesian_alpha_to_triplet()
