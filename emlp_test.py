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
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["JAX_PLATFORM_NAME"] = "gpu"
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
device = "cpu"
# device = "cuda:2"
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"

from neural_slinky import emlp_models_jax

# %%
import Slinky_NODE_jax as slinky_jax

slinky_data, _, _, _ = slinky_jax.read_data(down_sampling=1)
print(f"{slinky_data['state'].shape}")

length = 20

training_cutoff_ratio = 0.8
training_cutoff = int(len(slinky_data) * training_cutoff_ratio)
data_module = slinky_jax.SlinkyDataModule(
    [slinky_data[:training_cutoff]],
    [slinky_data],
    # [slinky_data[:training_cutoff]],
    input_length=1,
    target_length=length,
    val_length=500,
    batch_size=32,
    perturb=0,  # .01
)

data_module.setup()
# train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
batch = next(iter(val_loader))

# %%
import torch
hyper_parameters = torch.load("jax_checkpoints/76_cycles_quadratic_douglas/last.ckpt")["hyper_parameters"]

# input, true_output, time = regressor_jax.assemble_input_output(batch)
# start_state = input[:1, 0]

# %%
hyper_parameters
# %%
energy_fun = emlp_models_jax.SlinkyEnergyPredictorEMLP(dim_per_layer=384, num_layers=5)
# %%
import jax
force_fun = jax.grad(energy_fun)

# %%
slinky_data["state"][200, :, :3].shape
# %%
energy_fun(slinky_data["state"][200, :, :3].squeeze())
# %%
vforce_fun = jax.jit(jax.vmap(force_fun))
force = vforce_fun(slinky_data["state"][:32, :, :3])
print(force.shape)

# %%
force
# %%
slinky_jax.get_eqx_model_state(energy_fun)

# %%
ch = list(range(10, 400))
dim_list = []
for i in ch:
    # print(f"############################## {i} ##############################")
    middle_rep = emlp_models_jax.uniform_rep(i, energy_fun.input_group)
    # print(middle_rep)
    wdim, wproj = emlp_models_jax.bilinear_weights(emlp_models_jax.gated(middle_rep), emlp_models_jax.gated(middle_rep))
    dim_list.append(wdim)

from matplotlib import pyplot as plt
plt.plot(ch, dim_list)
# %%
plt.plot(ch, dim_list)
plt.xlim(0,300)
plt.ylim(0,30000)
plt.show()
plt.close()
# %%
emlp_models_jax.gated(middle_rep)
# %%
middle_rep = emlp_models_jax.uniform_rep(200, energy_fun.input_group)
emlp_models_jax.bilinear_weights(emlp_models_jax.gated(middle_rep), emlp_models_jax.gated(middle_rep))
# %%
emlp_models_jax.bilinear_weights(emlp_models_jax.gated(energy_fun.output_rep), emlp_models_jax.gated(energy_fun.output_rep))
# %%
# import jax.numpy as jnp
# from jaxtyping import Array, Float, PyTree, jaxtyped
# def transform_cartesian_alpha_to_cartesian_v2(
#     cartesian_alpha_input: Float[Array, "*batchcycles 9"], bar_length: float
# ) -> Float[Array, "*batchcycles 9 2"]:
#     """
#     Args:
#         cartesian_alpha_input: input tensor of shape (... x 9). The columns are
#             (center_x_1, center_z_1, alpha_1, center_x_2, center_z_2, alpha_2,
#             ...)
#         bar_length: A list of three numbers of the bar lengths
#     Returns:
#         Cartesian coordinates of all nodes in the 2D representation. Use the
#         center node of the first cycle in the triplet as the origin, and the
#         first link as the x axis.
#         In total there are 9 nodes, so the output shape is (... x 9 x 2), in
#         the order of (top_1, center_1, bottom_1, top_2, center_2, ...)
#     """
#     alpha_1 = cartesian_alpha_input[..., 2]
#     alpha_2 = cartesian_alpha_input[..., 5]
#     alpha_3 = cartesian_alpha_input[..., 8]

#     center_1 = cartesian_alpha_input[..., [0, 1]]
#     center_2 = cartesian_alpha_input[..., [3, 4]]
#     center_3 = cartesian_alpha_input[..., [6, 7]]

#     def cal_top_bottom(center, alpha, bar_length):
#         top = center.at[..., 0].add(0.5 * bar_length * jnp.cos(alpha))
#         top = top.at[..., 1].add(0.5 * bar_length * jnp.sin(alpha))
#         bottom = center.at[..., 0].add(-0.5 * bar_length * jnp.cos(alpha))
#         bottom = bottom.at[..., 1].add(-0.5 * bar_length * jnp.sin(alpha))
#         return top, bottom

#     top_1, bottom_1 = cal_top_bottom(center_1, alpha_1, bar_length)
#     top_2, bottom_2 = cal_top_bottom(center_2, alpha_2, bar_length)
#     top_3, bottom_3 = cal_top_bottom(center_3, alpha_3, bar_length)
#     return jnp.stack(
#         [
#             top_1,
#             center_1,
#             bottom_1,
#             top_2,
#             center_2,
#             bottom_2,
#             top_3,
#             center_3,
#             bottom_3,
#         ],
#         axis=-2,
#     )

# # %%
# # from emlp.groups import *
# # from emlp.reps import T,vis,V,Scalar
# import emlp.reps as reps
# from emlp.groups import O, S

# so2_rep = reps.V(O(2))
# node_rep = 3 * so2_rep
# # triplet_rep = 18 * node_rep
# scalar_rep = reps.Scalar(O(2))
# # %%
# node_rep

# # %%
# coords = slinky_data["state"][0, :, :3]

# # %%
# triple_coords = slinky_jax.ODEFuncJax._make_triplet_cartesian_alpha(coords)
# # %%
# cartesian_coords = transform_cartesian_alpha_to_cartesian_v2(triple_coords, 0.01)
# cartesian_coords = cartesian_coords.reshape(cartesian_coords.shape[0], -1)
# # %%
# import numpy as np
# from emlp.groups import Group

# class Flip(Group):
#     """ The alternating group in n dimensions"""
#     def __init__(self):
#         flip_first_bar = np.eye(9)
#         flip_first_bar[:3,:3] = np.fliplr(np.eye(3))
#         flip_middle_bar = np.eye(9)
#         flip_middle_bar[3:6,3:6] = np.fliplr(np.eye(3))
#         flip_overall = np.empty((9,9))
#         identity = np.eye(9)
#         flip_overall[:3] = identity[-3:]
#         flip_overall[-3:] = identity[:3]
#         self.discrete_generators = np.stack([flip_first_bar, flip_middle_bar, flip_overall], axis=0)
#         super().__init__(9)
# # %%
# from matplotlib import pyplot as plt
# input_group = Flip() * O(2)
# reps.V.rho_dense(input_group.sample())
# input_group = Flip() * O(2)
# plt.imshow(reps.V.rho_dense((input_group).sample()))
# # %%
# input_rep = reps.V(input_group)
# scalar_rep = reps.Scalar(input_group)
# emlp = emlp_models_jax.EMLP(input_rep, scalar_rep, input_group)

# # %%
# cartesian_coords.shape
# # %%
# idx = 8
# print(f"{emlp(cartesian_coords[idx])=}")
# transformation_mat = reps.V.rho_dense((input_group).samples(100))
# plt.imshow(transformation_mat[0])
# plt.show()
# plt.close()
# plt.imshow(transformation_mat[1])
# plt.show()
# plt.close()
# import jax
# print(f"{jax.vmap(emlp)(transformation_mat @ cartesian_coords[idx])=}")
# # %%
