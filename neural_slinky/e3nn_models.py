from typing import Dict
import torch

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn.models.v2106.gate_points_message_passing import MessagePassing
from torch_scatter import scatter
from torch_cluster import radius_graph


class SlinkyForcePredictor(torch.nn.Module):
    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        # irreps_edge_attr,
        irreps_node_output,
        max_radius,
        num_neighbors,
        num_nodes,
        mul=50,
        layers=3,
        lmax=2,
        pool_nodes=True,
    ) -> None:
        super().__init__()

        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = 10
        self.num_nodes = num_nodes
        # self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_edge_attr = o3.Irreps("")
        self.pool_nodes = pool_nodes

        irreps_node_hidden = o3.Irreps(
            [(mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]]
        )

        self.mp = MessagePassing(
            irreps_node_sequence=[irreps_node_input]
            + layers * [irreps_node_hidden]
            + [irreps_node_output],
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr
            + o3.Irreps.spherical_harmonics(lmax),
            fc_neurons=[self.number_of_basis, 100],
            num_neighbors=num_neighbors,
        )

        self.irreps_node_input = self.mp.irreps_node_input
        self.irreps_node_attr = self.mp.irreps_node_attr
        self.irreps_node_output = self.mp.irreps_node_output

    def preprocess(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "batch" in data:
            batch = data["batch"]
        else:
            batch = data["pos"].new_zeros(data["pos"].shape[0], dtype=torch.long)

        # Create graph
        if "edge_src" in data and "edge_dst" in data:
            edge_src = data["edge_src"]
            edge_dst = data["edge_dst"]
        else:
            edge_index = radius_graph(data["pos"], self.max_radius, batch)
            edge_src = edge_index[0]
            edge_dst = edge_index[1]

        # Edge attributes
        edge_vec = data["pos"][edge_src] - data["pos"][edge_dst]

        if "x" in data:
            node_input = data["x"]
        else:
            node_input = data["node_input"]

        node_attr = data["node_attr"]
        edge_attr = data["edge_attr"]

        return batch, node_input, node_attr, edge_attr, edge_src, edge_dst, edge_vec

    def forward(self, node_pos, bar_alpha) -> torch.Tensor:
        edge_vec = node_pos[..., 1] - node_pos[..., 0]
        # Edge attributes
        edge_sh = o3.spherical_harmonics(
            range(self.lmax + 1), edge_vec, True, normalization="component"
        )

        # edge_attr = torch.zeros(edge_vec.shape[:-1]+(1,), device=edge_vec.device)
        # edge_attr = torch.cat([edge_attr, edge_sh], dim=1)
        edge_attr = edge_sh

        # Edge length embedding
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = soft_one_hot_linspace(
            edge_length,
            0.0,
            self.max_radius,
            self.number_of_basis,
            basis="smooth_finite",  # the smooth_finite basis with cutoff = True goes to zero at max_radius
            cutoff=True,  # no need for an additional smooth cutoff
        ).mul(self.number_of_basis ** 0.5)

        node_input = bar_alpha.view(-1,1)
        # print("node_input.shape:", node_input.shape)
        node_attr = torch.ones_like(node_input)
        edge_src = torch.arange(edge_vec.shape[0], device=edge_vec.device) * 2
        edge_dst = edge_src + 1
        node_outputs = self.mp(
            node_input, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding
        )

        # if self.pool_nodes:
        #     return scatter(node_outputs, batch, int(batch.max()) + 1).div(
        #         self.num_nodes ** 0.5
        #     )
        # else:
        return node_outputs

class SlinkyForcePredictorCartesian(torch.nn.Module):
    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_node_output,
        max_radius,
        num_neighbors,
        num_nodes,
        mul=50,
        layers=3,
        lmax=2,
        pool_nodes=True,
    ) -> None:
        super().__init__()

        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = 10
        self.num_nodes = num_nodes
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.pool_nodes = pool_nodes

        irreps_node_hidden = o3.Irreps([
            (mul, (l, p))
            for l in range(lmax + 1)
            for p in [-1, 1]
        ])

        self.mp = MessagePassing(
            irreps_node_sequence=[irreps_node_input] + layers * [irreps_node_hidden] + [irreps_node_output],
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr + o3.Irreps.spherical_harmonics(lmax),
            fc_neurons=[self.number_of_basis, 100],
            num_neighbors=num_neighbors,
        )

        self.irreps_node_input = self.mp.irreps_node_input
        self.irreps_node_attr = self.mp.irreps_node_attr
        self.irreps_node_output = self.mp.irreps_node_output

    def preprocess(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        if 'batch' in data:
            batch = data['batch']
        else:
            batch = data['pos'].new_zeros(data['pos'].shape[0], dtype=torch.long)

        # Create graph
        if 'edge_src' in data and 'edge_dst' in data:
            edge_src = data['edge_src']
            edge_dst = data['edge_dst']
        else:
            edge_index = radius_graph(data['pos'], self.max_radius, batch)
            edge_src = edge_index[0]
            edge_dst = edge_index[1]

        # Edge attributes
        edge_vec = data['pos'][edge_src] - data['pos'][edge_dst]

        if 'x' in data:
            node_input = data['x']
        else:
            node_input = data['node_input']

        node_attr = data['node_attr']
        edge_attr = data['edge_attr']

        return batch, node_input, node_attr, edge_attr, edge_src, edge_dst, edge_vec

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch, node_input, node_attr, edge_attr, edge_src, edge_dst, edge_vec = self.preprocess(data)
        del data

        # Edge attributes
        edge_sh = o3.spherical_harmonics(range(self.lmax + 1), edge_vec, True, normalization='component')
        edge_attr = torch.cat([edge_attr, edge_sh], dim=1)

        # Edge length embedding
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = soft_one_hot_linspace(
            edge_length,
            0.0,
            self.max_radius,
            self.number_of_basis,
            basis='smooth_finite',  # the smooth_finite basis with cutoff = True goes to zero at max_radius
            cutoff=True,  # no need for an additional smooth cutoff
        ).mul(self.number_of_basis**0.5)

        node_outputs = self.mp(node_input, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding)

        if self.pool_nodes:
            return scatter(node_outputs, batch[...,None], dim=0).div(self.num_nodes**0.5)
        else:
            return node_outputs

