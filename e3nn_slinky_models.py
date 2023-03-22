import torch
from torch import nn
from neural_slinky import coords_transform
from neural_slinky import e3nn_models
from neural_slinky.utils import get_model_device


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
            pool_nodes=False,
        )

    def forward(self, y: torch.Tensor):
        """Predict the elastic force of all 

        Args:
            y (torch.Tensor): neural ode input. Contains dof and vel. Shape (... x L x 6)

        Returns:
            torch.Tensor: the predicted force for the dof
        """
        device = get_model_device(self)
        if len(y.shape) < 3:
            y = y[None, ...]
        B, L, _ = y.shape
        # print("y.shape", y.shape)
        cartesian = coords_transform.transform_triplet_to_cartesian(
            coords_transform.transform_cartesian_alpha_to_triplet(
                coords_transform.group_coords(y[..., 0:3]).flatten(
                    start_dim=-2
                )  # (B x (L-2) x 9)
            ),  # (B x (L-2) x 6)
            [0.1, 0.1, 0.1],
        ).flatten(
            end_dim=1
        )  # (N x 9 x 2), N = B x (L-2)
        # print("cartesian.shape", cartesian.shape)

        num_triplets = cartesian.shape[0]
        batch = (
            torch.arange(num_triplets, device=device)
            .view(-1, 1)
            .repeat((1, 9))
            .flatten()
            .long()
        )
        edge_index = torch.tensor(
            [[1, 0], [1, 2], [4, 3], [4, 5], [7, 6], [7, 8], [1, 4], [4, 7]],
            device=device,
        ).repeat((num_triplets, 1)) + torch.arange(num_triplets, device=device).mul_(
            9
        ).repeat_interleave(
            8
        ).view(
            -1, 1
        )
        edge_index = edge_index.long()

        edge_attr = torch.nn.functional.one_hot(
            torch.tensor([1, 1, 1, 1, 1, 1, 0, 0], device=device).repeat(num_triplets),
            num_classes=2,
        ).float()

        node_attr = torch.nn.functional.one_hot(
            torch.tensor([1, 0, 1], device=device).repeat(3 * num_triplets),
            num_classes=2,
        ).float()

        cartesian = cartesian.reshape(-1, 2)
        node_input = cartesian.new_ones(cartesian.shape[0], 1)
        data = {
            "pos": torch.cat(
                [cartesian, cartesian.new_zeros(cartesian.shape[0], 1)], dim=-1
            ),
            "edge_src": edge_index[:, 0],
            "edge_dst": edge_index[:, 1],
            "node_input": node_input,
            "batch": batch,
            "node_attr": node_attr,
            "edge_attr": edge_attr,
        }
        output = self.e3nn(data)  # (9N x 3)
        # print("output.shape", output.shape)
        output = output.reshape(B, (L - 2), 3, 3, 3)
        cartesian = cartesian.reshape(B, (L - 2), 3, 3, 2)
        result = y.new_zeros(B, L, 3)
        result[:, 1:-1, 0:2] = output.sum(dim=-2)[..., 1, 0:2]
        tmp = coords_transform.cross2d(
            cartesian[..., 0, :] - cartesian[..., 1, :], output[..., 0, 0:2]
        ) + coords_transform.cross2d(
            cartesian[..., 2, :] - cartesian[..., 1, :], output[..., 2, 0:2]
        )
        # print("tmp.shape", tmp.shape)
        result[:, 1:-1, 2] = tmp[..., 1]
        return result.squeeze(0)

