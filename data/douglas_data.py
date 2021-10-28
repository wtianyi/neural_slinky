#!/usr/bin/env python3
import torch
from . import coords_transform
import pandas as pd


class DouglasModelTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cycle_triplet_coord):
        """
        Args:
            cycle_triplet_coord: input tensor of shape (N x 6). The columns are
                (l1, l2, theta, gamma1, gamma2, gamma3)
        Returns:
            Douglas model's degrees of freedom of shape (N x 2 x 3).
            The last axis is (dxi, dz, dphi) where dphi is gamma1 or gamma3
        """
        l1 = cycle_triplet_coord[..., 0]
        l2 = cycle_triplet_coord[..., 1]
        theta = cycle_triplet_coord[..., 2]
        gamma1 = cycle_triplet_coord[..., 3]
        gamma2 = cycle_triplet_coord[..., 4]
        gamma3 = cycle_triplet_coord[..., 5]

        #         psi_1 = 0.5 * (np.pi - theta) - gamma2 - 0.5 * gamma1
        #         psi_2 = 0.5 * (np.pi + theta) - gamma2 + 0.5 * gamma3
        psi_1 = 0.5 * (np.pi - theta) + gamma2 - 0.5 * gamma1
        psi_2 = 0.5 * (np.pi + theta) + gamma2 - 0.5 * gamma3

        dz_1 = l1 * torch.cos(psi_1)
        dxi_1 = l1 * torch.sin(psi_1)

        dz_2 = l2 * torch.cos(psi_2)
        dxi_2 = l2 * torch.sin(psi_2)

        first_pair_dof = torch.stack([dxi_1, dz_1, gamma1], dim=-1)  # (N x 3)
        second_pair_dof = torch.stack([dxi_2, dz_2, gamma3], dim=-1)  # (N x 3)
        return torch.stack([first_pair_dof, second_pair_dof], dim=1)  # (N x 2 x 3)


class DouglasModel(torch.nn.Module):
    def __init__(self, c_dxi, c_dxi_sq, c_dz_sq, c_dphi_sq):
        super().__init__()
        self.c_dxi = c_dxi
        self.c_dxi_sq = c_dxi_sq
        self.c_dz_sq = c_dz_sq
        self.c_dphi_sq = c_dphi_sq

    def forward(self, douglas_dof):
        """
        Args:
            douglas_dof: (... x 3) tensor. The columns are (dxi, dz, dphi)
        """
        dxi = douglas_dof[..., 0]
        dz = douglas_dof[..., 1]
        dphi = douglas_dof[..., 2]

        return (
            self.c_dxi * dxi
            + self.c_dxi_sq * dxi ** 2
            + self.c_dz_sq * dz ** 2
            + self.c_dphi_sq * dphi ** 2
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--triplet_data",
        type=str,
        required=True,
        help="The original triplet data in feather format",
    )
    parser.add_argument("--output", type=str, required=True, help="The output path")
    args = parser.parse_args()

    # * Import triplet df
    df = pd.read_feather(args.triplet_data)
    print(df.head())

    # * Transform to cartesian
    triplet_input = torch.from_numpy(
        df[["l1", "l2", "theta", "gamma1", "gamma2", "gamma3"]].to_numpy()
    )  # .requires_grad_()

    cartesian_alpha_coords = coords_transform.transform_triplet_to_cartesian_alpha(
        triplet_input
    )

    cartesian_alpha_coords_single = coords_transform.transform_triplet_to_cartesian_alpha_single(
        triplet_input
    )

    cartesian_coords = coords_transform.transform_triplet_to_cartesian(
        triplet_input, [0.1, 0.1, 0.1]
    )

    # * Generate force by autograd
    douglas_model = DouglasModel(
        c_dxi=0.04111493612707057,
        c_dxi_sq=24.626752530556853,
        c_dz_sq=77.00113656812572,
        c_dphi_sq=0.036598450117485276,
    )

    # ** For (x, z, alpha) coord
    cartesian_alpha_coords.requires_grad_(True)
    douglas_coords_alpha = coords_transform.transform_cartesian_alpha_to_douglas(
        cartesian_alpha_coords
    )
    douglas_energy_alpha = douglas_model(douglas_coords_alpha)
    douglas_force_alpha = torch.autograd.grad(
        douglas_energy_alpha,
        cartesian_alpha_coords,
        torch.ones_like(douglas_energy_alpha),
    )[0]

    # *** For single pairs
    cartesian_alpha_coords_single.requires_grad_(True)
    douglas_coords_alpha_single = coords_transform.transform_cartesian_alpha_to_douglas_single(
        cartesian_alpha_coords_single
    )
    douglas_energy_alpha_single = douglas_model(douglas_coords_alpha_single)
    douglas_force_alpha_single = torch.autograd.grad(
        douglas_energy_alpha_single,
        cartesian_alpha_coords_single,
        torch.ones_like(douglas_energy_alpha_single),
    )[0]

    # ** For (x, z) coord
    cartesian_coords.requires_grad_(True)
    douglas_coords = coords_transform.transform_cartesian_to_douglas(cartesian_coords)
    douglas_energy = douglas_model(douglas_coords)
    douglas_force = torch.autograd.grad(
        douglas_energy, cartesian_coords, torch.ones_like(douglas_energy)
    )[0]

    # * Save douglas data
    douglas_dataset = {
        "coords": {
            "cartesian": cartesian_coords,
            "cartesian_alpha": cartesian_alpha_coords,
            "cartesian_alpha_single": cartesian_alpha_coords_single.flatten(start_dim=-2),
        },
        "force": {
            "cartesian": douglas_force,
            "cartesian_alpha": douglas_force_alpha,
            "cartesian_alpha_single": douglas_force_alpha_single.flatten(start_dim=-2),
        },
    }
    for key, value in douglas_dataset.items():
        print(f"================== {key} ==================")
        [print("{:<10}: {}".format(k, value[k].shape)) for k in value]
    torch.save(
        douglas_dataset, args.output,
    )
