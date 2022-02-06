#!/usr/bin/env python3
import torch
from neural_slinky import coords_transform
from neural_slinky.douglas_models import DouglasModel
import pandas as pd

import logging

douglas_model = DouglasModel(
    c_dxi=0.04111493612707057,
    c_dxi_sq=24.626752530556853,
    c_dz_sq=77.00113656812572,
    c_dphi_sq=0.036598450117485276,
)


def calculate_douglas_force_cartesian(cartesian_coords: torch.Tensor):
    cartesian_coords.requires_grad_(True)
    douglas_coords = coords_transform.transform_cartesian_to_douglas(cartesian_coords)
    douglas_energy = douglas_model(douglas_coords)
    douglas_force = torch.autograd.grad(
        douglas_energy, cartesian_coords, torch.ones_like(douglas_energy)
    )[0]
    return douglas_force


def calculate_douglas_force_cartesian_alpha(cartesian_alpha_coords: torch.Tensor):
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
    return douglas_force_alpha


def calculate_douglas_force_cartesian_alpha_single(
    cartesian_alpha_coords_single: torch.Tensor,
):
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
    return douglas_force_alpha_single


logger = logging.getLogger(__name__)

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

    def parse_log_level(log_level: str) -> int:
        if log_level == "none":
            return logging.CRITICAL + 10
        else:
            numeric_level = getattr(logging, log_level.upper(), None)
            if not isinstance(numeric_level, int):
                raise ValueError("Invalid log level: {:s}".format(log_level))
            return numeric_level

    logging_levels = ["info", "warning", "debug", "error", "critical", "none"]
    parser.add_argument("--log", type=parse_log_level, default=logging.INFO)

    args = parser.parse_args()

    logger.setLevel(args.log)
    logger_handler = logging.StreamHandler()
    logger_handler.setLevel(args.log)
    logger_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(logger_handler)

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

    logger.info(
        "{0: <60} {1}".format(
            "cartesian-alpha coords shape:", list(cartesian_alpha_coords.shape)
        )
    )
    logger.info(
        "{0: <60} {1}".format(
            "cartesian-alpha coords (single cycle pair ver.) shape:",
            list(cartesian_alpha_coords_single.shape),
        )
    )
    logger.info(
        "{0: <60} {1}".format(
            "cartesian-only coords shape:", list(cartesian_coords.shape)
        )
    )

    # * Generate force by autograd
    # ** For (x, z, alpha) coord
    douglas_force_alpha = calculate_douglas_force_cartesian_alpha(
        cartesian_alpha_coords
    )

    # *** For single pairs
    douglas_force_alpha_single = calculate_douglas_force_cartesian_alpha_single(
        cartesian_alpha_coords_single
    )

    # ** For (x, z) coord
    douglas_force = calculate_douglas_force_cartesian(cartesian_coords)

    # logger.info(
    #     "{0: <60} {1}".format(
    #         "Cartesian-alpha Douglas coords shape:", list(douglas_coords_alpha.shape)
    #     )
    # )
    # logger.info(
    #     "{0: <60} {1}".format(
    #         "Cartesian-alpha-single Douglas coords shape:",
    #         list(douglas_coords_alpha_single.shape),
    #     )
    # )
    # logger.info(
    #     "{0: <60} {1}".format(
    #         "Cartesian-only Douglas coords shape:",
    #         list(douglas_coords.shape),
    #     )
    # )

    # * Save douglas data
    douglas_dataset = {
        "coords": {
            "cartesian": cartesian_coords,
            "cartesian_alpha": cartesian_alpha_coords,
            "cartesian_alpha_single": cartesian_alpha_coords_single.flatten(
                start_dim=-2
            ),
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
