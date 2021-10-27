#!/usr/bin/env python3

# %%
import numpy as np

# %%
import scipy.io as scio
import pandas as pd

# %%
from glob import glob
import os

# %%
def preprocess(df, inplace=True):
    if not inplace:
        df = df.copy()
    df.loc[:, "gamma2"] = (df.loc[:, "gamma2"] + np.pi / 2) % (np.pi) - np.pi / 2
    return df


# %%
def transform_z_xi(df):
    result = pd.DataFrame(
        columns=[
            "l1",
            "l2",
            "theta",
            "d_z_1",
            "d_xi_1",
            "d_z_2",
            "d_xi_2",
            "gamma1",
            "gamma2",
            "gamma3",
        ]
    )
    result["l1"] = df["l1"]
    result["l2"] = df["l2"]
    result["theta"] = df["theta"]
    result["gamma1"] = df["gamma1"]
    result["gamma2"] = df["gamma2"]
    result["gamma3"] = df["gamma3"]

    phi_1 = 0.5 * (np.pi - df["theta"]) + df["gamma2"] - 0.5 * df["gamma1"]
    phi_2 = 0.5 * (np.pi + df["theta"]) + df["gamma2"] - 0.5 * df["gamma3"]
    # phi_1 = 0.5 * (np.pi - df["theta"]) - df["gamma2"] - 0.5 * df["gamma1"]
    # phi_2 = 0.5 * (np.pi + df["theta"]) - df["gamma2"] + 0.5 * df["gamma3"]

    result["phi_1"] = phi_1
    result["phi_2"] = phi_2

    result["d_z_1"] = df["l1"] * np.cos(phi_1)
    result["d_xi_1"] = df["l1"] * np.sin(phi_1)

    result["d_z_2"] = df["l2"] * np.cos(phi_2)
    result["d_xi_2"] = df["l2"] * np.sin(phi_2)
    return result


# %%
def make_feature_df(douglas_df):
    feature_df = pd.DataFrame(
        columns=[
            "d_xi_1",
            "d_xi_1^2",
            "d_xi_2",
            "d_xi_2^2",
            "d_z_1^2",
            "d_z_2^2",
            "gamma1^2",
            "gamma3^2",
        ]
    )
    feature_df["d_xi_1"] = douglas_df["d_xi_1"]
    feature_df["d_xi_1^2"] = douglas_df["d_xi_1"] ** 2
    feature_df["d_xi_2"] = douglas_df["d_xi_2"]
    feature_df["d_xi_2^2"] = douglas_df["d_xi_2"] ** 2
    feature_df["d_z_1^2"] = douglas_df["d_z_1"] ** 2
    feature_df["d_z_2^2"] = douglas_df["d_z_2"] ** 2
    feature_df["gamma1^2"] = douglas_df["gamma1"] ** 2
    feature_df["gamma3^2"] = douglas_df["gamma3"] ** 2
    return feature_df


# %%
if __name__ == "__main__":
    import argparse
    import itertools

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mat_files",
        nargs="+",
        required=True,
        type=str,
        help="The path of the .mat files to aggregate",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="The path of the output file"
    )
    args = parser.parse_args()

    df_list = []
    slinky_idx = 0
    # for path in args.mat_files:
    #     print(glob(path))
    for mat_file in itertools.chain(*(glob(path) for path in args.mat_files)):
        print(mat_file)
        mat_data = scio.loadmat(mat_file)
        for mat, energy, total_energy in zip(
            mat_data["NNInput_All"].transpose(2, 1, 0),
            np.squeeze(mat_data["NNOutput_All"].transpose(2, 1, 0)),
            mat_data["circleEnergy"].sum(axis=1),
        ):
            df = pd.DataFrame(
                mat, columns=["l1", "l2", "theta", "gamma1", "gamma2", "gamma3"]
            )
            # feature_df = make_feature_df(transform_z_xi(preprocess(df, inplace=False)))
            df = transform_z_xi(preprocess(df))
            df["energy"] = energy
            df["total_energy"] = total_energy
            df["slinky_idx"] = slinky_idx
            df["triplet_idx"] = np.arange(mat.shape[0])
            slinky_idx += 1
            df["comment"] = os.path.basename(mat_file)
            df_list.append(df)
    total_df = pd.concat(df_list)

    # %%
    total_df.reset_index().to_feather(args.output)
