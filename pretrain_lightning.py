import itertools
import os
from typing import Dict, Optional

import configargparse
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import loggers as pl_loggers
from sklearn.model_selection import train_test_split
from torch.utils import data as torch_data

import wandb
from neural_slinky.utils import plot_regression_scatter
from slinky import func as node_f

# from torchtyping import TensorType


# from neural_slinky import emlp_models


class DouglasForceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=8,
        perturb: float = 0,
        datafile_path: str = "data/douglas_dataset.pt",
        fit_energy: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.perturb = perturb
        self.datafile_path = datafile_path

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("DouglasForceData")
        parser.add_argument("--batch_size", type=int, help="Batch size")
        parser.add_argument(
            "--datafile_path",
            type=str,
            default="data/douglas_dataset.pt",
            help="Data file path",
        )
        parser.add_argument(
            "--perturb",
            type=float,
            default=0,
            help="Gaussian perturbation augmentation scale",
        )
        parser.add_argument(
            "--fit_energy",
            action="store_true",
            default=False,
            help="Whether to fit the energy",
        )
        return parent_parser

    def setup(self, stage: Optional[str] = None):
        key = "cartesian_alpha"

        douglas_dataset_dict: Dict[str, Dict[str, torch.Tensor]] = torch.load(
            self.datafile_path
        )
        # column_perm_inds = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        input_tensor = (
            douglas_dataset_dict["coords"][key]
            .float()
            .clone()
            .detach()
            .requires_grad_(False)
        )
        output_tensor = (
            douglas_dataset_dict["force"][key]
            .float()
            .clone()
            .detach()
            .requires_grad_(False)
        )

        input_tensor = input_tensor.reshape(input_tensor.shape[:-1] + (3, 3))
        # appending additional 3 dimensions to conform with the data format in NODE paradigm
        input_tensor = torch.cat((input_tensor, torch.zeros_like(input_tensor)), dim=-1)
        output_tensor = output_tensor.reshape(output_tensor.shape[:-1] + (3, 3))
        print("input shape:", input_tensor.shape)
        print("output shape:", output_tensor.shape)

        if self.hparams.fit_energy:
            energy_tensor = (
                douglas_dataset_dict["energy"]
                .float()
                .clone()
                .detach()
                .requires_grad_(False)
            )
            print("energy shape:", energy_tensor.shape)
            (
                train_val_input,
                test_input,
                train_val_output,
                test_output,
                train_val_energy,
                test_energy,
            ) = train_test_split(
                input_tensor, output_tensor, energy_tensor, test_size=0.2, shuffle=True
            )
            (
                train_input,
                val_input,
                train_output,
                val_output,
                train_energy,
                val_energy,
            ) = train_test_split(
                train_val_input,
                train_val_output,
                train_val_energy,
                test_size=0.2,
                shuffle=True,
            )
            self.train_dataset = torch_data.TensorDataset(
                train_input, train_output, train_energy
            )
            self.val_dataset = torch_data.TensorDataset(
                val_input, val_output, val_energy
            )
            self.test_dataset = torch_data.TensorDataset(
                test_input, test_output, test_energy
            )
        else:
            (
                train_val_input,
                test_input,
                train_val_output,
                test_output,
            ) = train_test_split(
                input_tensor, output_tensor, test_size=0.2, shuffle=True
            )
            train_input, val_input, train_output, val_output = train_test_split(
                train_val_input, train_val_output, test_size=0.2, shuffle=True
            )
            self.train_dataset = torch_data.TensorDataset(train_input, train_output)
            self.val_dataset = torch_data.TensorDataset(val_input, val_output)
            self.test_dataset = torch_data.TensorDataset(test_input, test_output)

    def train_dataloader(self):
        return torch_data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=32,
        )

    def val_dataloader(self):
        return torch_data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=32,
        )

    def test_dataloader(self):
        return torch_data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=32,
        )

    # def transfer_batch_to_device(self, batch, device, dataloader_idx):
    #     [b.to_torch(device=device) for b in batch]
    #     return batch

    # def on_after_batch_transfer(self, batch, dataloader_idx):
    #     result = []
    #     for b in batch:
    #         if self.perturb:
    #             b["state"] += torch.randn_like(b["state"]) * self.perturb
    #         result.append(dict(b))
    #     return result


def SMAPE(target: torch.Tensor, pred: torch.Tensor):
    loss = 2 * (pred - target).abs() / (pred.abs() + target.abs() + 1e-8)
    return loss


class DouglasForceRegressor(pl.LightningModule):
    def __init__(
        self,
        net_type: str,
        dim_per_layer: int,
        n_layers: int,
        lr: float,
        weight_decay: float,
        pretrained_net: Optional[str] = None,
        fit_energy: bool = False,
        energy_loss_coef: float = 0,
        cvx_penalty: float = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        net_type_dict = {
            "ESNN": node_f.ODEFunc,
            "ESNN2": node_f.ODEFunc2,
            "quadratic": node_f.ODEFuncQuadratic,
            "CVXESNN2": node_f.ODEFunc2Cvx,
            "PCVXESNN2": node_f.ODEFunc2PartialCvx,
            # "EMLP": emlp_models.SlinkyForceODEFunc,
        }
        self.boundaries = (1, 1)
        assert net_type in net_type_dict, f"Unrecognized net_type: {net_type}"
        self.net = net_type_dict[net_type](
            NeuronsPerLayer=dim_per_layer,
            NumLayer=n_layers,
            Boundaries=self.boundaries,
        )
        if pretrained_net:
            self._load_pretrained_net(self.net, pretrained_net)
        self.register_buffer("feat_weights", torch.tensor([1e2, 1e2, 1]))

    def _anneal_feat_weights(self):
        if self.feat_weights[2] < 100:
            self.feat_weights[2] += 1

    def on_epoch_end(self) -> None:
        self._anneal_feat_weights()

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("SlinkyTrajectoryRegressor")
        parser.add_argument("--dim_per_layer", type=int, default=32)
        parser.add_argument("--n_layers", type=int, default=5)
        parser.add_argument(
            "--lr", type=float, help="The learning rate of the meta steps"
        )
        parser.add_argument(
            "--weight_decay", type=float, help="Weight decay of optimizer"
        )
        parser.add_argument(
            "--energy_loss_coef",
            type=float,
            help="The coeffcient of energy fitting loss",
        )
        parser.add_argument(
            "--cvx_penalty",
            type=float,
            default=0,
            help="The coeffcient of convexity penalty",
        )
        parser.add_argument(
            "--net_type",
            default="ESNN",
            choices=["ESNN", "EMLP", "ESNN2", "quadratic", "CVXESNN2", "PCVXESNN2"],
            type=str,
            help="The type of force prediction network",
        )
        return parent_parser

    def forward(
        self,
        input: torch.Tensor,
        truth: Optional[torch.Tensor] = None,
        energy_truth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward

        Args:
            input (torch.Tensor): TensorType["batch", "cycles", "coords":6]

        Returns:
            torch.Tensor: TensorType["batch", "cycles", "forces":3]
        """
        output = self.net(input)
        # energy = 0
        loss = 0
        if self.hparams.fit_energy:
            energy = self.net.cal_energy(input[..., :3])
            if energy_truth is not None:
                # print(f"{energy_truth.shape=}")
                # print(f"{energy.shape=}")
                # print(f"{input.shape=}")
                loss += self.hparams.energy_loss_coef * self.energy_criterion(
                    energy_truth, energy.sum(dim=-1)
                )
        if truth is not None:
            loss += self.criterion(truth, output)

        if self.hparams.cvx_penalty:
            loss += self.hparams.cvx_penalty * self.net.net.compute_cvx_penalty()
        return {"output": output, "loss": loss}

    def _trim_boundaries(self, x: torch.Tensor) -> torch.Tensor:
        """Trim the cycles according to the boundary conditions

        Args:
            x (torch.Tensor): TensorType["batch", "cycles", "forces":3]

        Returns:
            torch.Tensor: TensorType["batch", "trimmed_cycles", "forces":3]
        """
        if self.boundaries[0] == 1:
            x = x[..., 1:, :]
        if self.boundaries[1] == 1:
            x = x[..., :-1, :]
        assert x.nelement() > 0, "Force/coord tensor trimmed to empty"
        return x

    def criterion(
        self,
        force_truth: torch.Tensor,
        force_prediction: torch.Tensor,
    ) -> torch.Tensor:
        """Loss

        Args:
            force_prediction (torch.Tensor): prediction. TensorType["batch", "cycles", "forces":3]
            force_truth (torch.Tensor): ground truth. TensorType["batch", "cycles", "forces":3]

        Returns:
            torch.Tensor: loss
        """
        force_truth = self._trim_boundaries(force_truth)
        force_prediction = self._trim_boundaries(force_prediction)
        squared_error = (force_prediction - force_truth) ** 2
        # if self.boundaries[0] == 1:
        #     squared_error[..., 0, :] = 0
        # if self.boundaries[1] == 1:
        #     squared_error[..., -1, :] = 0
        squared_error = squared_error * self.feat_weights
        return squared_error.mean()

    def energy_criterion(
        self, energy_truth: torch.Tensor, energy_prediction: torch.Tensor
    ) -> torch.Tensor:
        return torch.mean((energy_truth - energy_prediction) ** 2)

    def training_step(self, batch, batch_idx):
        if self.hparams.fit_energy:
            input, force_truth, energy_truth = batch
            result = self.forward(input, force_truth, energy_truth)
        else:
            input, force_truth = batch
            result = self.forward(input, force_truth)
        loss = result["loss"]
        force_prediction = result["output"]
        self.log(
            "smape/train/x",
            SMAPE(force_truth[..., 0], force_prediction[..., 0]).mean(),
            logger=True,
        )
        self.log(
            "smape/train/y",
            SMAPE(force_truth[..., 1], force_prediction[..., 1]).mean(),
            logger=True,
        )
        self.log(
            "smape/train/a",
            SMAPE(force_truth[..., 2], force_prediction[..., 2]).mean(),
            logger=True,
        )
        self.log("loss/train", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.hparams.fit_energy:
            input, force_truth, energy_truth = batch
            result = self.forward(input, force_truth, energy_truth)
        else:
            input, force_truth = batch
            result = self.forward(input, force_truth)
        loss = result["loss"]
        self.log("loss/val", loss, prog_bar=True, logger=True)
        if batch_idx % 90 == 0:
            force_truth = self._trim_boundaries(force_truth).flatten(start_dim=-2)
            force_prediction = self._trim_boundaries(result["output"]).flatten(
                start_dim=-2
            )
            var_names = [
                f"{v}_{i}"
                for i, v in itertools.product(
                    range(force_truth.shape[1] // 3), ["x", "y", "a"]
                )
            ]
            truth_df = pd.DataFrame(
                force_truth.detach().cpu().numpy(),
                columns=[f"true_force_{v}" for v in var_names],
            )
            pred_df = pd.DataFrame(
                force_prediction.detach().cpu().numpy(),
                columns=[f"pred_force_{v}" for v in var_names],
            )
            df = pd.concat([truth_df, pred_df], axis=1)
            for var in var_names:
                scatter = plot_regression_scatter(
                    df, f"true_force_{var}", f"pred_force_{var}", f"scatter of {var}"
                )
                wandb.log({f"scatter/{var}": scatter})
        return loss

    def test_step(self, batch, batch_idx):
        if self.hparams.fit_energy:
            input, force_truth, energy_truth = batch
            result = self.forward(input, force_truth, energy_truth)
        else:
            input, force_truth = batch
            result = self.forward(input, force_truth)
        loss = result["loss"]
        self.log("loss/test", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=10, verbose=True, min_lr=1e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "loss/val",
        }

    def _load_pretrained_net(self, net: torch.nn.Module, pretrained_checkpoint: str):
        state_dict = torch.load(pretrained_checkpoint, map_location=self.device)
        net.load_state_dict(state_dict)
        return net


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(
        default_config_files=["pretrain_default_config.yaml"],
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    parser.add_argument("--seed", type=int, default=0, help="The random seed")
    parser.add_argument(
        "--log", action="store_true", default=False, help="Whether to enable the logger"
    )
    parser.add_argument("--name", type=str, help="Run name")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument(
        "--test_only",
        action="store_true",
        default=False,
        help="Only evaluate the model. (Not very meaningful if not loading a trained checkpoint)",
    )

    # pos_inds = [0, 1, 3, 4, 6, 7]
    # rot_inds = [2, 5, 8]

    DouglasForceDataModule.add_argparse_args(parser)
    DouglasForceRegressor.add_argparse_args(parser)
    pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    data_module = DouglasForceDataModule.from_argparse_args(args)
    data_module.setup()

    if args.log:
        logger = pl_loggers.WandbLogger(
            name=args.name, project="slinky-pretrain-lightning"
        )
    else:
        logger = pl_loggers.WandbLogger(
            name=args.name, project="slinky-pretrain-lightning", mode="disabled"
        )

    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        dirpath=os.path.join("checkpoints", wandb.run.name),
        # filename="best-checkpoint",
        save_top_k=1,
        save_last=True,
        verbose=True,
        monitor="loss/val",
        mode="min",
    )
    early_stopping_callback = pl_callbacks.EarlyStopping(
        monitor="loss/val", patience=40
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    if args.checkpoint:
        model = DouglasForceRegressor.load_from_checkpoint(args.checkpoint)
    else:
        model = DouglasForceRegressor(**vars(args))

    if not args.test_only:
        trainer.fit(model=model, datamodule=data_module)
