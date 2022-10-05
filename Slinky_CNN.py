import os
import random
from multiprocessing.sharedctypes import Value
import sys
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import pytorch_forecasting
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from torch import nn

import wandb
from neural_slinky import trajectory_dataset
from neural_slinky.utils import animate_2d_slinky_trajectories
from priority_memory import batch


class SlinkyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_trajectories,
        test_trajectories,
        input_length,
        target_length,
        batch_size=8,
        perturb: float = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_trajectories = train_trajectories
        self.test_trajectories = test_trajectories
        self.batch_size = batch_size
        self.input_length = input_length
        self.target_length = target_length
        self.perturb = perturb

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = trajectory_dataset.TrajectoryDataset(
            self.train_trajectories,
            input_length=self.input_length,
            target_length=self.target_length,
        )
        self.test_dataset = trajectory_dataset.TrajectoryDataset(
            self.test_trajectories,
            input_length=self.input_length,
            target_length=self.target_length,
        )

    def train_dataloader(self):
        return self.train_dataset.to_dataloader(self.batch_size, "random")

    def val_dataloader(self):
        return self.test_dataset.to_dataloader(self.batch_size, "random")

    def test_dataloader(self):
        return self.test_dataset.to_dataloader(self.batch_size, "random")

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        [b.to_torch(device=device) for b in batch]
        return batch

    def on_after_batch_transfer(self, batch, dataloader_idx):
        result = []
        for b in batch:
            if self.perturb:
                b["state"] += torch.randn_like(b["state"]) * self.perturb
            result.append(dict(b))
        return result


# class SlinkyRNN(pl.LightningModule):
#     def __init__(self, encoder_mlp_dims, decoder_mlp_dims, num_lstm_layers) -> None:
#         super().__init__()
#         encoder_layers = self._make_fc_layers(encoder_mlp_dims)
#         self.encoder_mlp = nn.Sequential(encoder_layers)
#         decoder_layers = self._make_fc_layers(decoder_mlp_dims)
#         self.decoder_mlp = nn.Sequential(decoder_layers)
#         self.lstm = nn.LSTM(
#             input_size=encoder_mlp_dims[-1],
#             hidden_size=decoder_mlp_dims[0],
#             num_layers=num_lstm_layers,
#             batch_first=True,
#         )

#     @staticmethod
#     def _make_fc_layers(layer_dim_list: List[List[int]]):
#         layers = []
#         for i, (in_size, out_size) in enumerate(layer_dim_list):
#             if i != len(layer_dim_list) - 1:
#                 layers.append(
#                     torch.nn.Sequential(
#                         torch.nn.Linear(in_size, out_size),
#                         torch.nn.ReLU(),
#                         # torch.nn.BatchNorm1d(out_size),
#                     )
#                 )
#             else:
#                 layers.append(torch.nn.Linear(in_size, out_size))
#         return layers


class CenteringLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("averaging_kernel", torch.nn.Parameter(torch.tensor([])))

    def forward(self, slinky_states: torch.Tensor):
        """Center the x, y for input triplet

        Args:
            slinky_states (torch.Tensor): TensorType[..., "state": 6, "cycles"]

        Returns:
            centered_slinky_states (torch.Tensor): TensorType[..., "state": 6, "cycles"]
        """
        pass
        # return slinky_states - 

class SlinkyCNN(nn.Module):
    def __init__(self, dim_list: List[int], residual=True):
        super(SlinkyCNN, self).__init__()
        # dim_list = [in_dim] + dim_list + [out_dim]
        dim_list = [[dim_list[i], dim_list[i + 1]] for i in range(len(dim_list) - 1)]
        self.layer_list = self._make_cnn_layers(dim_list)
        self.residual = residual
        for i, layer in enumerate(self.layer_list):
            self.add_module(f"layer_{i}", layer)

    def forward(self, x: torch.Tensor):
        """Forward

        Args:
            x (torch.Tensor): (batch x slinky_cycle x dof)

        Returns:
            torch.Tensor: prediction of the coords at next step
        """
        for i, layer in enumerate(self.layer_list):
            out = layer(x)
            if self.residual and out.shape == x.shape:
                out = out + x
            x = out
        return out

    @staticmethod
    def _make_cnn_layers(layer_dim_list: List[List[int]]):
        layers = []
        for i, (in_size, out_size) in enumerate(layer_dim_list):
            if i != len(layer_dim_list) - 1:
                layers.append(
                    torch.nn.Sequential(
                        torch.nn.Conv1d(
                            in_channels=in_size,
                            out_channels=out_size,
                            kernel_size=3,
                            padding=1,
                            padding_mode="replicate",
                        ),
                        torch.nn.GELU(),
                        # torch.nn.BatchNorm1d(out_size),
                    )
                    # torch.nn.Sequential(
                    #     torch.nn.Conv1d(
                    #         in_channels=in_size,
                    #         out_channels=3 * out_size,
                    #         kernel_size=3,
                    #         padding=0,
                    #     ),
                    #     torch.nn.ConvTranspose1d(
                    #         in_channels=3 * out_size,
                    #         out_channels=out_size,
                    #         kernel_size=3,
                    #     ),
                    #     torch.nn.GELU(),
                    #     # torch.nn.BatchNorm1d(out_size),
                    # )
                )
            else:
                layers.append(
                    torch.nn.Conv1d(
                        in_channels=in_size,
                        out_channels=out_size,
                        kernel_size=3,
                        padding=1,
                    )
                )
        return layers


class SlinkyTrajectoryRegressor(pl.LightningModule):
    def __init__(
        self,
        dim_list: List[int],
        residual: bool,
        teacher_forcing: float = 0,
        lr: float = 1e-4,
        weight_decay: float = 0,
        boundaries: Sequence[int] = (1, 1),
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = SlinkyCNN(dim_list=dim_list, residual=residual)
        # self.criterion = nn.MSELoss()
        self.teacher_forcing: float = teacher_forcing
        self.register_buffer("feat_weights", torch.tensor([1e2, 1e2, 1, 1e2, 1e2, 1]))
        self.boundaries = boundaries

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("SlinkyTrajectoryRegressor")
        parser.add_argument("--dim_list", type=int, nargs="+")
        parser.add_argument("--boundaries", type=int, nargs="+")
        parser.add_argument("--residual", type=bool, default=True)
        parser.add_argument("--teacher_forcing", type=float, default=1)
        parser.add_argument(
            "--lr", type=float, help="The learning rate of the meta steps"
        )
        parser.add_argument(
            "--weight_decay", type=float, help="Weight decay of optimizer"
        )
        return parent_parser

    def criterion(self, x, y):
        return torch.mean((torch.max(torch.abs(x - y), dim=-1)[0] * self.feat_weights))

    def _recenter_coords(self, input: torch.Tensor) -> torch.Tensor:
        """Remove translation

        Args:
            input (torch.Tensor): TensorType["batches", "time", "coords", "cycles"]

        Returns:
            recentered_input (torch.Tensor): TensorType["batches", "time", "coords", "cycles"]
        """
        x = input[..., [0, 3], :]
        x_center = torch.mean(x, dim=-2, keepdim=True)
        x -= x_center
        y = input[..., [0, 3], :]
        y_center = torch.mean(y, dim=-2, keepdim=True)
        y -= y_center
        return input

    def forward_step(self, input: torch.Tensor):
        try:
            n_batch, n_time, n_feat, n_cycle = input.shape
        except:
            raise ValueError(
                f"The shape of the input is expected to be (n_batch, n_time, n_feat, n_cycle). Got {input.shape=}"
            )
        # centered_input = self._recenter_coords(input)
        centered_input = input
        output: torch.Tensor = self.model(
            centered_input.flatten(start_dim=0, end_dim=1)
        )
        output = output.reshape(n_batch, n_time, n_feat, n_cycle)
        output += input
        if self.boundaries[0] == 1:
            output[..., 0] = input[..., 0]
        if self.boundaries[1] == 1:
            output[..., -1] = input[..., -1]
        return output

    def forward_autoregressive(self, start: torch.Tensor, n_steps: int):
        try:
            n_batch, n_feat, n_cycle = start.shape
        except:
            raise ValueError(
                f"The shape of the input is expected to be (n_batch, n_feat, n_cycle). Got {start.shape=}"
            )
        output = []
        start = start.unsqueeze(1)
        for _ in range(n_steps):
            start = self.forward_step(start)
            output.append(start)
        return torch.concat(output, dim=1)

    def forward(self, input: torch.Tensor, true_output: Optional[torch.Tensor] = None):
        if true_output is None:
            loss = 0
            output = self.forward_step(input)
        else:
            use_teacher_forcing: bool = (
                self.teacher_forcing and self.teacher_forcing > random.random()
            )
            if use_teacher_forcing:
                # print(f"{input.shape=}")
                # print(f"{true_output.shape=}")
                input = torch.cat((input, true_output[:, :-1]), dim=1)
                output = self.forward_step(input)
            else:
                output = self.forward_autoregressive(input[:, 0], true_output.shape[1])
            loss = self.criterion(output, true_output)
        return loss, output

    def training_step(self, batch, batch_idx):
        input, true_output = batch[0]["state"], batch[1]["state"]
        self.train()
        loss, output = self.forward(input, true_output)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input, true_output = batch[0]["state"], batch[1]["state"]
        self.eval()
        loss, output = self.forward(input, true_output)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input, true_output = batch[0]["state"], batch[1]["state"]
        self.eval()
        loss, output = self.forward(input, true_output)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


class TeacherForcingStepper(pl.Callback):
    def __init__(
        self,
        init_teacher_forcing: float,
        final_teacher_forcing: float,
        teacher_forcing_anneal_epochs: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.init_ratio: float = init_teacher_forcing
        self.final_ratio: float = final_teacher_forcing
        self.epochs = teacher_forcing_anneal_epochs
        self.step_size = (
            final_teacher_forcing - init_teacher_forcing
        ) / teacher_forcing_anneal_epochs
        self._teacher_forcing_ratio = init_teacher_forcing
        self.counter = 0

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("TeacherForcingStepper")
        parser.add_argument("--init_teacher_forcing", type=float)
        parser.add_argument("--final_teacher_forcing", type=float)
        parser.add_argument("--teacher_forcing_anneal_epochs", type=float)
        return parent_parser

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        pl_module.log(
            "teacher_forcing", self._teacher_forcing_ratio, prog_bar=True, logger=True
        )
        if self.counter < self.epochs:
            pl_module.teacher_forcing = self._teacher_forcing_ratio
            self._teacher_forcing_ratio += self.step_size
            self.counter += 1
            # trainer.test_dataloaders[0].dataset.target_length += 1


class ClipLengthStepper(pl.Callback):
    def __init__(
        self,
        init_clip_len: int,
        final_clip_len: int,
        anneal_clip_lenght_epochs: int,
        **kwargs,
    ) -> None:
        super().__init__()
        # self.freq: int = freq
        self.init_len: int = init_clip_len
        self.final_len: int = final_clip_len
        self.cur_len: int = init_clip_len
        self.epochs: int = anneal_clip_lenght_epochs
        self.steps = np.linspace(
            init_clip_len, final_clip_len, anneal_clip_lenght_epochs
        ).astype(int)
        self.counter = 0

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("ClipLengthStepper")
        parser.add_argument("--init_clip_len", type=int, default=1)
        parser.add_argument("--final_clip_len", type=int)
        parser.add_argument("--anneal_clip_lenght_epochs", type=int)
        return parent_parser

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.counter < self.epochs:
            self.cur_len = self.steps[self.counter]
            trainer.train_dataloader.dataset.datasets.target_length = self.cur_len
            trainer.val_dataloaders[0].dataset.target_length = self.cur_len
            self.counter += 1
        pl_module.log("clip_length", self.cur_len, prog_bar=True, logger=True)


if __name__ == "__main__":
    import configargparse

    parser = configargparse.ArgumentParser(
        default_config_files=["slinky_cnn_default_config.yaml"],
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )

    parser.add_argument(
        "--log", action="store_true", default=False, help="Whether to enable the logger"
    )
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--seed", type=int, default=0, help="The random seed")
    parser.add_argument("--name", type=str, help="Run name")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Checkpoint path")
    parser.add_argument(
        "--test_only",
        action="store_true",
        default=False,
        help="Only evaluate the model. (Not very meaningful if not loading a trained checkpoint)",
    )

    def read_data():
        # folder = "NeuralODE_Share2/SlinkyGroundTruth"
        folder = "slinky-is-sliding/SlinkyGroundTruth"
        num_cycles = 76
        true_y = np.loadtxt(folder + "/helixCoordinate_2D.txt", delimiter=",")
        true_v = np.loadtxt(folder + "/helixVelocity_2D.txt", delimiter=",")
        true_y = torch.from_numpy(true_y).float()
        true_v = torch.from_numpy(true_v).float()
        true_y = torch.reshape(true_y, (true_y.size()[0], num_cycles, 3))
        true_v = torch.reshape(true_v, (true_v.size()[0], num_cycles, 3))
        # print(true_v.shape)
        delta_t = 0.001
        time = np.arange(true_y.shape[0]) * delta_t
        # coord_names = []
        # vel_names = []
        # for i in range(80):
        #     coord_names.append(f"x_{i}")
        #     coord_names.append(f"y_{i}")
        #     coord_names.append(f"a_{i}")
        #     vel_names.append(f"xv_{i}")
        #     vel_names.append(f"yv_{i}")
        #     vel_names.append(f"av_{i}")
        # df = pd.DataFrame(
        #     np.concatenate((true_y, true_v), -1), columns=coord_names + vel_names
        # )
        # df["time"] = time
        # df.index.name = "step"
        # return df.reset_index(), coord_names + vel_names
        return (
            batch.Batch(
                state=torch.cat((true_y, true_v), axis=-1)
                .transpose(-2, -1)
                .contiguous(),
                time=time,
            ),
            num_cycles,
            delta_t,
        )

    SlinkyTrajectoryRegressor.add_argparse_args(parser)
    ClipLengthStepper.add_argparse_args(parser)
    TeacherForcingStepper.add_argparse_args(parser)
    pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    slinky_data, num_cycles, delta_t = read_data()

    # training_cutoff = 700
    training_cutoff = len(slinky_data)
    # training_cutoff = 65

    data_module = SlinkyDataModule(
        [slinky_data[:training_cutoff]],
        # [slinky_data[training_cutoff:]],
        [slinky_data[:training_cutoff]],
        input_length=1,
        target_length=args.init_clip_len,
        batch_size=args.batch_size,
        perturb=0.001,
    )
    data_module.setup()
    # print(f"{len(data_module.train_dataloader())=}")
    # print(f"{len(data_module.val_dataloader())=}")

    if args.log:
        logger = WandbLogger(name=args.name, project="slinky-CNN")
    else:
        logger = WandbLogger(name=args.name, project="slinky-CNN", mode="disabled")

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join("checkpoints", wandb.run.name),
        # filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=40)

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        enable_checkpointing=True,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            TeacherForcingStepper(**vars(args)),
            ClipLengthStepper(**vars(args)),
        ],
        # clipping gradients is a hyperparameter and important to prevent divergance
        # of the gradient for recurrent neural networks
        reload_dataloaders_every_n_epochs=1,
        # strategy="dp",
    )

    if args.checkpoint_path:
        model = SlinkyTrajectoryRegressor.load_from_checkpoint(**vars(args))
    else:
        model = SlinkyTrajectoryRegressor(**vars(args))

    if not args.test_only:
        trainer.fit(model=model, datamodule=data_module)

    start_point = slinky_data[0]["state"].unsqueeze(0)

    true_trajectory = (
        slinky_data["state"]
        .detach()
        .transpose(-2, -1)
        .reshape(-1, num_cycles, 2, 3)[..., 0, :]
    )

    ################################### TEMP ###################################
    true_trajectory, num_cycles, delta_t = (
        torch.load("40_vertical_still.pth"),
        40,
        0.001,
    )
    slinky_data = torch.cat(
        [true_trajectory, torch.zeros_like(true_trajectory)], dim=-1
    )
    slinky_data = slinky_data.transpose(-2, -1)
    start_point = slinky_data[0].unsqueeze(0)
    ################################### TEMP ###################################

    # print(start_point)
    # sys.exit()
    model.eval()
    output = model.forward_autoregressive(start_point, len(slinky_data) - 1).squeeze()
    # print(f"{output['prediction'].shape}=")
    # print(f"{slinky_data.shape=}")

    pred_trajectory = output.transpose(-2, -1).reshape(-1, num_cycles, 2, 3)[..., 0, :]
    pred_trajectory = torch.cat([true_trajectory[:1], pred_trajectory], dim=0)

    evaluate_animation = animate_2d_slinky_trajectories(
        torch.stack([pred_trajectory, true_trajectory]), delta_t
    )

    animation_html_file = "animation_tmp.html"
    evaluate_animation.write_html(animation_html_file, full_html=False)
    with open(animation_html_file, "r") as f:
        evaluate_animation_html = "\n".join(f.readlines())

    torch.save(pred_trajectory, "pred_trajectory.pth")
    torch.save(true_trajectory, "true_trajectory.pth")
    print(f"{pred_trajectory.shape=}")
    print(f"{true_trajectory.shape=}")

    wandb.log({"evaluate_animation": wandb.Html(evaluate_animation_html, inject=False)})
