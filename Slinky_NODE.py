import os
from random import choices
from typing import Callable, List, Optional, Tuple

import configargparse
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from torchdiffeq import odeint_adjoint as odeint

# from torchdiffeq import odeint as odeint

import wandb
from neural_slinky import trajectory_dataset, coords_transform
from neural_slinky.utils import (
    animate_2d_slinky_trajectories,
    get_model_device,
)
from priority_memory import batch
from slinky import func as node_f

# from neural_slinky import emlp_models


class SlinkyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_trajectories,
        test_trajectories,
        input_length,
        target_length,
        val_length,
        batch_size=8,
        perturb: float = 0,
    ):
        super().__init__()
        self.train_trajectories = train_trajectories
        self.test_trajectories = test_trajectories
        self.batch_size = batch_size
        self.input_length = input_length
        self.target_length = target_length
        self.test_length = val_length
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
            target_length=self.test_length,
        )

    def train_dataloader(self):
        return self.train_dataset.to_dataloader(
            self.batch_size, "random", drop_last=False
        )

    def val_dataloader(self):
        return self.test_dataset.to_dataloader(
            self.batch_size, "sequential", drop_last=False
        )

    def test_dataloader(self):
        return self.test_dataset.to_dataloader(
            self.batch_size, "sequential", drop_last=False
        )

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


class SlinkyTrajectoryRegressor(pl.LightningModule):
    def __init__(
        self,
        dim_per_layer: int,
        n_layers: int,
        kinetic_reg: float = 0,
        n_cycles_loss: int = 20,
        net_type: str = "ESNN",
        lr: float = 1e-3,
        weight_decay: float = 0,
        odeint_method: str = "dopri5",
        angular_loss_weight: float = 1,
        pretrained_net: Optional[str] = None,
        length_scaling: float = 1,
        loss_coord_frame: str = "global",
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        if net_type == "ESNN":
            self.net = node_f.ODEFunc(
                NeuronsPerLayer=dim_per_layer, NumLayer=n_layers, Boundaries=(1, 1)
            )
        elif net_type == "ESNN2":
            self.net = node_f.ODEFunc2(
                NeuronsPerLayer=dim_per_layer, NumLayer=n_layers, Boundaries=(1, 1)
            )
        elif net_type == "EMLP":
            self.net = emlp_models.SlinkyForceODEFunc(
                num_layers=n_layers, boundaries=(1, 1)
            )
        elif net_type == "quadratic":
            self.net = node_f.ODEFuncQuadratic(douglas=True, Boundaries=(1, 1))
            # self.net = node_f.ODEFuncQuadratic(douglas=False, boundaries=(1, 1))

        if pretrained_net:
            print("loading " + pretrained_net)
            self._load_pretrained_net(self.net, pretrained_net)
            print("loading " + pretrained_net + " done")

        self.model = node_f.ODEPhys(self.net)
        self.register_buffer(
            "feat_weights",
            torch.tensor(
                [1e2, 1e2, angular_loss_weight, 1e2, 1e2, angular_loss_weight]
            ),
        )
        self.atol = 1e-4
        self.rtol = 1e-4
        self.n_cycles_loss = n_cycles_loss
        self.kinetic_reg = kinetic_reg
        self.mse = torch.nn.MSELoss()
        self.length_scaling = length_scaling
        self.loss_coord_frame = loss_coord_frame

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("SlinkyTrajectoryRegressor")
        parser.add_argument("--dim_per_layer", type=int, default=32)
        parser.add_argument("--n_layers", type=int, default=5)
        parser.add_argument(
            "--kinetic_reg",
            type=float,
            default=0,
            help="The kinetic regularization coefficient",
        )
        parser.add_argument(
            "--n_cycles_loss", type=int, help="Number of cycles to calculate loss for"
        )
        parser.add_argument(
            "--net_type",
            default="ESNN",
            choices=["ESNN", "EMLP", "ESNN2", "quadratic"],
            type=str,
            help="The type of force prediction network",
        )
        parser.add_argument(
            "--odeint_method",
            default="dopri5",
            choices=[
                "dopri8",
                "dopri5",
                "bosh3",
                "fehlberg2",
                "adaptive_heun",
                "euler",
                "midpoint",
                "rk4",
                "explicit_adams",
                "implicit_adams",
                "fixed_adams",
                "scipy_solver",
            ],
            type=str,
            help="The odeint method",
        )
        parser.add_argument(
            "--lr", type=float, help="The learning rate of the meta steps"
        )
        parser.add_argument(
            "--angular_loss_weight",
            type=float,
            default=1,
            help="The weight of angular loss terms",
        )
        parser.add_argument(
            "--weight_decay", type=float, help="Weight decay of optimizer"
        )
        parser.add_argument(
            "--pretrained_net",
            type=str,
            default=None,
            help="Pretrained core network state dict path",
        )
        parser.add_argument(
            "--loss_coord_frame",
            type=str,
            choices=["global", "douglas"],
            default="global",
        )
        return parent_parser

    # def _kinetic_function(self, t, x):
    #     f = self.model(t, x)
    #     return (f**2).sum()
    # return (f**2).sum(dim=tuple(range(1, len(f.shape) + 1)))

    def _regularizer_augmented_model(
        self, t: torch.Tensor, aug_x: Tuple[torch.Tensor, torch.Tensor]
    ):
        x = aug_x[0]
        # e = aug_x[1]
        f = self.model(t, x)
        return (f, (f**2).mean(dim=0).sum())

    def _solve_dynamics(self, start, t: torch.Tensor, method):
        t = t[0]  # TODO: this only applies to evenly-spaced trajectories
        if self.kinetic_reg:
            start = (start, torch.tensor(0).type_as(start))
            output, regularization = self._odeint(
                start,
                t,
                method,
                fun=self._regularizer_augmented_model,
                adjoint_params=tuple(self.model.parameters()),
                adjoint_options=dict(norm="seminorm"),
            )
            duration = len(t)  # .detach_()
            # print(f"{t=}")
            # if duration == 0:
            #     duration = 1
            regularization = regularization[-1] / duration
        else:
            regularization = 0
            output = self._odeint(start, t, method, fun=self.model)
        output = output[1:].transpose(0, 1)
        return output, regularization

    def criterion(self, x, y):
        if self.loss_coord_frame == "global":
            return torch.mean(((x - y) * self.feat_weights) ** 2) / (
                self.length_scaling**2
            )
        elif self.loss_coord_frame == "douglas":
            return torch.mean((x - y) ** 2) / (self.length_scaling**2)

    def _odeint(
        self, start: torch.Tensor, t: torch.Tensor, method: str, fun: Callable, **kwargs
    ):
        pred_y = odeint(
            fun, start, t, atol=self.atol, rtol=self.rtol, method=method, **kwargs
        )
        # print(f"{start.shape=}")
        # print(f"{pred_y.shape=}")
        return pred_y

    def _convert_cartesian_alpha_to_douglas(self, state: torch.Tensor):
        """_summary_

        Args:
            state (torch.Tensor): TensorType[..., "cycles": n, "state": 6]
        Returns:
            douglas (torch.Tensor): TensorType[..., "cycles": n-1, "douglas_coords": 3]
        """
        coords = state[..., :3]
        coord_pairs = torch.stack((coords[..., :-1, :], coords[..., 1:, :]), dim=-2)
        return coords_transform.transform_cartesian_alpha_to_douglas_single(coord_pairs)

    def forward(
        self,
        input: torch.Tensor,
        t: torch.Tensor,
        true_output: Optional[torch.Tensor] = None,
        # method: str = "dopri5",
        method: str = None,
        regularization: bool = True,
    ):
        start = input[:, 0]

        if method is None:
            method = self.hparams.odeint_method

        output, regularization = self._solve_dynamics(start, t, method)
        # print(f"{output.shape=}")
        # print(f"{true_output.shape=}")

        if true_output is None:
            return {"loss": 0, "output": output, "mse": 0, "kinetic_regularization": 0}
        else:
            mse = self.mse(output, true_output) / (args.length_scaling**2)
            if self.loss_coord_frame == "douglas":
                select_cycles = torch.randperm(output.shape[-2] - 1)[
                    : self.n_cycles_loss
                ]
                loss_output = self._convert_cartesian_alpha_to_douglas(output)
                loss_true_output = self._convert_cartesian_alpha_to_douglas(true_output)
            elif self.loss_coord_frame == "global":
                select_cycles = torch.randperm(input.shape[-2])[: self.n_cycles_loss]
                loss_output = output[..., select_cycles, :]
                loss_true_output = true_output[..., select_cycles, :]
            loss = self.criterion(
                loss_output,
                loss_true_output,
            )
            if regularization:
                loss = loss + self.kinetic_reg * regularization
        return {
            "loss": loss,
            "output": output,
            "mse": mse,
            "kinetic_regularization": regularization,
        }

    def training_step(self, batch, batch_idx):
        input, true_output = (
            batch[0]["state"],
            batch[1]["state"],
        )
        time = torch.cat([batch[0]["time"], batch[1]["time"]], axis=1)
        self.train()
        result = self.forward(input, time, true_output)
        output = result["output"]
        loss = result["loss"]
        mse = result["mse"]
        kinetic_regularization = result["kinetic_regularization"]
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_mse", mse, prog_bar=True, logger=True)
        if self.kinetic_reg:
            self.log(
                "kinetic_regularization",
                kinetic_regularization,
                prog_bar=True,
                logger=True,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        input, true_output = batch[0]["state"], batch[1]["state"]
        # print(f"{true_output.shape=}")
        time = torch.cat([batch[0]["time"], batch[1]["time"]], axis=1)
        self.eval()
        result = self.forward(input, time, true_output, regularization=False)
        loss = result["mse"]
        self.log("val_mse", loss, prog_bar=True, logger=True)
        for n, p in self.named_parameters():
            self.log(f"params/{n}", p, prog_bar=False, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input, true_output = batch[0]["state"], batch[1]["state"]
        time = torch.cat([batch[0]["time"], batch[1]["time"]], axis=1)
        self.eval()
        result = self.forward(input, time, true_output, regularization=False)
        loss = result["mse"]
        self.log("test_mse", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        # return torch.optim.AdamW(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return torch.optim.Adam(
            # return torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    def _load_pretrained_net(self, net: torch.nn.Module, pretrained_checkpoint: str):
        try:
            import pretrain_lightning
            print(f"Loading pretrained model from {pretrained_checkpoint}")
            pretrained_model = pretrain_lightning.DouglasForceRegressor.load_from_checkpoint(pretrained_checkpoint)
            net.load_state_dict(pretrained_model.net.state_dict())
        except:
            print("Load pretrained model failed. Trying to load it as a simple state_dict")
            state_dict = torch.load(pretrained_checkpoint, map_location=self.device)
            net.load_state_dict(state_dict)
        return net


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
        pl_module.log("clip_length", self.cur_len, prog_bar=True, logger=True)
        if self.counter < self.epochs:
            self.cur_len = self.steps[self.counter]
            trainer.train_dataloader.dataset.datasets.target_length = self.cur_len
            trainer.val_dataloaders[0].dataset.target_length = self.cur_len
            self.counter += 1


class ClipVisualization(pl.Callback):
    def __init__(
        self,
        vis_freq: int,
        vis_num: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.freq: int = vis_freq
        self.num: int = vis_num
        self.counter = 0

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("ClipVisualization")
        parser.add_argument("--vis_freq", type=int, default=1)
        parser.add_argument("--vis_num", type=int, default=1)
        return parent_parser

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.counter == 0:
            wandb.log({"val_animation": wandb.Html(animation_html, inject=False)})
        self.counter = (self.counter + 1) % self.freq


# class ClipLengthStepper(pl.Callback):
#     def __init__(self, freq: int, max_len: int) -> None:
#         super().__init__()
#         self.freq: int = freq
#         self.max_len: int = max_len
#         self.counter = 0

#     def on_train_epoch_end(
#         self, trainer: pl.Trainer, pl_module: pl.LightningModule
#     ) -> None:
#         self.counter = (self.counter + 1) % self.freq
#         if (
#             self.counter == 0
#             and trainer.train_dataloader.dataset.datasets.target_length < self.max_len
#         ):
#             trainer.train_dataloader.dataset.datasets.target_length += 1
#             trainer.val_dataloaders[0].dataset.target_length += 1
#             # trainer.test_dataloaders[0].dataset.target_length += 1
#         return


def read_data(
    delta_t: float = 1e-3,
    length_scaling: float = 1,
    down_sampling: int = 10,
    **kwargs,
):
    # folder = "NeuralODE_Share2/SlinkyGroundTruth"
    folder = "slinky-is-sliding/SlinkyGroundTruth"
    num_cycles = 76
    true_y = np.loadtxt(folder + "/helixCoordinate_2D.txt", delimiter=",")
    true_v = np.loadtxt(folder + "/helixVelocity_2D.txt", delimiter=",")
    true_y = torch.from_numpy(true_y).float()
    true_v = torch.from_numpy(true_v).float()
    true_y = torch.reshape(true_y, (true_y.size()[0], num_cycles, 3))
    true_v = torch.reshape(true_v, (true_v.size()[0], num_cycles, 3))

    y_scale = true_y.std(dim=(0, 1))
    v_scale = true_v.std(dim=(0, 1))
    # true_y /= y_scale
    # true_v /= v_scale
    # print(f"{true_y.shape=}")
    # print(f"{true_v.shape=}")
    print(f"{y_scale=}")
    print(f"{v_scale=}")

    true_delta_t = 1e-3
    # delta_t = args.delta_t
    true_v = length_scaling * true_v * true_delta_t / delta_t
    true_y = length_scaling * true_y
    # delta_t = 1.0

    time = np.arange(true_y.shape[0]) * delta_t
    return (
        batch.Batch(
            state=torch.cat((true_y, true_v), axis=-1)[::down_sampling],
            time=time[::down_sampling],
        ),
        num_cycles,
        y_scale,
        v_scale,
    )


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(
        default_config_files=["node_default_config.yaml"],
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )

    parser.add_argument(
        "--log", action="store_true", default=False, help="Whether to enable the logger"
    )
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--shuffle", type=bool, help="Whether to shuffle the data")
    # parser.add_argument("--grad_clip", type=float, default=0, help="Gradient clipping")
    # parser.add_argument("--min_epochs", type=int, help="Min training epochs")
    parser.add_argument("--seed", type=int, default=0, help="The random seed")
    # parser.add_argument("--devices", action="append", type=int, help="Torch device")
    parser.add_argument(
        "--incr_freq", type=int, help="Clip length increment frequency in epochs"
    )
    parser.add_argument("--max_length", type=int, help="Max clip length")
    parser.add_argument("--val_length", type=int, help="Max clip length")
    parser.add_argument("--init_length", type=int, help="Initial clip length")
    parser.add_argument(
        "--down_sampling", type=int, default=10, help="Data downsampling"
    )
    parser.add_argument(
        "--delta_t", type=float, default=0.001, help="Delta t assumed in data"
    )
    parser.add_argument(
        "--length_scaling",
        type=float,
        default=1,
        help="Scaling of length assumed in data",
    )
    parser.add_argument(
        "--perturb",
        type=float,
        default=0,
        help="Gaussian perturbation augmentation scale",
    )

    parser.add_argument("--name", default=None, type=str, help="Run name")
    parser.add_argument(
        "--project_name", default="NeuralSlinky", type=str, help="Run name"
    )

    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument(
        "--test_only",
        action="store_true",
        default=False,
        help="Only evaluate the model. (Not very meaningful if not loading a trained checkpoint)",
    )

    ClipLengthStepper.add_argparse_args(parser)
    SlinkyTrajectoryRegressor.add_argparse_args(parser)
    pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    slinky_data, num_cycles, y_scale, v_scale = read_data(**vars(args))

    pl.seed_everything(args.seed)
    training_cutoff_ratio = 0.8
    training_cutoff = int(len(slinky_data) * training_cutoff_ratio)
    # training_cutoff = 65

    data_module = SlinkyDataModule(
        [slinky_data[:training_cutoff]],
        [slinky_data[training_cutoff:]],
        # [slinky_data[:training_cutoff]],
        input_length=1,
        target_length=args.init_length,
        val_length=args.val_length,
        batch_size=args.batch_size,
        perturb=args.perturb,  # .01
    )
    data_module.setup()
    print(len(data_module.train_dataloader()))
    print(len(data_module.val_dataloader()))

    if args.log:
        logger = WandbLogger(args.name, project=args.project_name)
    else:
        logger = WandbLogger(args.name, project=args.project_name, mode="disabled")

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join("checkpoints", wandb.run.name),
        # filename="best-checkpoint",
        save_top_k=1,
        save_last=True,
        verbose=True,
        monitor="val_mse",
        mode="min",
    )

    early_stopping_callback = EarlyStopping(monitor="val_mse", patience=4)

    wandb.config.update(args)

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        enable_checkpointing=True,
        reload_dataloaders_every_n_epochs=1,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            # ClipLengthStepper(freq=args.incr_freq, max_len=args.max_length),
            ClipLengthStepper(**vars(args)),
        ],
        #     # profiler="advanced",
        #     # fast_dev_run=True
    )

    if args.checkpoint:
        model = SlinkyTrajectoryRegressor.load_from_checkpoint(
            args.checkpoint  # , dim_per_layer=32, n_layers=5
        )
        # model.load_state_dict(checkpoint.state_dict())
    else:
        model = SlinkyTrajectoryRegressor(**vars(args))

    if not args.test_only:
        trainer.fit(model=model, datamodule=data_module)

    start_point = slinky_data[0]["state"][None, None, :]
    # print(start_point)
    # sys.exit()
    model.eval()
    evaluate_time = torch.from_numpy(slinky_data["time"]).to(get_model_device(model))

    true_trajectory = (
        slinky_data["state"].detach().reshape(-1, num_cycles, 2, 3)[1:, :, 0, :]
    )

    result = model.forward(start_point, evaluate_time.unsqueeze(0), method="dopri5")
    output = result["output"]
    output = output.squeeze()

    pred_trajectory = output.reshape(-1, num_cycles, 2, 3)[..., 0, :]

    evaluate_animation = animate_2d_slinky_trajectories(
        torch.stack([true_trajectory, pred_trajectory]), 0.01
    )

    animation_html_file = "animation_tmp.html"
    evaluate_animation.write_html(animation_html_file, full_html=False)
    with open(animation_html_file, "r") as f:
        evaluate_animation_html = "\n".join(f.readlines())

    torch.save(pred_trajectory, "pred_trajectory.pth")
    torch.save(true_trajectory[1:], "true_trajectory.pth")

    wandb.log({"evaluate_animation": wandb.Html(evaluate_animation_html, inject=False)})
