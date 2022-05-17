import os
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

import wandb
from neural_slinky import trajectory_dataset
from neural_slinky.utils import (
    animate_2d_slinky_trajectories,
    get_model_device,
)
from priority_memory import batch
from slinky import func as node_f

parser = configargparse.ArgumentParser(
    default_config_files=["node_default_config.yaml"],
    config_file_parser_class=configargparse.YAMLConfigFileParser,
)

parser.add_argument(
    "--log", action="store_true", default=False, help="Whether to enable the logger"
)
parser.add_argument("--batch_size", type=int, help="Batch size")
parser.add_argument("--shuffle", type=bool, help="Whether to shuffle the data")
parser.add_argument("--lr", type=float, help="The learning rate of the meta steps")
parser.add_argument("--grad_clip", type=float, default=0, help="Gradient clipping")
parser.add_argument("--weight_decay", type=float, help="Weight decay of optimizer")
parser.add_argument("--min_epochs", type=int, help="Min training epochs")
parser.add_argument("--seed", type=int, default=0, help="The random seed")
parser.add_argument("--devices", action="append", type=int, help="Torch device")
parser.add_argument(
    "--incr_freq", type=int, help="Clip length increment frequency in epochs"
)
parser.add_argument("--max_length", type=int, help="Max clip length")
parser.add_argument("--init_length", type=int, help="Initial clip length")
parser.add_argument(
    "--perturb", type=float, default=0, help="Gaussian perturbation augmentation scale"
)
parser.add_argument(
    "--kinetic_reg", type=float, help="The kinetic regularization coefficient"
)
parser.add_argument("--name", default="NODE", type=str, help="Run name")

args = parser.parse_args()


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

    y_scale = true_y.std(dim=(0, 1))
    v_scale = true_v.std(dim=(0, 1))
    # true_y /= y_scale
    # true_v /= v_scale
    # print(f"{true_y.shape=}")
    # print(f"{true_v.shape=}")
    print(f"{y_scale=}")
    print(f"{v_scale=}")

    delta_t = 0.001
    # delta_t = 1.0

    time = np.arange(true_y.shape[0]) * delta_t
    return (
        batch.Batch(
            state=torch.cat((true_y, true_v), axis=-1)[::10],
            time=time[::10],
        ),
        num_cycles,
        y_scale,
        v_scale,
    )


slinky_data, num_cycles, y_scale, v_scale = read_data()


class SlinkyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_trajectories,
        test_trajectories,
        input_length,
        target_length,
        test_length,
        batch_size=8,
        perturb: float = 0,
    ):
        super().__init__()
        self.train_trajectories = train_trajectories
        self.test_trajectories = test_trajectories
        self.batch_size = batch_size
        self.input_length = input_length
        self.target_length = target_length
        self.test_length = test_length
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
        kinetic_regularization: float = 0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.net = node_f.ODEFunc(NeuronsPerLayer=dim_per_layer, NumLayer=n_layers)
        self.model = node_f.ODEPhys(self.net)
        self.register_buffer("feat_weights", torch.tensor([1e2, 1e2, 1, 1e2, 1e2, 1]))
        self.atol = 1e-4
        self.rtol = 1e-4
        self.n_cycles_loss = 20
        self.kinetic_regularization = kinetic_regularization
        self.mse = torch.nn.MSELoss()

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
        return (f, (f**2).sum())

    def _solve_dynamics(self, start, t, method):
        if self.kinetic_regularization:
            start = (start, torch.tensor(0).type_as(start))
            output, regularization = self.odeint(
                start,
                t,
                method,
                fun=self._regularizer_augmented_model,
                adjoint_params=tuple(self.model.parameters()),
            )
            regularization = regularization[-1]
        else:
            regularization = 0
            output = self.odeint(start, t, method, fun=self.model)
        output = output[1:].transpose(0, 1)
        return output, regularization

    def criterion(self, x, y):
        return torch.mean(((x - y) * self.feat_weights) ** 2)

    def odeint(
        self, start: torch.Tensor, t: torch.Tensor, method: str, fun: Callable, **kwargs
    ):
        t = t[0]  # TODO: this only applies to evenly-spaced trajectories
        pred_y = odeint(
            fun, start, t, atol=self.atol, rtol=self.rtol, method=method, **kwargs
        )
        # print(f"{start.shape=}")
        # print(f"{pred_y.shape=}")
        return pred_y

    def forward(
        self,
        input: torch.Tensor,
        t: torch.Tensor,
        true_output: Optional[torch.Tensor] = None,
        method: str = "dopri5",
        regularization: bool = True,
        # method: str = "rk4",
    ):
        start = input[:, 0]

        output, regularization = self._solve_dynamics(start, t, method)
        # print(f"{output.shape=}")
        # print(f"{true_output.shape=}")

        if true_output is None:
            return 0, output
        else:
            select_cycles = torch.randperm(input.shape[-2])[: self.n_cycles_loss]
            mse = self.mse(output, true_output)
            loss = self.criterion(
                output[..., select_cycles, :], true_output[..., select_cycles, :]
            )
            if regularization:
                loss = loss + self.kinetic_regularization * regularization
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
        if self.kinetic_regularization:
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
            self.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )


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
    test_length=args.init_length,
    batch_size=args.batch_size,
    perturb=args.perturb,  # .01
)
data_module.setup()

checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join("checkpoints", args.name),
    # filename="best-checkpoint",
    save_top_k=1,
    save_last=True,
    verbose=True,
    monitor="val_mse",
    mode="min",
)

early_stopping_callback = EarlyStopping(monitor="val_mse", patience=4)


class ClipLengthStepper(pl.Callback):
    def __init__(self, freq: int, max_len: int) -> None:
        super().__init__()
        self.freq: int = freq
        self.max_len: int = max_len
        self.counter = 0

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self.counter = (self.counter + 1) % self.freq
        if (
            self.counter == 0
            and trainer.train_dataloader.dataset.datasets.target_length < self.max_len
        ):
            trainer.train_dataloader.dataset.datasets.target_length += 1
            trainer.val_dataloaders[0].dataset.target_length += 1
            # trainer.test_dataloaders[0].dataset.target_length += 1
        return


if args.log:
    logger = WandbLogger(args.name)
else:
    logger = WandbLogger(args.name, mode="disabled")

wandb.config.update(args)

trainer = pl.Trainer(
    gpus=args.devices,
    logger=logger,
    enable_checkpointing=True,
    # max_epochs=N_EPOCHS,
    min_epochs=args.min_epochs,
    callbacks=[
        checkpoint_callback,
        early_stopping_callback,
        ClipLengthStepper(freq=args.incr_freq, max_len=args.max_length),
    ],
    log_every_n_steps=10,
    gradient_clip_val=args.grad_clip,
    reload_dataloaders_every_n_epochs=1,
    # clipping gradients is a hyperparameter and important to prevent divergance
    # of the gradient for recurrent neural networks
    # gradient_clip_val=0.1,
    # profiler="advanced",
    # progress_bar_refresh_rate=1,
    # strategy="dp",
    # fast_dev_run=True
)

model = SlinkyTrajectoryRegressor(
    dim_per_layer=32, n_layers=5, kinetic_regularization=args.kinetic_reg
)

trainer.fit(model=model, datamodule=data_module)

start_point = slinky_data[0]["state"][None, None, :]
# print(start_point)
# sys.exit()
model.eval()
evaluate_time = torch.from_numpy(slinky_data["time"]).to(get_model_device(model))

true_trajectory = (
    slinky_data["state"].detach().reshape(-1, num_cycles, 2, 3)[1:, :, 0, :]
)
# print(f"{true_trajectory.shape=}")

# print(f"{start_point.shape=}")
result = model.forward(start_point, evaluate_time.unsqueeze(0), method="rk4")
output = result["output"]
output = output.squeeze()
# print(f"{output['prediction'].shape}=")
# print(f"{slinky_data.shape=}")

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
