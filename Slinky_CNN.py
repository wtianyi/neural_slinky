import random
from multiprocessing.sharedctypes import Value
import sys
from typing import List, Optional

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


def read_data():
    folder = "NeuralODE_Share2/SlinkyGroundTruth"
    true_y = np.loadtxt(folder + "/helixCoordinate_2D.txt", delimiter=",")
    true_v = np.loadtxt(folder + "/helixVelocity_2D.txt", delimiter=",")
    true_y = torch.from_numpy(true_y).float()
    true_v = torch.from_numpy(true_v).float()
    true_y = torch.reshape(true_y, (true_y.size()[0], 80, 3))
    true_v = torch.reshape(true_v, (true_v.size()[0], 80, 3))
    # print(true_v.shape)
    delta_t = 0.01
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
    return batch.Batch(
        state=torch.cat((true_y, true_v), axis=-1).transpose(-2, -1).contiguous(),
        time=time,
    )


slinky_data = read_data()


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
        return self.test_dataset.to_dataloader(1, "random")

    def test_dataloader(self):
        return self.test_dataset.to_dataloader(1, "random")

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
                        ),
                        torch.nn.GELU(),
                        # torch.nn.BatchNorm1d(out_size),
                    )
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
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model = SlinkyCNN(dim_list=dim_list, residual=residual)
        # self.criterion = nn.MSELoss()
        self.teacher_forcing: float = teacher_forcing
        self.register_buffer("feat_weights", torch.tensor([1e2,1e2,1,1e2,1e2,1]))

    def criterion(self, x, y):
        return torch.mean((torch.max(torch.abs(x - y), dim=-1)[0] * self.feat_weights))

    def forward_step(self, input: torch.Tensor):
        try:
            n_batch, n_time, n_cycle, n_feat = input.shape
        except:
            raise ValueError(
                f"The shape of the input is expected to be (n_batch, n_time, n_cycle, n_feat). Got {input.shape=}"
            )
        output: torch.Tensor = self.model(input.flatten(start_dim=0, end_dim=1))
        output = output.reshape(n_batch, n_time, n_cycle, n_feat)
        return output

    def forward_autoregressive(self, start: torch.Tensor, n_steps: int):
        try:
            n_batch, n_cycle, n_feat = start.shape
        except:
            raise ValueError(
                f"The shape of the input is expected to be (n_batch, n_cycle, n_feat). Got {start.shape=}"
            )
        output = []
        for _ in range(n_steps):
            start = self.model(start)
            output.append(start)
        return torch.stack(output, dim=1)

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
        return torch.optim.AdamW(self.parameters(), lr=0.0001)


# train_dataset = trajectory_dataset.TrajectoryDataset([slinky_data[:training_cutoff]], input_length=1, target_length=1)
# train_dataloader = train_dataset.to_dataloader(batch_size=20, sampler="stratified")

# val_dataset = trajectory_dataset.TrajectoryDataset([slinky_data[training_cutoff:]], input_length=1, target_length=1)
# val_dataloader = val_dataset.to_dataloader(batch_size=20, sampler="stratified")

pl.seed_everything(42)
training_cutoff = 700
# training_cutoff = 65

N_EPOCHS = 50
BATCH_SIZE = 8
data_module = SlinkyDataModule(
    [slinky_data[:training_cutoff]],
    # [slinky_data[training_cutoff:]],
    [slinky_data[:training_cutoff]],
    input_length=1,
    target_length=10,
    batch_size=BATCH_SIZE,
    perturb=0.001
)
data_module.setup()

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min",
)

early_stopping_callback = EarlyStopping(monitor="val_loss", patience=4)
# for input, output in data_module.train_dataloader():
#     print(input.keys())
#     print(output.keys())
#     print(input["state"].shape)
#     print(output["state"].shape)
#     print(input["time"].shape)
#     print(output["time"].shape)
#     break
# sys.exit()

logger = WandbLogger()

trainer = pl.Trainer(
    gpus=[2],
    logger=logger,
    checkpoint_callback=checkpoint_callback,
    # max_epochs=N_EPOCHS,
    min_epochs=100,
    callbacks=[early_stopping_callback],
    # clipping gradients is a hyperparameter and important to prevent divergance
    # of the gradient for recurrent neural networks
    # gradient_clip_val=0.1,
    progress_bar_refresh_rate=30
    # strategy="dp",
)

model = SlinkyTrajectoryRegressor(
    dim_list=[6, 64, 64, 64, 64, 6], residual=True, teacher_forcing=0.8
)

trainer.fit(model=model, datamodule=data_module)

start_point = slinky_data[0]["state"].unsqueeze(0)
# print(start_point)
# sys.exit()
model.eval()
output = model.forward_autoregressive(start_point, len(slinky_data) - 1).squeeze()
# print(f"{output['prediction'].shape}=")
# print(f"{slinky_data.shape=}")

true_trajectory = (
    slinky_data["state"].detach().transpose(-2, -1).reshape(-1, 80, 2, 3)[..., 0, :]
)
print(f"{true_trajectory.shape=}")
pred_trajectory = output.transpose(-2, -1).reshape(-1, 80, 2, 3)[..., 0, :]

evaluate_animation = animate_2d_slinky_trajectories(
    torch.stack([pred_trajectory, true_trajectory[1:]]), 0.01
)

animation_html_file = "animation_tmp.html"
evaluate_animation.write_html(animation_html_file, full_html=False)
with open(animation_html_file, "r") as f:
    evaluate_animation_html = "\n".join(f.readlines())

torch.save(pred_trajectory, "pred_trajectory.pth")
torch.save(true_trajectory[1:], "true_trajectory.pth")

wandb.log({"evaluate_animation": wandb.Html(evaluate_animation_html, inject=False)})
# model = nn.LSTM
# # find optimal learning rate
# res = trainer.tuner.lr_find(
#     model,
#     train_dataloaders=train_dataloader,
#     val_dataloaders=val_dataloader,
#     max_lr=10.0,
#     min_lr=1e-6,
# )
