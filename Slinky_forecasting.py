import torch
import numpy as np
import pandas as pd

# from neural_slinky import trajectory_dataset
from pytorch_forecasting import MAE, Baseline, DeepAR, TimeSeriesDataSet
from pytorch_forecasting.metrics import SMAPE, QuantileLoss, MultiLoss

from pytorch_forecasting import TemporalFusionTransformer, DeepAR, RecurrentNetwork

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb

from neural_slinky.utils import animate_2d_slinky_trajectories

# from priority_memory import batch


pl.seed_everything(42)

def read_data():
    folder = "NeuralODE_Share2/SlinkyGroundTruth"
    true_y = np.loadtxt(folder + "/helixCoordinate_2D.txt", delimiter=",")
    true_v = np.loadtxt(folder + "/helixVelocity_2D.txt", delimiter=",")
    # true_y = torch.from_numpy(true_y).float()
    # true_v = torch.from_numpy(true_v).float()
    # true_y = torch.reshape(true_y, (true_y.size()[0], 80, 3))
    # true_v = torch.reshape(true_v, (true_v.size()[0], 80, 3))
    # print(true_v.shape)
    delta_t = 0.01
    time = np.arange(true_y.shape[0]) * delta_t
    coord_names = []
    vel_names = []
    for i in range(80):
        coord_names.append(f"x_{i}")
        coord_names.append(f"y_{i}")
        coord_names.append(f"a_{i}")
        vel_names.append(f"xv_{i}")
        vel_names.append(f"yv_{i}")
        vel_names.append(f"av_{i}")
    df = pd.DataFrame(
        np.concatenate((true_y, true_v), -1), columns=coord_names + vel_names
    )
    df["time"] = time
    df.index.name = "step"
    return df.reset_index(), coord_names + vel_names


slinky_data, feature_names = read_data()
slinky_data["slinky_id"] = 0
# slinky_data = slinky_data.copy()
training_set = TimeSeriesDataSet(
    data=slinky_data[:-101],
    time_idx="step",
    target=feature_names,
    group_ids=["slinky_id"],
    time_varying_unknown_reals=feature_names,
    max_encoder_length=1,
    max_prediction_length=10,
)
validation_set = TimeSeriesDataSet.from_dataset(
    training_set, slinky_data[-101:], predict=True, stop_randomization=True
)


# tft = TemporalFusionTransformer.from_dataset(
#     training_set,
#     # not meaningful for finding the learning rate but otherwise very important
#     learning_rate=0.03,
#     hidden_size=16,  # most important hyperparameter apart from learning rate
#     # number of attention heads. Set to up to 4 for large datasets
#     attention_head_size=1,
#     dropout=0.1,  # between 0.1 and 0.3 are good values
#     hidden_continuous_size=16,  # set to <= hidden_size
#     output_size=[1 for _ in feature_names],  # 7 quantiles by default
#     loss=MultiLoss([QuantileLoss() for _ in feature_names]),
#     # reduce learning rate if no improvement in validation loss after x epochs
#     reduce_on_plateau_patience=4,
# )
# dar = DeepAR.from_dataset(training_set)

rnn = RecurrentNetwork.from_dataset(
    training_set,
    learning_rate=0.03,
    hidden_size=16,
    rnn_layers=2,
    dropout=0.1,
    output_size=[1 for _ in feature_names],  # 7 quantiles by default
    # loss=MultiLoss([SMAPE() for _ in feature_names]),
    loss=MultiLoss([MAE() for _ in feature_names]),
)

trainer = pl.Trainer(
    gpus=[3],
    # clipping gradients is a hyperparameter and important to prevent divergance
    # of the gradient for recurrent neural networks
    gradient_clip_val=0.1,
    strategy="dp",
)

batch_size = 32  # set this between 32 to 128
train_dataloader = training_set.to_dataloader(
    train=True, batch_size=batch_size, num_workers=0
)
val_dataloader = validation_set.to_dataloader(
    train=False, batch_size=batch_size * 10, num_workers=0
)

# find optimal learning rate
# res = trainer.tuner.lr_find(
#     rnn,
#     train_dataloaders=train_dataloader,
#     val_dataloaders=val_dataloader,
#     max_lr=10.0,
#     min_lr=1e-6,
# )

# print(f"suggested learning rate: {res.suggestion()}")
# fig = res.plot(show=True, suggest=True)
# fig.show()

rnn.hparams.learning_rate = 1.04
early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
)
lr_logger = LearningRateMonitor()

trainer = pl.Trainer(
    max_epochs=100,
    # min_epochs=100,
    gpus=[2],
    weights_summary="top",
    gradient_clip_val=0.1,
    callbacks=[early_stop_callback, lr_logger],
    limit_train_batches=30,
    enable_checkpointing=True,
    # profiler="simple",
    logger=WandbLogger("slinky-forcasting"),
    # fast_dev_run=True
)

trainer.fit(
    rnn,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

full_set = TimeSeriesDataSet(
    data=slinky_data,
    time_idx="step",
    target=feature_names,
    group_ids=["slinky_id"],
    time_varying_unknown_reals=feature_names,
    max_encoder_length=1,
    max_prediction_length=len(slinky_data)-1,
)
start_point = next(iter(full_set.to_dataloader(shuffle=False, batch_size=1)))[0]
# print(start_point)
rnn.eval()
output = rnn(start_point)
# print(f"{output['prediction'].shape}=")
# print(f"{slinky_data.shape=}")

true_trajectory = slinky_data[feature_names].to_numpy().reshape(-1,2,80,3).transpose(0,2,3,1)[...,0]
print(true_trajectory.shape)
true_trajectory = torch.from_numpy(true_trajectory)
pred_trajectory = torch.cat(output["prediction"], dim=-1).squeeze_().reshape(-1,2,80,3).permute((0,2,3,1))[...,0]

evaluate_animation = animate_2d_slinky_trajectories(torch.stack([true_trajectory[1:], pred_trajectory]), 0.01)

animation_html_file = "animation_tmp.html"
evaluate_animation.write_html(animation_html_file, full_html=False)
with open(animation_html_file, "r") as f:
    evaluate_animation_html = "\n".join(f.readlines())

wandb.log({
    "evaluate_animation": wandb.Html(evaluate_animation_html, inject=False)
})