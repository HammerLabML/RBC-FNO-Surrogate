import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
import torch
from torch import Tensor
import wandb

from rbc_fno_surrogate.callbacks import metrics_2d as metrics
from rbc_fno_surrogate.data.dataset import Field2D


class SequenceMetricsCallback(Callback):
    def __init__(
        self,
        key_groundtruth: str,
        key_prediction: str,
        domain: tuple[float, float] = (2 * np.pi, 2.0),
    ):
        self.key_groundtruth = key_groundtruth
        self.key_prediction = key_prediction
        self.domain = domain

        self.data = []

    # Testing callbacks
    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        pred = outputs[self.key_prediction].cpu()
        gt = outputs[self.key_groundtruth].cpu()

        # Update each metric
        self.update(pred, gt, batch_idx)

    def on_test_end(self, trainer, pl_module) -> None:
        df = pd.DataFrame(self.data)
        im = self.plot_metrics(df, "nrsse")

        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.log_table("test/Table-Metrics", dataframe=df)
            trainer.logger.log_image("test/Plot-NRSSE", [im])

        # clear test data
        self.data = []

    def update(self, pred: Tensor, target: Tensor, batch_idx: int):
        for idx in range(target.shape[0]):
            for step in range(target.shape[2]):
                # compute metrics
                nrsse = metrics.nrsse(pred[idx, :, step], target[idx, :, step]).item()
                rmse = metrics.rmse(pred[idx, :, step], target[idx, :, step]).item()

                # heat flux
                T_ref = target[:, Field2D.T].mean()
                q_pred = metrics.compute_q(pred[idx, :, step], T_ref)
                q_target = metrics.compute_q(target[idx, :, step], T_ref)

                # compute divergence
                div_pred, _, _ = metrics.compute_divergence(
                    pred[idx, :, step], self.domain
                )
                div_rms_pred = torch.sqrt(torch.mean(div_pred**2)).item()
                div_target, _, _ = metrics.compute_divergence(
                    target[idx, :, step], self.domain
                )
                div_rms_target = torch.sqrt(torch.mean(div_target**2)).item()

                self.data.append(
                    {
                        "idx": batch_idx * target.shape[0] + idx,
                        "batch_idx": batch_idx,
                        "sample_idx": idx,
                        "step": step,
                        "rmse": rmse,
                        "nrsse": nrsse,
                        "mean_q_pred": q_pred,
                        "mean_q_target": q_target,
                        "div_pred": div_rms_pred,
                        "div_target": div_rms_target,
                    }
                )

    def plot_metrics(self, df: pd.DataFrame, metric: str):
        fig = plt.figure()
        sns.set_theme()
        ax = sns.lineplot(data=df, x="step", y=metric)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.set_xlabel("Time Step")
        ax.set_ylim(bottom=0, top=0.5)

        # save as image
        im = wandb.Image(fig, caption=metric)
        plt.close(fig)
        return im
