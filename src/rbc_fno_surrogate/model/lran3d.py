from itertools import chain
from typing import Any, Dict

import lightning.pytorch as pl
import torch.nn as nn
from torch.nn.functional import mse_loss
import torch
from torch import Tensor

from rbc_pinn_surrogate.model.components import (
    Autoencoder3D,
    KoopmanOperator,
)
import rbc_pinn_surrogate.callbacks.metrics_3d as metrics


class LRAN3DModule(pl.LightningModule):
    def __init__(
        self,
        # LRAN params
        input_size: int,
        ae_ckpt: str,
        latent_dimension: int,
        lambda_id: float,
        lambda_fwd: float,
        lambda_hid: float,
        # Optimizer params
        lr_operator: float,
        lr_autoencoder: float,
        train_autoencoder: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["denormalize"])

        # Debugging
        self.example_input_array = torch.zeros(1, *input_size)

        # Autoencoder
        ckpt = torch.load(ae_ckpt, map_location=self.device)
        self.autoencoder = Autoencoder3D.from_checkpoint(ckpt)
        self.autoencoder.requires_grad_(train_autoencoder)
        if not train_autoencoder:
            self.autoencoder.eval()

        # Get latent spatial dimensions
        with torch.no_grad():
            dummy = self.autoencoder.encode(self.example_input_array)
            latent_shape = dummy.shape[1:]
            latent_flat = dummy[0].numel()

        # latent layers
        self.encoder_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_flat, latent_dimension),
            nn.GELU(),
        )
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dimension, latent_flat),
            nn.GELU(),
            nn.Unflatten(1, latent_shape),
        )

        # Time dynamics model
        self.operator = KoopmanOperator(latent_dimension)

        # Loss
        self.loss = mse_loss

    def forward(self, g: Tensor) -> Tensor:
        return self.operator(g)

    def encode(self, x: Tensor) -> Tensor:
        z = self.autoencoder.encode(x)
        return self.encoder_linear(z)

    def decode(self, g: Tensor) -> Tensor:
        z = self.decoder_linear(g)
        return self.autoencoder.decode(z)

    def predict(self, input: Tensor, length) -> Tensor:
        self.eval()
        with torch.no_grad():
            # encode input to latent space
            g = self.encode(input.squeeze(dim=2))

            # autoregressive model steps
            pred = []
            for _ in range(length):
                g = self.forward(g)
                x_hat = self.decode(g)
                pred.append(x_hat)

            return torch.stack(pred, dim=2)

    def model_step(
        self, input: Tensor, target: Tensor, stage: str
    ) -> Dict[str, Tensor]:
        x = input.squeeze(dim=2)
        y = target
        seq_length = target.shape[2]
        g, g_hat, y_hat = [], [], []

        # Get ground truth for observables
        with torch.no_grad():
            g0 = self.encode(x).detach()
            g = [self.encode(y[:, :, t]).detach() for t in range(seq_length)]

        # Reconstruction
        x_hat = self.decode(g0)

        # Predict sequence in latent space
        g_hat, y_hat = [], []
        g_prev = g0
        for _ in range(seq_length):
            g_next = self.operator(g_prev)
            y_hat.append(self.decode(g_next))
            g_hat.append(g_next)
            g_prev = g_next

        # To tensor
        g = torch.stack(g, dim=1)
        g_hat = torch.stack(g_hat, dim=1)
        y_hat = torch.stack(y_hat, dim=2)

        # Loss
        reconstruction = self.loss(x_hat, x)
        forward = self.loss(y_hat, y)
        hidden = self.loss(g_hat, g)
        loss = (
            self.hparams.lambda_id * reconstruction
            + self.hparams.lambda_fwd * forward
            + self.hparams.lambda_hid * hidden
        )

        # compute Metrics for each t
        rmse = (
            torch.stack(
                [metrics.rmse(y_hat[:, :, t], y[:, :, t]) for t in range(seq_length)]
            )
            .detach()
            .cpu()
        )
        nrsse = (
            torch.stack(
                [metrics.nrsse(y_hat[:, :, t], y[:, :, t]) for t in range(seq_length)]
            )
            .detach()
            .cpu()
        )

        # Log
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(
            {
                f"{stage}/RMSE": rmse.mean(),
                f"{stage}/NRSSE": nrsse.mean(),
                f"{stage}/loss/reconstruction": reconstruction,
                f"{stage}/loss/forward": forward,
                f"{stage}/loss/hidden": hidden,
            },
            on_step=False,
            on_epoch=True,
        )

        return {
            "loss": loss,
            "rmse": rmse,
            "nrsse": nrsse,
        }

    def training_step(self, batch: Tensor, batch_idx: int) -> Dict[str, Tensor]:
        x, y = batch
        return self.model_step(x, y, stage="train")

    def validation_step(self, batch: Tensor, batch_idx: int) -> Dict[str, Tensor]:
        x, y = batch
        return self.model_step(x, y, stage="val")

    def test_step(self, batch: Tensor, batch_idx: int) -> Dict[str, Tensor]:
        x, y = batch
        return self.model_step(x, y, stage="test")

    def configure_optimizers(self) -> Dict[str, Any]:
        autoencoder_params = [
            self.encoder_linear.parameters(),
            self.decoder_linear.parameters(),
        ]
        if self.hparams.train_autoencoder:
            autoencoder_params.append(self.autoencoder.parameters())

        optimizer = torch.optim.Adam(
            params=[
                {
                    "params": chain(*autoencoder_params),
                    "lr": self.hparams.lr_autoencoder,
                },
                # operator weights
                {
                    "params": self.operator.parameters(),
                    "lr": self.hparams.lr_operator,
                },
            ],
        )
        return {"optimizer": optimizer}

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.hparams.train_autoencoder:
            self.autoencoder.eval()
        return self
