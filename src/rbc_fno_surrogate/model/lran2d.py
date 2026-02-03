from typing import Any, Dict, Callable, List

import lightning.pytorch as pl
import torch.nn as nn
import torch
from torch import Tensor

from rbc_fno_surrogate.model.components import Autoencoder2D, KoopmanOperator


class LRAN2DModule(pl.LightningModule):
    def __init__(
        self,
        # Autoencoder params
        latent_dimension: int,
        input_channel: int,
        channels: List[int],
        kernel_size: int,
        ae_ckpt: str,
        # Loss params
        lambda_id: float,
        lambda_fwd: float,
        lambda_hid: float,
        # Optimizer params
        lr_operator: float,
        lr_autoencoder: float,
        # Misc
        denormalize: Callable = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["denormalize"])

        # Model
        activation = nn.GELU
        self.autoencoder = Autoencoder2D(
            latent_dimension, input_channel, channels, kernel_size, activation
        )
        if ae_ckpt is not None:
            ckpt = torch.load(ae_ckpt, map_location="cpu", weights_only=True)
            self.autoencoder.load_weights(ckpt, freeze=True)
            print(f"Loaded autoencoder weights from {ae_ckpt}")
        self.operator = KoopmanOperator(latent_dimension)

        # Loss
        self.loss = torch.nn.functional.mse_loss

        # Denormalize
        self.denormalize = denormalize

        # Debugging
        self.example_input_array = torch.zeros(1, input_channel, 64, 96)

    def forward(self, x: Tensor) -> Tensor:
        g = self.encode(x)
        g_next = self.evolve(g)
        x_next = self.decode(g_next)
        return x_next

    def evolve(self, g: Tensor) -> Tensor:
        return self.operator(g)

    def encode(self, x: Tensor) -> Tensor:
        return self.autoencoder.encode(x)

    def decode(self, g: Tensor) -> Tensor:
        return self.autoencoder.decode(g)

    def predict(self, input: Tensor, length) -> Tensor:
        self.eval()
        with torch.no_grad():
            # encode input to latent space
            g = self.encode(input[:, :, -1].squeeze(dim=2))

            # autoregressive model steps
            pred = []
            for _ in range(length):
                g = self.evolve(g)
                x_hat = self.decode(g)
                pred.append(x_hat)

            return torch.stack(pred, dim=2)

    def model_step(self, x: Tensor, y: Tensor, stage: str) -> Dict[str, Tensor]:
        # x.shape [B, C, T, H, W], y.shape [B, C, T, H, W]
        x = x[:, :, -1]  # use only last input frame
        horizon = y.shape[2]
        g, g_hat, y_hat = [], [], []

        # Get ground truth for observables
        with torch.no_grad():
            g0 = self.encode(x)
            for tau in range(0, horizon):
                g.append(self.encode(y[:, :, tau]).detach())

        # reconstruction and first step
        x_hat = self.decode(g0)
        g_hat.append(self.evolve(g0))

        # Predict target sequence
        for tau in range(0, horizon - 1):
            y_hat.append(self.decode(g_hat[tau]))
            g_hat.append(self.evolve(g_hat[tau]))
        y_hat.append(self.decode(g_hat[-1]))

        # To tensors
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

        # Log
        self.log_dict(
            {
                f"{stage}/loss": loss,
                f"{stage}/loss/reconstruction": reconstruction,
                f"{stage}/loss/forward": forward,
                f"{stage}/loss/hidden": hidden,
            }
        )

        # Apply inverse transform
        if self.denormalize is not None:
            with torch.no_grad():
                y = self.denormalize(y.detach())
                y_hat = self.denormalize(y_hat.detach())

        return {
            "loss": loss,
            "ground_truth": y,
            "prediction": y_hat,
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
        optimizer = torch.optim.Adam(
            params=[
                {
                    "params": self.autoencoder.parameters(),
                    "lr": self.hparams.lr_autoencoder,
                },
                {
                    "params": self.operator.parameters(),
                    "lr": self.hparams.lr_operator,
                },
            ],
        )
        return {"optimizer": optimizer}
