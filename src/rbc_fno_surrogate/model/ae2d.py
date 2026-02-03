from typing import Any, Dict, List, Tuple, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import mse_loss

from lightning.pytorch import LightningModule

from rbc_fno_surrogate.model.components import Autoencoder2D


class Autoencoder2DModule(LightningModule):
    def __init__(
        self,
        latent_dimension: int,
        input_channel: int,
        channels: List[int],
        kernel_size: int,
        lr: float,
        denormalize: Callable = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["denormalize"])

        # model
        self.activation = nn.GELU
        self.autoencoder = Autoencoder2D(
            latent_dimension, input_channel, channels, kernel_size, self.activation
        )

        # Denormalize
        self.denormalize = denormalize

        # Debugging
        self.example_input_array = torch.zeros(1, input_channel, 64, 96)

    def forward(self, x: Tensor) -> Tensor:
        # forward
        x_hat, _ = self.autoencoder(x)
        return x_hat

    def model_step(self, input: Tensor, stage: str) -> Tuple[Tensor, Tensor, Tensor]:
        # model forward (make sure only one time step)
        x = input[:, :, -1]
        x_hat = self.forward(x)

        # dim expand for callbacks and denormalization
        x = x.unsqueeze(dim=2)
        x_hat = x_hat.unsqueeze(dim=2)

        # compute loss
        loss = mse_loss(x_hat, x)
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Apply inverse transform
        if self.denormalize is not None:
            with torch.no_grad():
                x = self.denormalize(x.detach())
                x_hat = self.denormalize(x_hat.detach())

        return {
            "loss": loss,
            "prediction": x_hat,
            "ground_truth": x,
        }

    def training_step(self, batch, batch_idx):
        x, _ = batch
        return self.model_step(x, stage="train")

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        return self.model_step(x, stage="val")

    def test_step(self, batch, batch_idx):
        x, _ = batch
        return self.model_step(x, stage="test")

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(
            params=self.trainer.model.parameters(), lr=self.hparams.lr
        )
        return {"optimizer": optimizer}
