from typing import Any, Dict, Tuple, Callable, List, Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import mse_loss
from lightning.pytorch import LightningModule

from rbc_pinn_surrogate.model.components import Autoencoder3Dv2, Autoencoder3D


class Autoencoder3DModule(LightningModule):
    def __init__(
        self,
        version: Literal["v1", "v2"],
        input_size: Tuple[int, int, int, int],
        latent_channels: int,
        channels: List[int],
        pooling: List[bool],
        kernel_size: int,
        latent_kernel_size: int,
        drop_rate: float,
        batch_norm: bool,
        lr: float,
        inv_transform: Callable = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["inv_transform"])
        self.example_input_array = torch.zeros(1, *input_size)

        # model
        if version == "v1":
            self.autoencoder = Autoencoder3D(
                latent_channels=latent_channels,
                input_size=input_size,
                channels=channels,
                pooling=pooling,
                kernel_size=kernel_size,
                latent_kernel_size=latent_kernel_size,
                drop_rate=drop_rate,
                batch_norm=batch_norm,
                activation=nn.GELU,
            )
        else:
            self.autoencoder = Autoencoder3Dv2(
                rb_dims=(48, 48, 32),
                encoder_channels=channels,
                latent_channels=latent_channels,
                v_kernel_size=kernel_size,
                h_kernel_size=kernel_size,
                latent_v_kernel_size=latent_kernel_size,
                latent_h_kernel_size=latent_kernel_size,
                drop_rate=drop_rate,
                pool_layers=pooling,
                nonlinearity=nn.GELU,
            )

        # Denormalize
        self.denormalize = inv_transform

    def forward(self, x: Tensor) -> Tensor:
        return self.autoencoder(x)

    def model_step(
        self, x: Tensor, stage: str, examples: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # check input dimensions
        assert x.shape[2] == 1, (
            f"Expect sequence length of 1 for autoencoder training, got {x.shape}"
        )

        # model forward
        x_hat = self.forward(x.squeeze(dim=2))
        x_hat = x_hat.unsqueeze(dim=2)

        # compute loss
        loss = mse_loss(x_hat, x)
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/RMSE", torch.sqrt(loss), on_step=False, on_epoch=True)

        # return first batch sample
        if examples:
            return {
                "loss": loss,
                "ground_truth": self.denormalize(x)[0],
                "prediction": self.denormalize(x_hat)[0],
            }

        return {
            "loss": loss,
        }

    def training_step(self, batch, batch_idx):
        x, _ = batch
        return self.model_step(x, stage="train")

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        return self.model_step(x, stage="val", examples=True)

    def test_step(self, batch, batch_idx):
        x, _ = batch
        return self.model_step(x, stage="test", examples=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(
            params=self.autoencoder.parameters(),
            lr=self.hparams.lr,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }
