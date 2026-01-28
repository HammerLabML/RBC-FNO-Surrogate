from typing import Any, Dict, Callable, Optional, Tuple
import torch
from torch import Tensor
import lightning as L

from rbc_pinn_surrogate.model.components.unet import UNet


class UNet2DModule(L.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        features: Tuple[int, ...] = (32, 64, 128, 256),
        padding: Tuple[str, ...] = ("zeros", "circular"),
        in_channels: int = 3,
        out_channels: int = 3,
        denormalize: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["denormalize"])
        self.denormalize = denormalize

        # Model
        self.model = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            padding=padding,
            nl=torch.nn.GELU(),
        )
        self.model.to(device=self.device)

        self.loss = torch.nn.MSELoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def multi_step_2d(
        self, x: Tensor, length: int
    ) -> Tensor:  # x has shape [B, C, H, W]
        xt = x
        # preds has shape [length, B, C, H, W]
        preds = x.new_empty(length, *x.shape)

        # autoregressive prediction
        for t in range(length):
            y_next = self.forward(xt)
            preds[t] = y_next
            xt = y_next

        # return [B, C, T, H, W]
        return preds.permute(1, 2, 0, 3, 4)

    def model_step(self, x: Tensor, y: Tensor, stage: str) -> Dict[str, Tensor]:
        # get prediction
        length = y.shape[2]
        y_hat = self.multi_step_2d(x[:, :, -1], length)

        # loss
        loss = self.loss(y_hat, y)
        self.log(f"{stage}/loss", loss, prog_bar=True, logger=True)

        # denormalize for logging
        if self.denormalize is not None:
            y_hat = self.denormalize(y_hat)
            y = self.denormalize(y)

        return {
            "loss": loss,
            "ground_truth": y,
            "prediction": y_hat,
        }

    def training_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> Dict[str, Tensor]:
        x, y = batch
        return self.model_step(x, y, stage="train")

    def validation_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> Dict[str, Tensor]:
        x, y = batch
        return self.model_step(x, y, stage="val")

    def test_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> Dict[str, Tensor]:
        x, y = batch
        return self.model_step(x, y, stage="test")

    def predict(self, input: Tensor, length: int) -> Tensor:
        with torch.inference_mode():
            return self.multi_step_2d(input[:, :, -1].squeeze(2), length)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.model.parameters(),
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
