from typing import Any, Dict, Tuple
import torch
from torch import Tensor
import lightning as L

from rbc_pinn_surrogate.model.components.unet import UNet
import rbc_pinn_surrogate.callbacks.metrics_3d as metrics


class UNet3DModule(L.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        features: Tuple[int, ...] = (32, 64, 128),
        padding: Tuple[str, ...] = ("circular", "zeros", "circular"),
        in_channels: int = 4,
        out_channels: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model
        self.model = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            padding=padding,
            nl=torch.nn.GELU(),
        )

        self.loss = torch.nn.MSELoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def multi_step(
        self, x: Tensor, length: int
    ) -> Tensor:  # x has shape [B, C, D, H, W]
        xt = x
        # preds has shape [length, B, C, D, H, W]
        preds = x.new_empty(length, *x.shape)

        # autoregressive prediction
        for t in range(length):
            y_next = self.forward(xt)
            preds[t] = y_next
            xt = y_next

        # return [B, C, T, H, W]
        return preds.permute(1, 2, 0, 3, 4, 5)

    def model_step(self, input: Tensor, target: Tensor, stage: str) -> Dict[str, Tensor]:
        # get prediction
        horizon = target.shape[2]
        preds = self.multi_step(input.squeeze(dim=2), horizon)

        # compute loss per time step, then reduce for optimization
        loss_per_step = torch.stack(
            [self.loss(preds[:, :, t], target[:, :, t]) for t in range(horizon)]
        )
        loss = loss_per_step.mean()
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # time metrics
        rmse_per_step = torch.stack(
            [metrics.rmse(preds[:, :, t], target[:, :, t]) for t in range(horizon)]
        )
        rmse = rmse_per_step.mean()
        nrsse_per_step = torch.stack(
            [metrics.nrsse(preds[:, :, t], target[:, :, t]) for t in range(horizon)]
        )
        nrsse = nrsse_per_step.mean()

        self.log(f"{stage}/RMSE", rmse, on_step=False, on_epoch=True)
        self.log(f"{stage}/NRSSE", nrsse, on_step=False, on_epoch=True)

        return {
            "loss": loss,
            "loss_per_step": loss_per_step.detach().cpu(),
            "rmse_per_step": rmse_per_step.detach().cpu(),
            "nrsse_per_step": nrsse_per_step.detach().cpu(),
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
            return self.multi_step(input.squeeze(2), length)

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
