from typing import Dict
import torch
from torch import Tensor
import lightning as L
import neuralop as no

import rbc_fno_surrogate.callbacks.metrics_3d as metrics


class FNO3DModule(L.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        n_modes_xy: int = 16,
        n_modes_z: int = 16,
        hidden_channels: int = 16,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: int = 16,
        n_layers: int = 2,
        loss: str = "h1",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["denormalize"])

        # Model parameters
        self.model = no.models.TFNO3d(
            n_modes_depth=n_modes_xy,
            n_modes_height=n_modes_z,
            n_modes_width=n_modes_xy,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channels=channels,
            projection_channels=channels,
            n_layers=n_layers,
        )

        # Loss Function
        if loss == "l2":
            self.loss = torch.nn.MSELoss()
        elif loss == "h1":
            self.loss = no.H1Loss(d=3)
        else:
            raise ValueError(f"Unknown loss function: {loss}")

    def forward(self, x):
        return self.model(x)

    def predict(self, input: Tensor, length) -> Tensor:
        with torch.no_grad():
            return self.multi_step(input.squeeze(dim=2), length)

    def multi_step(self, x: Tensor, length: int) -> Tensor:
        # x has shape [B, C, D, H, W]
        xt = x
        # preds has shape [length, B, C, D, H, W]
        preds = x.new_empty(length, *x.shape)

        # autoregressive prediction
        for t in range(length):
            y_next = self.forward(xt)
            preds[t] = y_next
            xt = y_next

        # return [B, C, T, D, H, W]
        return preds.permute(1, 2, 0, 3, 4, 5)

    def model_step(
        self, input: Tensor, target: Tensor, stage: str
    ) -> Dict[str, Tensor]:
        # autoregressive model steps
        preds = self.multi_step(input.squeeze(dim=2), target.shape[2])
        horizon = target.shape[2]

        # compute loss per time step, then reduce for optimization
        loss_per_step = torch.stack(
            [self.loss(preds[:, :, t], target[:, :, t]) for t in range(horizon)]
        )
        loss = loss_per_step.mean()
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # compute sequence metrics
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

    def training_step(self, batch, batch_idx):
        x, y = batch
        return self.model_step(x, y, stage="train")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        return self.model_step(x, y, stage="val")

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = batch
            return self.model_step(x, y, stage="test")

    def configure_optimizers(self):
        wd = getattr(self.hparams, "weight_decay", 0.0)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=wd,
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

    def load_state_dict(self, state_dict, strict: bool = True):
        # Remove metadata from neuralop library TODO check if useful
        state_dict.pop("_metadata", None)
        return super().load_state_dict(state_dict, strict)
