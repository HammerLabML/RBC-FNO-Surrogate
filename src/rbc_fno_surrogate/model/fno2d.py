import math
from typing import Any, Dict, Callable, Optional, Literal, Tuple
import torch
from torch import Tensor
import lightning as L
import neuralop as no


class FNO2DModule(L.LightningModule):
    def __init__(
        self,
        dim: Literal["2d", "3d"] = "3d",  # autoregressive 2d or 3d
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        n_modes_space: int = 16,
        n_modes_time: int = 16,
        hidden_channels: int = 16,
        in_channels: int = 3,
        out_channels: int = 3,
        lifting_channels: int = 16,
        projection_channels: int = 16,
        n_layers: int = 2,
        loss: Literal["L2", "H1", "MSE"] = "H1",
        denormalize: Optional[Callable[[Tensor], Tensor]] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["denormalize"])
        self.denormalize = denormalize

        # Model parameters
        self.dim = dim
        if dim == "3d":
            self.model = no.models.TFNO3d(
                n_modes_width=n_modes_space,
                n_modes_height=n_modes_space,
                n_modes_depth=n_modes_time,
                hidden_channels=hidden_channels,
                in_channels=in_channels,
                out_channels=out_channels,
                lifting_channels=lifting_channels,
                projection_channels=projection_channels,
                n_layers=n_layers,
            )
        elif dim == "2d":
            self.model = no.models.TFNO2d(
                n_modes_width=n_modes_space,
                n_modes_height=n_modes_space,
                hidden_channels=hidden_channels,
                in_channels=in_channels,
                out_channels=out_channels,
                lifting_channels=lifting_channels,
                projection_channels=projection_channels,
                n_layers=n_layers,
            )
        else:
            raise ValueError(f"Unknown FNO2DModule type: {dim}")

        # Loss Function
        if loss == "L2":
            self.loss = no.LpLoss(d=2, p=2)
        elif loss == "H1":
            self.loss = no.H1Loss(d=2)
        elif loss == "MSE":
            self.loss = torch.nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss function: {loss}")

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def multi_step_2d(self, x: Tensor, length: int) -> Tensor:
        # x has shape [B, C, H, W]
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

    def multi_step_3d(self, x: Tensor, length: int) -> Tensor:
        input_length = x.shape[2]
        # x has shape [B, C, T, H, W]
        preds = x.new_empty(x.shape[0], x.shape[1], length, *x.shape[3:])

        n_chunks = math.ceil(length / input_length)
        xt = x
        for k in range(n_chunks):
            yt = self.forward(xt)

            # important for last step
            start = k * input_length
            end = min(start + input_length, length)
            step = end - start

            preds[:, :, start:end] = yt[:, :, :step]
            xt = yt

        return preds

    def model_step(self, x: Tensor, y: Tensor, stage: str) -> Dict[str, Tensor]:
        # get prediction
        length = y.shape[2]
        if self.dim == "2d":
            y_hat = self.multi_step_2d(x[:, :, -1], length)
        else:
            y_hat = self.multi_step_3d(x, length)

        # compute loss per time step (H1 loss is spatial)
        loss = torch.stack(
            [self.loss(y_hat[:, :, t], y[:, :, t]) for t in range(length)]
        ).mean()
        self.log(f"{stage}/loss", loss, prog_bar=True)

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
            if self.dim == "2d":
                y_hat = self.multi_step_2d(input[:, :, -1], length)
            else:
                y_hat = self.multi_step_3d(input, length)

            return y_hat

    def configure_optimizers(self) -> Dict[str, Any]:
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

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> Any:
        # Remove metadata inserted by neuralop (if present)
        state_dict.pop("_metadata", None)
        return super().load_state_dict(state_dict, strict)
