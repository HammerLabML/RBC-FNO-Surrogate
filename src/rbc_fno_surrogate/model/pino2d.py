from typing import Dict
import torch
from torch import nn
from torch import Tensor
import lightning as L
import neuralop as no

from rbc_pinn_surrogate.callbacks import OperatorFinetuneCallback


class PINO2DModule(L.LightningModule):
    def __init__(
        self,
        pino_loss: nn.Module,
        pino_weight: float = 1.0,
        operator_weight: float = 1.0,
        data_loss: nn.Module = None,
        data_weight: float = 1.0,
        lr: float = 1e-3,
        n_modes_width: int = 16,
        n_modes_height: int = 16,
        n_modes_depth: int = 16,
        hidden_channels: int = 16,
        in_channels: int = 3,
        out_channels: int = 3,
        lifting_channels: int = 16,
        projection_channels: int = 16,
        n_layers: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pino_loss", "operator"])

        # Model parameters
        self.model = no.models.TFNO3d(
            n_modes_width=n_modes_width,
            n_modes_height=n_modes_height,
            n_modes_depth=n_modes_depth,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
        )

        # Loss Function
        if data_loss is None:
            self.data_loss = no.H1Loss(d=3)
        else:
            self.data_loss = data_loss
        self.eqn_loss = pino_loss

        # operator for finetuning
        self.finetuning = False

    def forward(self, x):
        return self.model(x)

    def set_finetuning_phase(self):
        self.finetuning = True
        print("Finetuning phase activated. Model will now use operator finetuning.")

    def model_step(self, x: Tensor, y: Tensor, stage: str) -> Dict[str, Tensor]:
        prog_bar_log = stage == "train"

        # Forward pass
        pred = self(x)

        # data loss
        data_loss = self.data_loss(pred, y)
        self.log(f"{stage}/data_loss", data_loss, prog_bar=prog_bar_log, logger=True)

        # pino loss
        pino_loss = self.eqn_loss(pred)
        pino_baseline = self.eqn_loss(y)
        self.log(f"{stage}/pino_loss", pino_loss, prog_bar=prog_bar_log, logger=True)
        self.log(f"{stage}/pino_baseline", pino_baseline, logger=True)

        # operator loss
        if self.finetuning:
            reference = self.get_reference_model()
            with torch.no_grad():
                ref_pred = reference(x)
            op_loss = self.data_loss(ref_pred, y)
            self.log(f"{stage}/op_loss", op_loss, prog_bar=prog_bar_log, logger=True)

            # operator + pino loss
            loss = (self.hparams.operator_weight * op_loss) + (
                self.hparams.pino_weight * pino_loss
            )
        else:
            # data + pino loss
            loss = (self.hparams.data_weight * data_loss) + (
                self.hparams.pino_weight * pino_loss
            )

        self.log(f"{stage}/loss", loss, prog_bar=prog_bar_log, logger=True)
        return {
            "loss": loss,
            "x": x,
            "y": y,
            "y_hat": pred,
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        return self.model_step(x, y, stage="train")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        return self.model_step(x, y, stage="val")

    def test_step(self, batch, batch_idx):
        input, target = batch
        input_length = input.shape[2]
        target_length = target.shape[2]

        # autoregressive model steps
        length = 0
        x = input
        pred = torch.empty(
            x.shape[0],
            x.shape[1],
            target_length,
            *x.shape[3:],
            device=x.device,
            dtype=x.dtype,
        )

        while length < target_length:
            y = self.forward(x)
            step = min(input_length, target_length - length)
            pred[:, :, length : length + step] = y[:, :, :step]
            x = y
            length += step

        # Compute and log metrics
        loss = self.data_loss(pred, target)
        self.log("test/data_loss", loss, logger=True)

        if self.eqn_loss is not None:
            pino_loss = self.eqn_loss(pred)
            pino_baseline = self.eqn_loss(target)
            self.log("test/pino_loss", pino_loss, logger=True)
            self.log("test/pino_baseline", pino_baseline, logger=True)

        return {
            "loss": loss,
            "y": target,
            "y_hat": pred,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.lr,
        )
        return {"optimizer": optimizer}

    def load_state_dict(self, state_dict, strict: bool = True):
        # Remove metadat from neuralop library TODO check if useful
        state_dict.pop("_metadata", None)
        return super().load_state_dict(state_dict, strict)

    def get_reference_model(self):
        ref_cb = next(
            cb
            for cb in self.trainer.callbacks
            if isinstance(cb, OperatorFinetuneCallback)
        )
        if ref_cb.reference_model is not None:
            return ref_cb.reference_model
        else:
            raise ValueError("Reference model is not set in OperatorFinetuneCallback.")
