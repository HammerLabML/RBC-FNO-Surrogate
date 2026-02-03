import logging

import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from rbc_fno_surrogate.utils.vis_2d import sequence2video


class Examples2DCallback(Callback):
    def __init__(self, freq: int = 5):
        self.freq = freq

        logger = logging.getLogger("matplotlib.animation")
        logger.setLevel(logging.ERROR)

    # Training callbacks
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        epoch = trainer.current_epoch
        if batch_idx == 0 and epoch % self.freq == 0 and epoch > 0:
            with torch.no_grad():
                self.log_output(outputs, 0, "train", trainer.logger)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        epoch = trainer.current_epoch
        if batch_idx == 0 and epoch % self.freq == 0 and epoch > 0:
            with torch.no_grad():
                self.log_output(outputs, 0, "val", trainer.logger)

    # Testing callbacks
    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        self.log_output(outputs, 0, "test", trainer.logger)

    def log_output(self, outputs: dict, idx: int, stage: str, logger: Logger):
        # unpack sequence
        y = outputs["ground_truth"][idx].detach().cpu().numpy()
        y_hat = outputs["prediction"][idx].detach().cpu().numpy()

        # generate videos with GT / Prediction / Difference side-by-side
        videos = []
        captions = []
        for field in ["T", "U", "W"]:
            videos.append(sequence2video(y, y_hat, field))
            captions.append(field)

        # log to wandb
        logger.log_video(
            f"{stage}/examples",
            videos,
            caption=captions,
            format=["mp4" for i in range(len(videos))],
        )
