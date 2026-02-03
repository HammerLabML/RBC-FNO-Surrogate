from lightning.pytorch.callbacks import Callback
import torch
import gc


class ClearMemoryCallback(Callback):
    # Testing callbacks
    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        # Get tensors
        pred = outputs["prediction"].cpu()
        gt = outputs["ground_truth"].cpu()

        # Free memory
        del pred, gt
        outputs.clear()
        torch.cuda.empty_cache()
        gc.collect()
