from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat

from .functional.nse import normalized_sum_error


class NormalizedSumError(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("nse", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.nse.append(normalized_sum_error(preds, target))

    def compute(self):
        preds = dim_zero_cat(self.nse)
        return preds.mean()
