from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat

from .functional.nsse import normalized_sum_squared_error


class NormalizedSumSquaredError(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("nsse", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.nsse.append(normalized_sum_squared_error(preds, target))

    def compute(self):
        preds = dim_zero_cat(self.nsse)
        return preds.mean()
