from typing import Any

from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat

from .functional.nssse import norm_sum_squared_scaled_error


class NormSumSquaredScaledError(Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    norm_sum_squared_scaled_error: Tensor
    total: Tensor

    def __init__(
        self,
        horizon_weight: float = 1,
        normalize: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.horizon_weight = horizon_weight

        self.add_state("nssse", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        self.nssse.append(
            norm_sum_squared_scaled_error(
                preds, target, horizon_weight=self.horizon_weight
            )
        )

    def compute(self) -> Tensor:
        """Compute mean squared scaeld error over state."""
        nssse = dim_zero_cat(self.nssse)
        return nssse.mean()
