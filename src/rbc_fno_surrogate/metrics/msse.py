from typing import Any

import torch
from torch import Tensor, tensor
from torchmetrics.metric import Metric

from .functional.msse import (
    _mean_squared_scaled_error_compute,
    _mean_squared_scaled_error_update,
)


class MeanSquaredScaledError(Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    sum_squared_scaled_error: Tensor
    total: Tensor

    def __init__(
        self,
        horizon_weight: float = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.horizon_weight = horizon_weight

        self.add_state(
            "sum_squared_scaled_error", default=torch.zeros(1), dist_reduce_fx="sum"
        )
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        sum_squared_scaled_error, num_obs = _mean_squared_scaled_error_update(
            preds,
            target,
            horizon_weight=self.horizon_weight,
        )

        self.sum_squared_scaled_error += sum_squared_scaled_error
        self.total += num_obs

    def compute(self) -> Tensor:
        """Compute mean squared scaeld error over state."""
        return _mean_squared_scaled_error_compute(
            self.sum_squared_scaled_error, self.total
        )
