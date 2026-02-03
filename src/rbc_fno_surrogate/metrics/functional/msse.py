from typing import Tuple, Union

import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape


def _mean_squared_scaled_error_update(
    preds: Tensor, target: Tensor, horizon_weight: float
) -> Tuple[Tensor, int]:
    _check_same_shape(preds, target)
    batch_size = preds.shape[0]
    seq_length = preds.shape[1]
    preds = preds.view(batch_size, seq_length, -1)
    target = target.view(batch_size, seq_length, -1)
    feature_length = preds.shape[2]

    # Compute squared errors
    squared_errors = (preds - target).pow(2).sum(dim=2)

    # Compute scaling for time axis
    scale = torch.ones_like(squared_errors)
    for tau in range(seq_length):
        scale[:, tau] = horizon_weight**tau
    scale = scale / scale.sum(dim=1, keepdim=True)

    # Compute Sum Squared Scaled Error
    squared_scaled_errors = squared_errors * scale
    sum_squared_scaled_errors = squared_scaled_errors.sum()

    return sum_squared_scaled_errors, (batch_size * feature_length)


def _mean_squared_scaled_error_compute(
    sum_squared_scaled_errors: Tensor, num_obs: Union[int, Tensor]
) -> Tensor:
    return sum_squared_scaled_errors / num_obs


def mean_squared_scaled_error(
    preds: Tensor, target: Tensor, horizon_weight: float
) -> Tensor:
    sum_squared_scaled_error, num_obs = _mean_squared_scaled_error_update(
        preds, target, horizon_weight
    )
    return _mean_squared_scaled_error_compute(sum_squared_scaled_error, num_obs)
