from typing import Tuple

import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape


def norm_sum_squared_scaled_error(
    preds: Tensor, target: Tensor, horizon_weight: float
) -> Tuple[Tensor, int]:
    _check_same_shape(preds, target)
    # reshape
    batch_size = preds.shape[0]
    seq_length = preds.shape[1]
    preds = preds.view(batch_size, seq_length, -1)
    target = target.view(batch_size, seq_length, -1)

    # Compute squared errors
    eps = torch.finfo(target.dtype).eps
    nsse = torch.linalg.vector_norm(
        preds - target,
        dim=[0, 2],
    ) / torch.linalg.vector_norm(
        target + eps,
        dim=[0, 2],
    )

    # Compute scaling for time axis
    scale = torch.ones_like(nsse)
    for tau in range(nsse.shape[0]):
        scale[tau] = horizon_weight**tau
    scale = scale / scale.sum()

    # Compute Sum Squared Scaled Error
    return (nsse * scale).mean()
