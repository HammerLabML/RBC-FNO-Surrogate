from typing import Any, Dict

import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric


class NusseltNumber(Metric):
    def __init__(self, rbc_parameters: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.add_state("input", default=torch.tensor(0), dist_reduce_fx="mean")

        # Get rbc parameters from dict
        ra = rbc_parameters.get("ra", 1000000)
        pr = rbc_parameters.get("pr", 0.7)
        bcT = rbc_parameters.get("bcT", [2, 1])
        domain = rbc_parameters.get("domain", [[-1, 1]])
        H = domain[0][1] - domain[0][0]
        # compute conductivity
        kappa = 1.0 / np.sqrt(pr * ra)
        self.conductivity = torch.tensor(kappa * (bcT[0] - bcT[1]) / H)

    def update(self, input: Tensor) -> None:
        self.input = input

    def compute(self) -> Tensor:
        return self.input.mean() / self.conductivity
