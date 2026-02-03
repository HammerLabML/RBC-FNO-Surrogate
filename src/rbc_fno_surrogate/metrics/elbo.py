from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat

from .functional.elbo import elbo


class EvidenceLowerBound(Metric):
    def __init__(self, beta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.add_state("elbo", default=[], dist_reduce_fx="cat")

    def update(
        self,
        target: Tensor,
        preds: Tensor,
        logscale: Tensor,
        z: Tensor,
        mu: Tensor,
        std: Tensor,
    ) -> None:
        self.elbo.append(elbo(target, preds, logscale, z, mu, std, self.beta))

    def compute(self):
        preds = dim_zero_cat(self.elbo)
        return preds.mean()
