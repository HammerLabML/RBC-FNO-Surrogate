# Inspired by
# https://github.com/NickGeneva/blog-code/blob/master/koopman-intro/models/koopmanNN.py

import torch
import torch.nn as nn
from torch import Tensor


class KoopmanOperator(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        # Parameters
        self.dim = dim
        # Skew-symmetric matrix with a diagonal
        self.K_diag = nn.Parameter(torch.rand(dim))
        self.K_upper = nn.Parameter(0.01 * torch.randn(int(dim * (dim - 1) / 2)))

    def forward(self, g: Tensor) -> Tensor:
        """Applies the learned koopman operator on the given observables.

        Args:
            g (Tensor): [b x g] batch of observables, must match dim of koopman transform
        Returns:
            (Tensor): [b x g] predicted observables at the next time-step
        """
        assert g.shape[-1] == self.dim, (
            f"Observables should have dim {self.dim}. Observable has dim {g.shape[-1]}"
        )

        # Get Koopman Operator
        K = self.get_operator()

        # Apply Koopman
        K_batches = K.expand(g.size(0), K.size(0), K.size(0))
        g_next = torch.bmm(g.unsqueeze(1), K_batches)
        return g_next.squeeze(1)

    def get_operator(self, requires_grad=False) -> Tensor:
        """Returns current Koopman operator matrix.

        Args:
        requires_grad (bool, optional): if the parameter requires gradient. Default: `False`
        Returns:
            (Tensor): [dim x dim] Koopman Operator Matrix
        """
        K = torch.autograd.Variable(
            torch.Tensor(self.dim, self.dim), requires_grad=requires_grad
        ).to(self.K_diag.device)

        upper_idx = torch.triu_indices(self.dim, self.dim, offset=1)
        diag_idx = torch.stack(
            [
                torch.arange(0, self.dim, dtype=torch.long).unsqueeze(0),
                torch.arange(0, self.dim, dtype=torch.long).unsqueeze(0),
            ],
            dim=0,
        )
        K[upper_idx[0], upper_idx[1]] = self.K_upper
        K[upper_idx[1], upper_idx[0]] = -self.K_upper
        K[diag_idx[0], diag_idx[1]] = torch.nn.functional.relu(self.K_diag)

        return K
