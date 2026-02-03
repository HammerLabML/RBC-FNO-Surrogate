import torch
from torch import Tensor
from torch.distributions import Normal


def mc_kld(z: Tensor, mu: Tensor, std: Tensor) -> Tensor:
    # define the first two probabilities (in this case Normal for both)
    p = Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = Normal(mu, std)
    # get the probabilities from the equation
    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)
    # kl
    kl = log_qzx - log_pz
    kl = kl.sum(-1)
    return kl


def gaussian_likelihood(x_hat: Tensor, logscale: Tensor, x: Tensor):
    scale = torch.exp(logscale)
    mean = x_hat
    dist = torch.distributions.Normal(mean, scale)
    # measure prob of seeing image under p(x|z)
    log_pxz = dist.log_prob(x)
    return log_pxz.sum(dim=(1, 2, 3))


def elbo(
    x: Tensor,
    x_hat: Tensor,
    logscale: Tensor,
    z: Tensor,
    mu: Tensor,
    std: Tensor,
    beta: float = 1.0,
) -> Tensor:
    recon_loss = gaussian_likelihood(x_hat, logscale, x)
    kl = mc_kld(z, mu, std)
    return (beta * kl - recon_loss).mean()
