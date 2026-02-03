import torch


def normalized_sum_error(x_hat, x):
    eps = torch.finfo(x.dtype).eps
    nom = torch.linalg.norm(x_hat - x)
    denom = torch.linalg.norm(x) + eps
    return nom / denom
