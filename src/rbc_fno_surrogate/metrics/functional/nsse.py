import torch


def normalized_sum_squared_error(x_hat, x):
    eps = torch.finfo(x.dtype).eps
    nom = torch.square(torch.linalg.norm(x_hat - x))
    denom = torch.square(torch.linalg.norm(x)) + eps
    return nom / denom
