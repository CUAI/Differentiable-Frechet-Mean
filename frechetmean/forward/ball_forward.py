import torch

from frechetmean.utils import EPS, arcosh


def l_prime(y):
    cond = y < 1e-12
    val = 4 * torch.ones_like(y)
    ret = torch.where(cond, val, 2 * arcosh(1 + 2 * y) / (y.pow(2) + y).sqrt())
    return ret


def frechet_ball_forward(X, w, K=-1.0, max_iter=1000, rtol=1e-6, atol=1e-6, verbose=False):
    """
    Args
    ----
        X (tensor): point of shape [..., points, dim]
        w (tensor): weights of shape [..., points]
        K (float): curvature (must be negative)
    Returns
    -------
        frechet mean (tensor): shape [..., dim]
    """
    mu = X[..., 0, :].clone()

    x_ss = X.pow(2).sum(dim=-1)

    mu_prev = mu
    iters = 0
    for _ in range(max_iter):
        mu_ss = mu.pow(2).sum(dim=-1)
        xmu_ss = (X - mu.unsqueeze(-2)).pow(2).sum(dim=-1)

        alphas = l_prime(-K * xmu_ss / ((1 + K * x_ss) * (1 + K * mu_ss.unsqueeze(-1)))) / (1 + K * x_ss)

        alphas = alphas * w

        c = (alphas * x_ss).sum(dim=-1)
        b = (alphas.unsqueeze(-1) * X).sum(dim=-2)
        a = alphas.sum(dim=-1)

        b_ss = b.pow(2).sum(dim=-1)

        eta = (a - K * c - ((a - K * c).pow(2) + 4 * K * b_ss).sqrt()) / (2 * (-K) * b_ss)

        mu = eta.unsqueeze(-1) * b

        dist = (mu - mu_prev).norm(dim=-1)
        prev_dist = mu_prev.norm(dim=-1)
        if (dist < atol).all() or (dist / prev_dist < rtol).all():
            break

        mu_prev = mu
        iters += 1

    if verbose:
        print(iters)

    return mu
