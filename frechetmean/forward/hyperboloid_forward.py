import torch

from frechetmean.manifolds import Lorentz
from frechetmean.utils import EPS, darcosh


def frechet_hyperboloid_forward(X, w, K=-1.0, max_iter=1000, rtol=1e-6, atol=1e-6, verbose=False):
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

    mu_prev = mu
    iters = 0
    for _ in range(max_iter):
        inner = K * Lorentz._ldot(X, mu.unsqueeze(-2), keepdim=True)
        u = (w.unsqueeze(-1) * darcosh(inner) * X).sum(dim=-2)
        mu = u / (K * Lorentz._ldot(u, u, keepdim=True)).sqrt()

        dist = (mu - mu_prev).norm(dim=-1)
        prev_dist = mu_prev.norm(dim=-1)

        if (dist < atol).all() or (dist / prev_dist < rtol).all():
            break

        mu_prev = mu
        iters += 1

    if verbose:
        print(iters)

    return mu
