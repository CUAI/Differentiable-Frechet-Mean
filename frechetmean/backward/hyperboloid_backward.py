import torch
import torch.nn as nn

from frechetmean.manifolds import Lorentz
from frechetmean.utils import d2arcosh, darcosh, darcoshsq_diff

def hessian(X, y, w, K):
    """
    Args
    ----
        X (tensor): point of shape [..., points, dim]
        y (tensor): mean point of shape [..., dim]
        w (tensor): weight tensor of shape [..., points]
        K (float): curvature (must be negative)

    Returns
    -------
        hess (tensor): inverse hessian of [..., points, dim, dim]
    """
    X = X.clone()
    X[..., 0] *= -1
    xlT_M_y = (X * y.unsqueeze(-2)).sum(dim=-1)

    term1 = K**2 * d2arcosh(K * xlT_M_y).unsqueeze(-1).unsqueeze(-1) * (X.unsqueeze(-1) @ X.unsqueeze(-2))

    M = torch.diag(torch.tensor([-1.] + [1 for i in range(term1.shape[-1]-1)])).to(X)
    M = M.reshape((1,) * (len(term1.shape) - 2) + (term1.shape[-1], term1.shape[-1]))
    term2 = (K * darcosh(K * xlT_M_y) * xlT_M_y).unsqueeze(-1).unsqueeze(-1) * M

    return (w.unsqueeze(-1).unsqueeze(-1) * (term1 - K * term2)).sum(dim=-3) / -K


def hess_term(X, y, w, K):
    H = hessian(X.clone(), y, w, K)
    Hi = torch.inverse(H)

    mu = y.clone()
    mu[..., 0] *= -1
    mu = mu.unsqueeze(-1)

    num = Hi @ mu @ mu.transpose(-1, -2) @ Hi
    denom = mu.transpose(-1, -2) @ Hi @ mu
    return (num / denom) - Hi


def gradu(X, y, w, K):
    """
    Args
    ----
        X (tensor): point of shape [..., points, dim]
        y (tensor): mean point of shape [..., dim]
        w (tensor): weight tensor of shape [..., points]
        K (float): curvature (must be negative)

    Returns
    -------
        grad (tensor): gradient of variance [..., dim]
    """
    scalar = torch.zeros_like(X)
    scalar[..., 0] = 2 * torch.ones_like(X[..., 0])
    X_M = X - scalar * X
    xlT_M_y = (X_M * y.unsqueeze(-2)).sum(dim=-1, keepdim=True)

    main_term = -darcoshsq_diff(K * xlT_M_y) * X_M

    return (w.unsqueeze(-1) * main_term).sum(dim=-2)


def frechet_hyperboloid_backward(X, y, grad, w, K):
    """
    Args
    ----
        X (tensor): point of shape [..., points, dim]
        y (tensor): mean point of shape [..., dim]
        grad (tensor): gradient
        K (float): curvature (must be negative)

    Returns
    -------
        gradients (tensor, tensor, tensor): 
            gradient of X [..., points, dim], weights [..., dim], curvature []
    """
    if not torch.is_tensor(K):
        K = torch.tensor(K).to(X)

    with torch.no_grad():
        hess_t = hess_term(X, y, w=w, K=K)

    with torch.enable_grad():
        # clone variables
        X = nn.Parameter(X.detach())
        y = y.detach()
        w = nn.Parameter(w.detach())
        K = nn.Parameter(K)

        grad = (hess_t @ grad.unsqueeze(-1)).squeeze()
        gradf = gradu(X, y, w, K)
        dx, dw, dK = torch.autograd.grad(gradf, (X, w, K), grad)

    return dx, dw, dK
