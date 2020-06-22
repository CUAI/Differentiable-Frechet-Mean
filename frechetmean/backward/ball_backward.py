import torch
import torch.nn as nn

from frechetmean.utils import d2arcosh, darcosh


def grad_var(X, y, w, K):
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
    yl = y.unsqueeze(-2)
    xnorm = 1 + K * X.norm(dim=-1).pow(2)
    ynorm = 1 + K * yl.norm(dim=-1).pow(2)
    xynorm = (X - yl).norm(dim=-1).pow(2)

    D = xnorm * ynorm
    v = 1 - 2 * K * xynorm / D

    Dl = D.unsqueeze(-1)
    vl = v.unsqueeze(-1)

    first_term = (X - yl) / Dl
    sec_term = K / Dl.pow(2) * yl * xynorm.unsqueeze(-1) * xnorm.unsqueeze(-1)
    return -(4 * darcosh(vl) * w.unsqueeze(-1) * (first_term + sec_term)).sum(dim=-2)
    
def inverse_hessian(X, y, w, K):
    """
    Args
    ----
        X (tensor): point of shape [..., points, dim]
        y (tensor): mean point of shape [..., dim]
        w (tensor): weight tensor of shape [..., points]
        K (float): curvature (must be negative)

    Returns
    -------
        inv_hess (tensor): inverse hessian of [..., points, dim, dim]
    """
    yl = y.unsqueeze(-2)
    xnorm = 1 + K * X.norm(dim=-1).pow(2)
    ynorm = 1 + K * yl.norm(dim=-1).pow(2)
    xynorm = (X - yl).norm(dim=-1).pow(2)

    D = xnorm * ynorm
    v = 1 - 2 * K * xynorm / D

    Dl = D.unsqueeze(-1)
    vl = v.unsqueeze(-1)
    vll = vl.unsqueeze(-1)

    """
    \partial T/ \partial y
    """
    first_const = -8 * (K ** 2) * xnorm / D.pow(2)
    matrix_val = (first_const.unsqueeze(-1) * yl).unsqueeze(-1) * (X - yl).unsqueeze(-2)
    first_term = matrix_val + matrix_val.transpose(-1, -2)

    sec_const = -16 * (K ** 3) * xnorm.pow(2) / D.pow(3) * xynorm
    sec_term = (sec_const.unsqueeze(-1) * yl).unsqueeze(-1) * yl.unsqueeze(-2)

    third_const = -4 * K / D + 4 * (K ** 2) * xnorm /D.pow(2) * xynorm
    third_term = third_const.reshape(*third_const.shape, 1, 1) * torch.eye(y.shape[-1]).to(X).reshape((1, ) * len(third_const.shape) + (y.shape[-1], y.shape[-1]))

    Ty = first_term + sec_term + third_term

    """
    T
    """
    
    first_term = K / Dl * (X - yl)
    sec_term = K.pow(2) / Dl.pow(2) * yl * xynorm.unsqueeze(-1) * xnorm.unsqueeze(-1)
    T = 4 * (first_term + sec_term)

    """
    inverse of shape [..., points, dim, dim]
    """
    first_term = d2arcosh(vll) * T.unsqueeze(-1) * T.unsqueeze(-2)
    sec_term = darcosh(vll) * Ty
    hessian = ((first_term + sec_term) * w.unsqueeze(-1).unsqueeze(-1)).sum(dim=-3) / -K
    inv_hess = torch.inverse(hessian)
    return inv_hess


def frechet_ball_backward(X, y, grad, w, K):
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
        inv_hess = inverse_hessian(X, y, w=w, K=K)

    with torch.enable_grad():
        # clone variables
        X = nn.Parameter(X.detach())
        y = y.detach()
        w = nn.Parameter(w.detach())
        K = nn.Parameter(K)

        grad = (inv_hess @ grad.unsqueeze(-1)).squeeze()
        gradf = grad_var(X, y, w, K)
        dx, dw, dK = torch.autograd.grad(-gradf, (X, w, K), grad)

    return dx, dw, dK
