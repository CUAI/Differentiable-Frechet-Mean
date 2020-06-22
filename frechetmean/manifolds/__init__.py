import torch
from .ball import Poincare
from .hyperboloid import Lorentz

manifold_id = {Poincare:0, Lorentz: 1}

def get_manifold_id(x):
    if isinstance(x, Poincare):
        return 0
    elif isinstance(x, Lorentz):
        return 1
    else:
        raise NotImplementedError

#TODO don't need to include this
def to_ball(x, K):
    R = 1 / (-K).sqrt()
    return R * x[..., 1:] / (R + x[..., :1])

def to_hyperboloid(x, K):
    R = 1/ (-K).sqrt()
    xnormsq = x.norm(dim=-1, keepdim=True).pow(2)
    sec_part = 2 * R.pow(2) * x / (R.pow(2) - xnormsq)
    first_part = R * (R.pow(2) + xnormsq) / (R.pow(2) - xnormsq)
    return torch.cat((first_part, sec_part), dim=-1)
