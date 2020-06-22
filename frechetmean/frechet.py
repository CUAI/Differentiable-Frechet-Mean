import torch

from frechetmean.backward import frechet_ball_backward, frechet_hyperboloid_backward
from frechetmean.forward import frechet_ball_forward, frechet_hyperboloid_forward
from frechetmean.manifolds import Lorentz, Poincare, get_manifold_id

from .utils import TOLEPS


class FrechetMean(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, w, K, manifold_id):
        if manifold_id==0:
            mean = frechet_ball_forward(x, w, K, rtol=TOLEPS[x.dtype], atol=TOLEPS[x.dtype])
        elif manifold_id==1:
            mean = frechet_hyperboloid_forward(x, w, K, rtol=TOLEPS[x.dtype], atol=TOLEPS[x.dtype])
        else:
            raise NotImplementedError

        manifold_id = torch.tensor(manifold_id)
        ctx.save_for_backward(x, mean, w, K, manifold_id)
        return mean

    @staticmethod
    def backward(ctx, grad_output):
        X, mean, w, K, manifold_id = ctx.saved_tensors
        manifold_id = manifold_id.item()

        if manifold_id == 0:
            dx, dw, dK = frechet_ball_backward(X, mean, grad_output, w, K)
        elif manifold_id == 1:
            dx, dw, dK = frechet_hyperboloid_backward(X, mean, grad_output, w, K)
        else:
            raise NotImplementedError
        return dx, dw, dK, None

def frechet_mean(x, manifold, w=None):
    if w is None:
        w = torch.ones(x.shape[:-1]).to(x)
    return FrechetMean.apply(x, w, manifold.K, get_manifold_id(manifold))
