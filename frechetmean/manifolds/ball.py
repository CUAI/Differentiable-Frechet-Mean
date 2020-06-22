import torch

from .manifold import Manifold
from frechetmean.utils import EPS, cosh, sinh, tanh, arcosh, arsinh, artanh, sinhdiv, divsinh


class Poincare(Manifold):
    def __init__(self, K=-1.0, edge_eps=1e-3):
        super(Poincare, self).__init__()
        self.edge_eps = 1e-3
        assert K < 0
        if torch.is_tensor(K):
            self.K = K
        else:
            self.K = torch.tensor(K)

    def sh_to_dim(self, sh):
        if hasattr(sh, '__iter__'):
            return sh[-1]
        else:
            return sh

    def dim_to_sh(self, dim):
        if hasattr(dim, '__iter__'):
            return dim[-1]
        else:
            return dim

    def zero(self, *shape):
        return torch.zeros(*shape)

    def zero_tan(self, *shape):
        return torch.zeros(*shape)

    def zero_like(self, x):
        return torch.zeros_like(x)

    def zero_tan_like(self, x):
        return torch.zeros_like(x)

    def lambda_x(self, x, keepdim=False):
        return 2 / (1 + self.K * x.pow(2).sum(dim=-1, keepdim=keepdim)).clamp_min(min=EPS[x.dtype])

    def inner(self, x, u, v, keepdim=False):
        return self.lambda_x(x, keepdim=True).pow(2) * (u * v).sum(dim=-1, keepdim=keepdim)

    def proju(self, x, u):
        return u

    def projx(self, x):
        norm = x.norm(dim=-1, keepdim=True).clamp(min=EPS[x.dtype])
        maxnorm = (1 - self.edge_eps) / (-self.K).sqrt()
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def egrad2rgrad(self, x, u):
        return u / self.lambda_x(x, keepdim=True).pow(2)

    def mobius_addition(self, x, y):
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        y2 = y.pow(2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 - 2 * self.K * xy - self.K * y2) * x + (1 + self.K * x2) * y
        denom = 1 - 2 * self.K * xy + (self.K.pow(2)) * x2 * y2
        return num / denom.clamp_min(EPS[x.dtype])

    def exp(self, x, u):
        u_norm = u.norm(dim=-1, keepdim=True).clamp_min(min=EPS[x.dtype])
        second_term = (
            tanh((-self.K).sqrt() / 2 * self.lambda_x(x, keepdim=True) * u_norm) * u / ((-self.K).sqrt() * u_norm)
        )
        gamma_1 = self.mobius_addition(x, second_term)
        return gamma_1

    def log(self, x, y):
        sub = self.mobius_addition(-x, y)
        sub_norm = sub.norm(dim=-1, keepdim=True).clamp_min(EPS[x.dtype])
        lam = self.lambda_x(x, keepdim=True)
        return 2 / ((-self.K).sqrt() * lam) * artanh((-self.K).sqrt() * sub_norm) * sub / sub_norm

    def dist(self, x, y, squared=False, keepdim=False):
        dist = 2 * artanh((-self.K).sqrt() * self.mobius_addition(-x, y).norm(dim=-1)) / (-self.K).sqrt()
        return dist.pow(2) if squared else dist

    def _gyration(self, u, v, w):
        u2 = u.pow(2).sum(dim=-1, keepdim=True)
        v2 = v.pow(2).sum(dim=-1, keepdim=True)
        uv = (u * v).sum(dim=-1, keepdim=True)
        uw = (u * w).sum(dim=-1, keepdim=True)
        vw = (v * w).sum(dim=-1, keepdim=True)
        a = - self.K.pow(2) * uw * v2 - self.K * vw + 2 * self.K.pow(2) * uv * vw
        b = - self.K.pow(2) * vw * u2 + self.K * uw
        d = 1 - 2 * self.K * uv + self.K.pow(2) * u2 * v2
        return w + 2 * (a * u + b * v) / d.clamp_min(EPS[u.dtype])

    def transp(self, x, y, u):
        return (
            self._gyration(y, -x, u)
            * self.lambda_x(x, keepdim=True)
            / self.lambda_x(y, keepdim=True)
        )

    def __str__(self):
        return 'Poincare Ball'

    def squeeze_tangent(self, x):
        return x

    def unsqueeze_tangent(self, x):
        return x
