import torch

from frechetmean.utils import (EPS, arcosh, arsinh, artanh, cosh, divsinh,
                               sinh, sinhdiv, tanh)

from .manifold import Manifold


class Lorentz(Manifold):

    def __init__(self, K=-1.0):
        super(Lorentz, self).__init__()
        assert K < 0
        if torch.is_tensor(K):
            self.K = K
        else:
            self.K = torch.tensor(K)

    @staticmethod
    def _ldot(u, v, keepdim=False, dim=-1):
        m = u * v
        if keepdim:
            ret = torch.sum(m, dim=dim, keepdim=True) - 2 * m[..., 0:1]
        else:
            ret = torch.sum(m, dim=dim, keepdim=False) - 2 * m[..., 0]
        return ret

    def ldot(self, u, v, keepdim=False, dim=-1):
        return Lorentz._ldot(u, v, keepdim, dim)

    def sh_to_dim(self, sh):
        if hasattr(sh, '__iter__'):
            return sh[-1] - 1
        else:
            return sh - 1

    def dim_to_sh(self, dim):
        if hasattr(dim, '__iter__'):
            return dim[-1] + 1
        else:
            return dim + 1

    def zero(self, *shape):
        x = torch.zeros(*shape)
        x[..., 0] = 1 / (-self.K).sqrt().detach()
        return x

    def zero_tan(self, *shape):
        return torch.zeros(*shape)

    def zero_like(self, x):
        y = torch.zeros_like(x)
        y[..., 0] = 1 / (-self.K).sqrt().detach()
        return y

    def zero_tan_like(self, x):
        return torch.zeros_like(x)

    def inner(self, x, u, v, keepdim=False):
        return self.ldot(u, v, keepdim=keepdim)

    def proju(self, x, u):
        return u - self.K * self.ldot(x, u, keepdim=True).expand_as(u) * x.expand_as(u)

    @staticmethod
    def _proju(x, u, K):
        return u - K * Lorentz._ldot(x, u, keepdim=True).expand_as(u) * x.expand_as(u)

    def projx(self, x):
        x = x.clone()
        x.data[..., 0] = (1 / (-self.K) + x[..., 1:].pow(2).sum(dim=-1)).sqrt()
        return x

    def egrad2rgrad(self, x, u):
        scaling = torch.zeros_like(x)
        scaling[..., :1] = torch.ones_like(scaling[..., :1])
        u = u - 2 * x[..., :1] * scaling
        u = self.proju(x, u)
        return u

    def exp(self, x, u):
        un = self.ldot(u, u, keepdim=True)
        un = un.clamp(min=EPS[x.dtype]).sqrt() * (-self.K).sqrt()
        return x * cosh(un) + sinhdiv(un) * u

    def log(self, x, y):
        xy = self.K * self.ldot(x, y, keepdim=True)
        num = arcosh(xy)
        u = divsinh(num) * (y - xy * x)
        return self.proju(x, u)

    def dist(self, x, y, squared=False, keepdim=False):
        d = self.K * self.ldot(x, y)
        d.data.clamp(min=1)
        dist = arcosh(d) / (-self.K).sqrt()
        dist.data.clamp(min=EPS[x.dtype])
        return dist.pow(2) if squared else dist

    def transp(self, x, y, u):
        xy = self.ldot(x, y, keepdim=True).expand_as(u)
        uy = self.ldot(u, y, keepdim=True).expand_as(u)
        return u - (self.K * uy) / (1 + self.K * xy) * (x + y).expand_as(u)

    def __str__(self):
        return 'Hyperboloid'

    def squeeze_tangent(self, x):
        return x[..., 1:]

    def unsqueeze_tangent(self, x):
        return torch.cat((torch.zeros_like(x[..., 0]).unsqueeze(-1), x), dim=-1)
