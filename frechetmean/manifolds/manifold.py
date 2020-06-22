import abc

import numpy as np
import torch

from frechetmean.utils import EPS


class Manifold(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def zero(self, *shape):
        pass

    @abc.abstractmethod
    def zero_like(self, x):
        pass

    @abc.abstractmethod
    def zero_tan(self, *shape):
        pass

    @abc.abstractmethod
    def zero_tan_like(self, x):
        pass

    @abc.abstractmethod
    def inner(self, x, u, v, keepdim=False):
        pass

    def norm(self, x, u, squared=False, keepdim=False):
        norm_sq = self.inner(x, u, u, keepdim)
        norm_sq.data.clamp_(EPS[u.dtype])
        return norm_sq if squared else norm_sq.sqrt()

    @abc.abstractmethod
    def proju(self, x, u):
        pass

    def proju0(self, u):
        return self.proju(self.zero_like(u), u)

    @abc.abstractmethod
    def projx(self, x):
        pass

    def egrad2rgrad(self, x, u):
        return self.proju(x, u)

    @abc.abstractmethod
    def exp(self, x, u):
        pass

    def exp0(self, u):
        return self.exp(self.zero_like(u), u)

    @abc.abstractmethod
    def log(self, x, y):
        pass

    def log0(self, y):
        return self.log(self.zero_like(y), y)
        
    def dist(self, x, y, squared=False, keepdim=False):
        return self.norm(x, self.log(x, y), squared, keepdim)

    def pdist(self, x, squared=False):
        assert x.ndim == 2
        n = x.shape[0]
        m = torch.triu_indices(n, n, 1, device=x.device)
        return self.dist(x[m[0]], x[m[1]], squared=squared, keepdim=False)

    def transp(self, x, y, u):
        return self.proju(y, u)

    def transpfrom0(self, x, u):
        return self.transp(self.zero_like(x), x, u)
    
    def transpto0(self, x, u):
        return self.transp(x, self.zero_like(x), u)

    def mobius_addition(self, x, y):
        return self.exp(x, self.transp(self.zero_like(x), x, self.log0(y)))

    @abc.abstractmethod
    def sh_to_dim(self, shape):
        pass

    @abc.abstractmethod
    def dim_to_sh(self, dim):
        pass

    @abc.abstractmethod
    def squeeze_tangent(self, x):
        pass

    @abc.abstractmethod
    def unsqueeze_tangent(self, x):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass

    def frechet_variance(self, x, mu, w=None):
        """
        Args
        ----
            x (tensor): points of shape [..., points, dim]
            mu (tensor): mean of shape [..., dim]
            w (tensor): weights of shape [..., points]

            where the ... of the three variables line up
        
        Returns
        -------
            tensor of shape [...]
        """
        distance = self.dist(x, mu.unsqueeze(-2), squared=True)
        if w is None:
            return distance.mean(dim=-1)
        else:
            return (distance * w).sum(dim=-1)
