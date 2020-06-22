import torch
import torch.nn as nn
import math

from frechetmean.manifolds import Poincare, Lorentz
from frechetmean.frechet import frechet_mean


class RiemannianBatchNorm(nn.Module):
    def __init__(self, dim, manifold):
        super(RiemannianBatchNorm, self).__init__()
        self.man = manifold

        self.mean = nn.Parameter(self.man.zero_tan(self.man.dim_to_sh(dim)))
        self.var = nn.Parameter(torch.tensor(1.0))

        # statistics
        self.running_mean = None
        self.running_var = None
        self.updates = 0

    def forward(self, x, training=True, momentum=0.9):
        on_manifold = self.man.exp0(self.mean)
        if training:
            # frechet mean, use iterative and don't batch (only need to compute one mean)
            input_mean = frechet_mean(x, self.man)
            input_var = self.man.frechet_variance(x, input_mean)

            # transport input from current mean to learned mean
            input_logm = self.man.transp(
                input_mean,
                on_manifold,
                self.man.log(input_mean, x),
            )

            # re-scaling
            input_logm = (self.var / (input_var + 1e-6)).sqrt() * input_logm

            # project back
            output = self.man.exp(on_manifold.unsqueeze(-2), input_logm)

            self.updates += 1
            if self.running_mean is None:
                self.running_mean = input_mean
            else:
                self.running_mean = self.man.exp(
                    self.running_mean,
                    (1 - momentum) * self.man.log(self.running_mean, input_mean)
                )
            if self.running_var is None:
                self.running_var = input_var
            else:
                self.running_var = (
                    1 - 1 / self.updates
                ) * self.running_var + input_var / self.updates
        else:
            if self.updates == 0:
                raise ValueError("must run training at least once")

            input_mean = frechet_mean(x, self.man)
            input_var = self.man.frechet_variance(x, input_mean)

            input_logm = self.man.transp(
                input_mean,
                self.running_mean,
                self.man.log(input_mean, x),
            )

            assert not torch.any(torch.isnan(input_logm))

            # re-scaling
            input_logm = (
                self.running_var / (x.shape[0] / (x.shape[0] - 1) * input_var + 1e-6)
            ).sqrt() * input_logm

            # project back
            output = self.man.exp(on_manifold, input_logm)

        return output
