import math

import torch

EPS = {torch.float32: 1e-4, torch.float64: 1e-8}
TOLEPS = {torch.float32: 1e-6, torch.float64: 1e-12}


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + EPS[x.dtype], 1 - EPS[x.dtype])
        ctx.save_for_backward(x)
        res = (torch.log(1 + x).sub(torch.log(1 - x))).mul(0.5)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


class Arsinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        positive_case = x + torch.sqrt(1 + x.pow(2))
        negative_case = 1 / (torch.sqrt(1 + x.pow(2)) - x)
        return torch.where(x > 0, positive_case, negative_case).log()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 + input ** 2) ** 0.5


class Acosh(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        x = torch.clamp(x, min=1+EPS[x.dtype])
        z = torch.sqrt(x * x - 1)
        ctx.save_for_backward(z)
        return torch.log(x + z)

    @staticmethod
    def backward(ctx, g):
        z, = ctx.saved_tensors
        z.data.clamp(min=EPS[z.dtype])
        z = g / z
        return z, None


artanh = Artanh.apply


arsinh = Arsinh.apply


arcosh = Acosh.apply

cosh_bounds = {torch.float32: 85, torch.float64: 700}
sinh_bounds = {torch.float32: 85, torch.float64: 500}


def cosh(x):
    x.data.clamp_(max=cosh_bounds[x.dtype])
    return torch.cosh(x)


def sinh(x):
    x.data.clamp_(max=sinh_bounds[x.dtype])
    return torch.sinh(x)


def tanh(x):
    return x.tanh()


def sqrt(x):
    return torch.sqrt(x).clamp_min(EPS[x.dtype])


class Sinhdiv(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        y = sinh(x) / x
        y_stable = torch.ones_like(x)
        ctx.save_for_backward(x)
        return torch.where(x < EPS[x.dtype], y_stable, y)

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        y = (x * cosh(x) - sinh(x)) / x.pow(2)
        y_stable = torch.zeros_like(x)
        return torch.where(x < EPS[x.dtype], y_stable, y) * g


sinhdiv = Sinhdiv.apply


class Divsinh(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        y = x / sinh(x)
        y_stable = torch.ones_like(x)
        ctx.save_for_backward(x)
        return torch.where(x < EPS[x.dtype], y_stable, y)

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        y = (1 - x * cosh(x) / sinh(x)) / sinh(x)
        y_stable = torch.zeros_like(x)
        return torch.where(x < EPS[x.dtype], y_stable, y) * g


divsinh = Divsinh.apply

def darcosh(x):
    cond = (x < 1 + 1e-7)
    x = torch.where(cond, 2 * torch.ones_like(x), x)
    x = torch.where(~cond, 2 * arcosh(x) / torch.sqrt(x**2 - 1), x)
    return x


def d2arcosh(x):
    cond = (x < 1 + 1e-7)
    x = torch.where(cond, -2/3 * torch.ones_like(x), x)
    x = torch.where(~cond, 2 / (x**2 - 1) - 2 * x * arcosh(x) / ((x**2 - 1)**(3/2)), x)
    return x


class DAcoshSq(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        y = 2 * arcosh(x) / (x.pow(2) - 1).sqrt()
        y_stable = 2 * torch.ones_like(x)
        ctx.save_for_backward(x)
        return torch.where(x < 1 + EPS[x.dtype], y_stable, y)

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        y = 2 / (x.pow(2) - 1) - 2 * x * arcosh(x) / (x.pow(2) - 1).pow(3/2)
        y_stable = -2/3 * torch.ones_like(x)
        return torch.where(x < 1 + EPS[x.dtype], y_stable, y) * g

darcoshsq_diff = DAcoshSq.apply