# Differentiating through the Fréchet Mean

We provide code for Differentiating through the Fréchet Mean (https://arxiv.org/abs/2003.00335).

Fréchet Mean         |  Differentiating the Fréchet Mean
:-------------------------:|:-------------------------:
<img src="https://i.imgur.com/VZWGjRM.png" width="70%"> |  <img src="https://i.imgur.com/cy1TMWZ.png" width="100%">

## Installation

### Command

To install, simply run the following commands

```
git clone https://github.com/CUVL/Differentiable-Frechet-Mean.git
cd Differentiable-Frechet-Mean/
python setup.py install
```

### Software Requirements
This codebase requires Python 3, PyTorch 1.5+.

## Usage

### Demo - Frechet Mean Differentiation

```python
import torch
from frechetmean import frechet_mean, Poincare

# Variable Instantiation
man = Poincare()
x = torch.rand(5, 3)
x *= torch.rand(5, 1) / x.norm(dim=-1, keepdim=True)
x = torch.nn.Parameter(x)
w = torch.nn.Parameter(torch.rand(5)) #use ones or pass in None for non-weighted mean

# computation
y = frechet_mean(x, Poincare(), w)

# differentiation
y.sum().backward()
print(x.grad, w.grad)
```

### Demo - Riemannian Batch Normalization

```python
import torch
from frechetmean import Poincare
from riemannian_batch_norm import RiemannianBatchNorm

# Variable Instantiation
man = Poincare()
x = torch.rand(5, 3)
x *= torch.rand(5, 1) / x.norm(dim=-1, keepdim=True)

rbatch_norm = RiemannianBatchNorm(3, man)

# Training
train_normalized = rbatch_norm(x)

# Testing
test_normalized = rbatch_norm(x, training=False)
```

## Attribution

If you use this code or our results in your research, please cite:

```
@article{Lou2020DifferentiatingTT,
  title={Differentiating through the Fr{\'e}chet Mean},
  author={Aaron Lou and Isay Katsman and Qingxuan Jiang and Serge J. Belongie and Ser-Nam Lim and Christopher De Sa},
  journal={ArXiv},
  year={2020},
  volume={abs/2003.00335}
}
```
