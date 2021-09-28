import torch
import torch.nn.functional as F
from frechetmean.frechet import frechet_mean


def frechet_agg(x, adj, man):
    """
    Compute Frechet Graph Aggregation

    Args
    ----
        x (tensor): batched tensor of hyperbolic values. Batch size is size of adjacency matrix.
        adj (tensor): sparse coalesced adjacency matrix
        man (Manifold): hyperbolic manifold
    """
    indices = adj.indices().transpose(-1, -2)
    n, d = x.shape
    B = max([sum(indices[:, 0] == i) for i in range(n)])

    batched_tensor = []
    weight_tensor = []

    for i in range(n):
        si = indices[indices[:, 0] == i, -1]
        batched_tensor.append(F.pad(x[si], (0, 0, 0, B - len(si))))
        weight_tensor.append(F.pad(torch.ones_like(si), (0, B - len(si))))
    batched_tensor = torch.stack(batched_tensor)
    weight_tensor = torch.stack(weight_tensor)

    return frechet_mean(batched_tensor, man, w=weight_tensor)
