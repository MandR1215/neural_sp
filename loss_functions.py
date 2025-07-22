"""
Modified from original code by Sai Srivatsa Ravindranath
Licensed under the MIT License.
For full license terms, see https://github.com/saisrivatsan/deep-matching/
"""
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import List

def compute_st_unit(r:Tensor, p:Tensor, q:Tensor):
    """stv for one sample.

    Args:
        r (Tensor): shape (n,m)
        p (Tensor): shape (n,m) (p_{wf} = degree of preference of worker w for firm f)
        q (Tensor): shape (n,m) (q_{wf} = degree of preference of firm f for worker w)

    Returns:
        Tensor: stability violation
    """
    # import pdb; pdb.set_trace()
    assert r.ndim == p.ndim == q.ndim == 2
    assert r.shape == p.shape == q.shape

    n, m = r.shape
    wp = nn.functional.relu(p[:, None, :] - p[:, :, None])
    wq = nn.functional.relu(q[:, None, :] - q[None, :, :])  
    t = (1 - torch.sum(r, dim=0, keepdim=True))
    s = (1 - torch.sum(r, dim=1, keepdim=True))
    rgt_1 = torch.einsum('jk,ijk->ik', r, wq) + t * nn.functional.relu(q)
    rgt_2 = torch.einsum('ij,ijk->ik', r, wp) + s * nn.functional.relu(p)
    regret =  rgt_1 * rgt_2
    return regret.sum() * (1/n + 1/m) * 0.5

def compute_st_batch(batch: List[List[Tensor]]) -> Tensor:
    """stv for batch.

    Args:
        batch (List[List[Tensor]]): batch of samples formed by [r,p,q]

    Returns:
        Tensor: mean of stv.
    """

    return torch.mean(torch.cat([compute_st_unit(r,p,q).unsqueeze(0) for [r,p,q] in batch]))

def compute_ir_unit(r:Tensor, p:Tensor, q:Tensor):
    """irv for one sample.

    Args:
        r (Tensor): shape (n,m)
        p (Tensor): shape (n,m) (p_{wf} = degree of preference of worker w for firm f)
        q (Tensor): shape (n,m) (q_{wf} = degree of preference of firm f for worker w)

    Returns:
        Tensor: IR violation
    """
    assert r.ndim == p.ndim == q.ndim == 2
    assert r.shape == p.shape == q.shape
    #import pdb; pdb.set_trace()
    n, m = r.shape
    ir_1 = r * F.relu(-q)
    ir_2 = r * F.relu(-p)
    ir = ir_1.sum() / (2*m) + ir_2.sum() / (2*n)
    return ir

def compute_ir_batch(batch: List[List[Tensor]]) -> Tensor:
    """irv for batch.

    Args:
        batch (List[List[Tensor]]): batch of samples formed by [r,p,q]

    Returns:
        Tensor: mean of irv.
    """

    return torch.mean(torch.cat([compute_ir_unit(r,p,q).unsqueeze(0) for [r,p,q] in batch]))
