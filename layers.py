import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple
from abc import ABCMeta, abstractmethod
import ot

class TriangleWindowFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)

        output = torch.zeros_like(x)
        output[x > 0] = x[x > 0]
        output[x > 1] = 2 - x[x > 1]
        output[x > 2] = 0
        return output.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_x = torch.zeros_like(x)

        grad_x[x > 0] = 1
        grad_x[x > 1] = -1
        grad_x[x <= 0] = 0
        grad_x[x > 2] = 0
        return grad_x * grad_output

"""
Modified from original code by Sebastián Prillo and Julián Eisenschlos
Licensed under the MIT License (see https://github.com/sprillo/softsort for details).
"""    
class SoftSort(torch.nn.Module):
    def __init__(self, tau=1.0, hard=False, pow=1.0, t=None, how:str=None):
        super(SoftSort, self).__init__()
        assert (not hard) or (hard and how == 'scatter' or how == 'threshold' and 0.0 < t < 1.0)
        self.hard = hard
        self.t = t
        self.how = how
        self.tau = tau
        self.pow = pow

    def forward(self, scores: Tensor):
        """
        scores: elements to be sorted. Typical shape: batch_size x n
        """
        scores = scores.unsqueeze(-1)
        sorted = scores.sort(descending=True, dim=1)[0]
        pairwise_diff = (scores.transpose(1, 2) - sorted).abs().pow(self.pow).neg() / self.tau
        P_hat = pairwise_diff.softmax(-1)

        if self.hard:
            if self.how == 'scatter':
                P = torch.zeros_like(P_hat, device=P_hat.device)
                P.scatter_(-1, P_hat.topk(1, -1)[1], value=1)
                P_hat = (P - P_hat).detach() + P_hat
            else:
                P_hat = F.relu((1.0 / (1.0 - self.t)) * (P_hat - self.t))
                P_hat = P_hat / torch.sum(P_hat, dim=1, keepdim=True)
        return P_hat
    

class TensorSerialDictatorship(nn.Module):
    def __init__(self) -> None:
        super(TensorSerialDictatorship, self).__init__()
        
    # Algorithm 1: CreateRankingMasks
    def create_ranking_masks(self, R:Tensor) -> Tuple[Tensor, Tensor]:
        # Line 1, 2: init ranking masks
        ranking_masks_for_workers = torch.zeros(size=(self.n_firms, self.n_firms+1, self.n_firms+1))
        ranking_masks_for_firms = torch.zeros(size=(self.n_workers, self.n_workers+1, self.n_workers+1))
        
        # Line 5: create ranking masks for workers
        for f in range(self.n_firms):
            ranking_masks_for_workers[f, f, :] = -1.0
            
        # Line 6: create ranking masks for firms
        for w in range(self.n_workers):
            ranking_masks_for_firms[w, w, :] = -1.0

        # Line 4-6: create ranking masks (jointly compute the sum)
        worker_ranking = R[:self.n_workers].T
        firm_ranking = R[self.n_workers:].T

        ranking_masks_for_workers = torch.einsum('nf,fij->nij', firm_ranking, ranking_masks_for_workers)
        ranking_masks_for_firms = torch.einsum('nw,wij->nij', worker_ranking, ranking_masks_for_firms)

        return ranking_masks_for_workers, ranking_masks_for_firms
    
    # Algorithm 2: FindCounterpart
    def find_counterpart(self, P:Tensor) -> Tensor:
        # Line 1-3: colsum -> cumsum -> triangle_window
        first_j = TriangleWindowFunction.apply(torch.cumsum(torch.sum(P, axis=0), axis=0))
        correspondence = P @ first_j.unsqueeze(1) # Line 4: Pc^\top
        
        return correspondence
    
    # Algorithm 3: TSD
    def forward(self, W:Tensor, F:Tensor, R:Tensor) -> Tensor:
        """Matrix formulation of serial dictatorship.
        Input:
            W: (n_workers, n_firms+1, n_firms+1) with W[i][j][k] = 1 meaning worker i places firm [j] at position k (one of j denotes unmatch)
            F: (n_firms, n_workers+1, n_workers+1) with F[i][j][k] = 1 meaning firm i places worker [j] at position k (one of j denotes unmatch)
            R: (n_workers + n_firms, n_workers + n_firms) of the permutation (= ranking) matrix over workers and firms
        """
        self.n_workers = W.shape[0]
        self.n_firms = F.shape[0]
        device = W.device

        # Line 1: create ranking masks
        ranking_masks_for_workers, ranking_masks_for_firms = self.create_ranking_masks(R)

        # Line 2: init matching result
        matching = torch.zeros(size=(self.n_workers+1, self.n_firms+1), device=device)
        for r in range(R.shape[1]):
            # Line 4: dictator at this rank r
            d_w, d_f = R[:self.n_workers, r], R[self.n_workers:, r]

            # Line 5: dictator's preference
            P_w = torch.einsum('wij,w->ij', W, d_w)
            P_f = torch.einsum('fij,f->ij', F, d_f)

            # Line 6,7: counterpart of the dictator
            c_w = self.find_counterpart(P_w)
            c_f = self.find_counterpart(P_f)

            # Line 8,9: update result
            M_w = torch.cat([d_w, torch.tensor([0.0])]).unsqueeze(1) @ c_w.T
            M_f = torch.cat([d_f, torch.tensor([0.0])]).unsqueeze(1) @ c_f.T
            assert matching.shape == M_w.shape == M_f.T.shape
            matching = matching + M_w + M_f.T

            # Line 10,11: compute matching mask
            matching_mask_for_workers = (-1.0) * (c_w * torch.tensor([[1.0]]*self.n_firms + [[0.0]], device=device)).repeat(1,self.n_firms+1)
            matching_mask_for_firms = (-1.0) * (c_f * torch.tensor([[1.0]]*self.n_workers + [[0.0]], device=device)).repeat(1,self.n_workers+1)

            # Line 12-16: apply mask
            assert W[0].shape == matching_mask_for_workers.shape == ranking_masks_for_workers[r].shape
            assert F[0].shape == matching_mask_for_firms.shape == ranking_masks_for_firms[r].shape
            W = W + matching_mask_for_workers + ranking_masks_for_workers[r]
            F = F + matching_mask_for_firms + ranking_masks_for_firms[r]
            W = W.clamp(min=0.0) * (1.0-c_f).unsqueeze(-1)[:self.n_workers]
            F = F.clamp(min=0.0) * (1.0-c_w).unsqueeze(-1)[:self.n_firms]
        del ranking_masks_for_workers, ranking_masks_for_firms
        assert not (matching.min() < 0.0 and matching.max() > 1.0)
        return matching

    def predict(self, W:Tensor, F:Tensor, R:Tensor) -> Tensor:
        """Matrix formulation of serial dictatorship. W, F, and R are assumed to be int tensors.
        Input:
            W: (n_workers, n_firms+1, n_firms+1) with W[i][j][k] = 1 meaning worker i places firm [j] at position k (one of j denotes unmatch)
            F: (n_firms, n_workers+1, n_workers+1) with F[i][j][k] = 1 meaning firm i places worker [j] at position k (one of j denotes unmatch)
            R: (n_workers + n_firms, n_workers + n_firms) of the permutation (= ranking) matrix over workers and firms
        """
        self.n_workers = W.shape[0]
        self.n_firms = F.shape[0]
        device = W.device

        # Line 1: create ranking masks
        R = R.float()
        ranking_masks_for_workers, ranking_masks_for_firms = self.create_ranking_masks(R)

        # Line 2: init matching result
        matching = torch.zeros(size=(self.n_workers+1, self.n_firms+1), device=device, dtype=torch.int8)
        for r in range(R.shape[1]):
            # Line 4: dictator at this rank r
            d_w, d_f = R[:self.n_workers, r], R[self.n_workers:, r]

            # Line 5: dictator's preference
            P_w = torch.einsum('wij,w->ij', W.float(), d_w)
            P_f = torch.einsum('fij,f->ij', F.float(), d_f)

            # Line 6,7: counterpart of the dictator
            c_w = self.find_counterpart(P_w).to(torch.int8)
            c_f = self.find_counterpart(P_f).to(torch.int8)

            # Line 8,9: update result
            M_w = torch.cat([d_w.to(torch.int8), torch.tensor([0], dtype=torch.int8)]).unsqueeze(1) @ c_w.T
            M_f = torch.cat([d_f.to(torch.int8), torch.tensor([0], dtype=torch.int8)]).unsqueeze(1) @ c_f.T
            assert matching.shape == M_w.shape == M_f.T.shape
            matching = matching + M_w + M_f.T

            # Line 10,11: compute matching mask
            matching_mask_for_workers = (-1) * (c_w * torch.tensor([[1]]*self.n_firms + [[0]], device=device)).repeat(1,self.n_firms+1)
            matching_mask_for_firms = (-1) * (c_f * torch.tensor([[1]]*self.n_workers + [[0]], device=device)).repeat(1,self.n_workers+1)

            # Line 12-16: apply mask
            assert W[0].shape == matching_mask_for_workers.shape == ranking_masks_for_workers[r].shape
            assert F[0].shape == matching_mask_for_firms.shape == ranking_masks_for_firms[r].shape
            W = W + matching_mask_for_workers + ranking_masks_for_workers[r]
            F = F + matching_mask_for_firms + ranking_masks_for_firms[r]
            W = W.clamp(min=0) * (1-c_f).unsqueeze(-1)[:self.n_workers]
            F = F.clamp(min=0) * (1-c_w).unsqueeze(-1)[:self.n_firms]
        del ranking_masks_for_workers, ranking_masks_for_firms
        assert not (matching.min() < 0 and matching.max() > 1)
        return matching

class MatchingLoss(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, input:Tensor, target:Tensor) -> Tensor:
        """Loss function over two matchings.

        Args:
            input (Tensor): Estimated matching.
            target (Tensor): Target matching.

        Returns:
            Tensor: Loss
        """
        pass

class RowCrossEntropyLoss(MatchingLoss):
    def __init__(self) -> None:
        super(RowCrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # row-wise cross entropy loss
        return self.loss(input[:-1], target[:-1]).unsqueeze(0)
