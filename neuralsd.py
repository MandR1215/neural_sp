import torch
import torch.nn as nn
from torch import Tensor
from typing import List
from layers import SoftSort, TensorSerialDictatorship
from utils import rational_to_preference, preference_to_matrix, matrix_to_preference

class NeuralSD(nn.Module):
    def __init__(self, 
                 input_size:int,
                 embed_dim:int=10,
                 tau=1e-1, pow=2.0) -> None:
        super(NeuralSD, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1)

        self.ranker = nn.Sequential(
            nn.Linear(embed_dim, 1)
        )
        self.soft_sort = SoftSort(tau=tau, hard=False, pow=pow)
        self.matcher = TensorSerialDictatorship()

    def raw_ranking(self, X:Tensor) -> Tensor:
        query, key, value = X, X, X
        X, _ = self.attn(query=query, key=key, value=value)
        raw_ranking = self.ranker(X).transpose(0,1)
        raw_ranking = raw_ranking + torch.argsort(torch.argsort(raw_ranking)) # tie-break
        return raw_ranking
    
    def forward(self, batch:List[List[Tensor]]) -> List[Tensor]:
        """
        Args:
        - batch: Batch of (W:Tensor, F:Tensor, X:Tensor)
        """
        output = []

        for i, (W,F,X) in enumerate(batch):
            try:
                W = preference_to_matrix(W)
                F = preference_to_matrix(F)
                ranking = self.raw_ranking(X)
                R = self.soft_sort(ranking)[0]
                matching = self.matcher(W=W, F=F, R=R) # row is logit for the use of nn.CrossEntropyLoss 
                output.append(matching)
                del W,F
            except:
                import pdb; pdb.set_trace()
        return output
    
    def enforce_hard(self, how:str, t:float=None):
        assert how == 'scatter' or how == 'threshold' and 0.0 < t < 1.0
        self.soft_sort.hard = True
        self.soft_sort.t = t
        self.soft_sort.how = how

    def predict(self, batch:List[List[Tensor]], how:str, t:float=None) -> List[Tensor]:
        self.enforce_hard(how=how, t=t)
        output = []

        for i, (W,F,X) in enumerate(batch):
            try:
                W = preference_to_matrix(W).to(torch.int8)
                F = preference_to_matrix(F).to(torch.int8)
                ranking = self.raw_ranking(X)
                R = self.soft_sort(ranking)[0].to(torch.int8)
                matching = self.matcher.predict(W=W, F=F, R=R)
                output.append(matching)
                del W,F
            except:
                raise ValueError("Error in forward")
        self.soft_sort.hard = False
        self.soft_sort.hard = None
        self.soft_sort.how = None

        return output
    
    def predict_proba(self, batch:List[List[Tensor]], n_iter:int=10) -> List[Tensor]:
        """predictions by normalized matchings.

        Args:
            batch (List[List[Tensor]]): Batched data.
            n_iter (int, optional): Iterations of Sinkhorn normalization. Defaults to 10.

        Returns:
            List[Tensor]: Predicted matching matrices.
        """
        output = self.forward(batch)
        for i in range(len(output)):
            output[i] = self.normalize(output[i])
        
        return output

    def predict_ranking(self, batch:List[List[Tensor]], how:str, t:float=None) -> List[Tensor]:
        self.enforce_hard(how=how, t=t)
        R = []
        for i, (W,F,X) in enumerate(batch):
            try:
                R.append(self.soft_sort(self.raw_ranking(X))[0])
            except:
                raise ValueError("Error in predicting rankings")
        self.soft_sort.hard = False
        self.soft_sort.hard = None
        self.soft_sort.how = None

        return R
    
    def predict_on_rational_representation(self, p:Tensor, q:Tensor, px:Tensor, qx:Tensor, how:str, t:float) -> Tensor:
        W = preference_to_matrix(rational_to_preference(p))
        F = preference_to_matrix(rational_to_preference(q))
        X = torch.cat([px, qx])
        output = self.predict([[W,F,X]], how=how, t=t)

        return output
