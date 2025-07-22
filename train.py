import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from layers import MatchingLoss
from tqdm import tqdm
from loss_functions import compute_st_batch
from utils import matrix_to_preference, preference_to_rational

def train(model: nn.Module, 
          criterion: MatchingLoss, 
          optimizer: Optimizer, 
          data_loader: DataLoader, 
          epochs: int = 5,
          lam:float=0.0,
          verbose:bool=False):
    model.train()
    for epoch in tqdm(range(epochs), desc='Epochs'):
        with tqdm(data_loader, total=len(data_loader), leave=False, desc='Batch') as t:
            for inputs, labels in t:
                optimizer.zero_grad()  # Zero the gradients

                # Forward pass
                outputs = model(inputs)
                if verbose:
                    print("[",end="")
                    for i, x in enumerate(outputs[0]):
                        print("[",end="")
                        for j, xx in enumerate(x):
                            if j < len(x) - 1:
                                print(f'{xx:.3f}', end=",")
                            else: 
                                print(f'{xx:.3f}', end="")
                        if i < len(outputs[0])-1:
                            print(f"]  {float(x.sum()):.3f}\n ", end="")
                        else:
                            print(f"]] {x.sum():.3f}")
                    print(torch.sum(outputs[0], dim=0).squeeze().detach().numpy().tolist())
                losses = [criterion(output, label) for (output, label) in zip(outputs, labels)]
                loss = torch.cat(losses).sum()

                loss += lam * compute_st_batch([
                        [
                            output[:-1, :-1],
                            preference_to_rational(inputs[i][0]),
                            preference_to_rational(inputs[i][1]).T
                        ]
                        for i, output in enumerate(outputs)]
                    )

                if verbose:
                    for name, p in model.named_parameters():
                        print(name)
                        print(p)
                # Backward and optimize
                loss.backward()
                nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1e1, norm_type=1)
                optimizer.step()
                if verbose: print(list(model.named_parameters()))
                
                # Update the postfix information on the tqdm bar
                t.set_postfix({'loss': loss.item()})
