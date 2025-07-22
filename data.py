from torch.utils.data import Dataset, DataLoader
import itertools
import numpy as np

class MatchingDataset(Dataset):
    def __init__(self, data):
        """
        data: List of data. Each element has a form of ((W, F, X), y).
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (W, F, X), y = self.data[idx]
        return (W, F, X), y
    
def collate_fn(inputs):
    return [input[0] for input in inputs], [input[1] for input in inputs]
    
def create_loader(dataset:MatchingDataset, *args, **kwargs):
    return DataLoader(dataset=dataset, collate_fn=collate_fn, *args, **kwargs)


def generate_permutation_array(N, num_agents):
    P = np.zeros((N, num_agents))
    for i in range(N): P[i] = np.random.permutation(num_agents)
    return P
