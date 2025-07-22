import numpy as np
from typing import List
import torch
from torch import Tensor

def hamming_distance(m1: np.array, m2: np.array) -> int:
    """Compute Hamming distance between two matching matrices.

    Args:
        m1 (np.array): First matching matrix
        m2 (np.array): Second matching matrix

    Returns:
        float: Hamming distance
    """
    assert m1.shape == m2.shape

    return int(np.sum(np.abs(m1 - m2)))

def number_of_blocking_pairs(
    worker_preferences: List[List[int]], 
    firm_preferences: List[List[int]],
    matching: np.array) -> int:
    """Count the number of blocking pairs given a matching."""
    matching = matching.astype(int)
    
    n, m = matching.shape[0]-1, matching.shape[1]-1
    count = 0

    # Get a current match of each worker against the firm
    worker_to_firm = np.argmax(matching, axis=1)
    # Get a current match of each firm against the worker
    firm_to_worker = np.argmax(matching, axis=0)

    for w in range(n):
        for f in range(m):
            # If worker w and firm f do not match
            if matching[w, f] == 0:
                # Check if worker w prefers firm f to current matching
                if worker_preferences[w].index(f) < worker_preferences[w].index(worker_to_firm[w]):
                    # Check if firm f prefers worker w to current matching
                    if firm_preferences[f].index(w) < firm_preferences[f].index(firm_to_worker[f]):
                        count += 1
    return count


def preference_to_matrix(p: List[List[int]], is_descending: bool=True) -> Tensor:
    """
    Converts a preference list to a permutation matrix tensor.
    
    Args:
    - p (List[List[int]]): A list of preference lists, where p[i] is the ith agent's preference list over n_counterparts+1 options.
    - is_descending (bool): If True, preferences are given in descending order of preference (highest to lowest). If False, in ascending order (lowest to highest). Defaults to True.
    
    Returns:
    - Tensor: A tensor of shape (n_agents, n_counterparts+1, n_counterparts+1) representing the permutation matrix of preferences. Tensor[i][j][k] indicates whether agent i ranks counterpart j at position k (1 if true, 0 otherwise).
    """
    n_agents = len(p)
    n_counterparts = len(p[0]) - 1  # Assuming all agents have the same number of preferences
    
    # Initialize the tensor with zeros
    pref_matrix = torch.zeros(n_agents, n_counterparts + 1, n_counterparts + 1)
    
    for i, pref_list in enumerate(p):
        for rank, j in enumerate(pref_list):
            # Convert to descending order if preferences are given in ascending order
            if is_descending:
                converted_rank = rank
            else:
                converted_rank = n_counterparts - rank

            # Update the preference matrix
            pref_matrix[i, j, converted_rank] = 1
    
    return pref_matrix


def matrix_to_preference(matrix, is_descending=True):
    """
    Converts a permutation matrix tensor back to a preference list.
    
    Args:
    - matrix (Tensor): A tensor of shape (n_agents, n_counterparts+1, n_counterparts+1).
    - is_descending (bool): If True, output preferences are in descending order of preference. If False, in ascending order.
    
    Returns:
    - List[List[int]]: A list of preference lists, where the inner list contains preferences of an agent.
    """
    n_agents, n_counterparts_plus_one, _ = matrix.shape
    preference_list = []

    for i in range(n_agents):
        agent_pref = []
        for j in range(n_counterparts_plus_one):
            rank = torch.argmax(matrix[i, :, j]).item()
            agent_pref.append(rank)
        preference_list.append(agent_pref if is_descending else list(reversed(agent_pref)))

    return preference_list

def preference_to_rational(p: List[List[int]]) -> Tensor:
    """Representation of preferences.

    Args:
    - p (List[List[int]]): A list of preference lists, where p[i] is the ith agent's preference list over n_counterparts+1 options. 
                           The id for unmatch is considered to be equal to n_counterparts.

    Returns:
    - Tensor: Representation of preference orderes by rational numbers.
    """
    n_agents = len(p)
    n_counterparts = len(p[0]) - 1
    unmatch_id = n_counterparts

    rationals = [[0]*n_counterparts for _ in range(n_agents)]

    for agent in range(n_agents):
        unmatch_index = p[agent].index(unmatch_id)
        for j, counterpart in enumerate(p[agent]):
            if counterpart == unmatch_id:
                pass
            else:
                rationals[agent][counterpart] = (unmatch_index - j) / n_counterparts
    
    return torch.tensor(rationals)

def rational_to_preference(rationals:Tensor) -> List[List[int]]:
    unmatch_id = len(rationals[0])
    v, p = torch.sort(-torch.tensor(rationals))
    p = p.tolist()
    for agent in range(len(p)):
        if v[agent][0] > 0:
            p[agent].insert(0, unmatch_id)
        elif v[agent][-1] < 0:
            p[agent].insert(len(v[agent]), unmatch_id)
        else:
            for left in range(len(v[agent])):
                if v[agent][left] < 0 and v[agent][left+1] > 0:
                    p[agent].insert(left+1, unmatch_id)
                    break
    return p
