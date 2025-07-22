from abc import ABCMeta, abstractmethod
import numpy as np
from typing import List, Callable, Any
from itertools import permutations
from collections import deque
from collections import OrderedDict
import gurobipy as gp
from gurobipy import GRB

class MatchingAlgorithm(metaclass=ABCMeta):
    def __init__(self, n_workers:int, n_firms:int) -> None:
        self.n_workers = n_workers
        self.n_firms = n_firms

    @abstractmethod
    def match(self, worker_preferences:List[List[int]], firm_preferences:List[List[int]]) -> np.ndarray:
        """Matching Algorithm.

        Args:
            worker_preferences (List[List[int]]): descending order of workers' preferences, permutation of [0,..., n_firms] with n_firms denoting unmatch.
            firm_preferences (List[List[int]]): descending order of fims' preferences, permutation of [0,..., n_workers] with n_workers denoting unmatch.

        Returns:
            np.ndarray: (n_workers+1, n_firms+1) matching result. The entry at (w,f) means matched (1) or unmatched(0).
        """
        raise NotImplementedError()
    
class RandomSerialDictatorship(MatchingAlgorithm):
    def __init__(self, n_workers: int, n_firms: int) -> None:
        super().__init__(n_workers, n_firms)
        self.rankings = None
        self.n_rankings = None
        self.init_rankings()

    @abstractmethod
    def init_rankings(self) -> None:
        raise NotImplementedError()
    
    def match(self, worker_preferences: List[List[int]], firm_preferences: List[List[int]]) -> np.ndarray:
        matching = np.zeros((self.n_workers+1, self.n_firms+1))
        for ranking in self.rankings:
            sd = SerialDictatorship(n_workers=self.n_workers, n_firms=self.n_firms, ranking=ranking)
            matching += sd.match(worker_preferences=worker_preferences, firm_preferences=firm_preferences)
        matching = matching / self.n_rankings

        return matching
    
class SerialDictatorship(MatchingAlgorithm):
    def __init__(self, n_workers:int, n_firms:int, ranking:List[int]) -> None:
        '''
        Input:
            n_workers: number of workers
            n_firms: number of firms
            ranking: ranking over workers and firms
                     ranking[i] (0 <= i < n_workers) is the rank of i-th worker
                     ranking[i] (n_workers <= i < n_workers+n_firms) is the rank of i-th firm
        '''
        super().__init__(n_workers=n_workers, n_firms=n_firms)
        self.agents = list(range(n_workers+n_firms))
        self.ranking = ranking
        assert len(self.ranking) == len(self.agents)

    def match(self, worker_preferences: List[List[int]], firm_preferences: List[List[int]]) -> np.ndarray:
        ranked_agents = [self.agents[rank] for rank in self.ranking]
        matched_workers = [False]*self.n_workers
        matched_firms = [False]*self.n_firms

        matching = np.zeros((self.n_workers+1, self.n_firms+1))

        for agent in ranked_agents:
            if 0 <= agent < self.n_workers: # agent is a worker
                worker = agent
                # ignore if already matched 
                if matched_workers[worker]:
                    pass
                else: 
                    # walk over preference
                    for firm in worker_preferences[worker]:
                        # print(f"{worker=} ==> {firm=}")
                        if firm == self.n_firms: # no match is always possible
                            matching[worker,firm] = 1
                            matched_workers[worker] = True
                            break
                        else:
                            if not matched_firms[firm]:
                                # match
                                matching[worker,firm] = 1
                                matched_workers[worker] = True
                                matched_firms[firm] = True
                                break

            else: # agent is a firm
                firm = agent - self.n_workers
                # ignore if already matched 
                if matched_firms[firm]:
                    pass
                else: 
                    # walk over preference
                    for worker in firm_preferences[firm]:
                        # print(f"{firm=} ==> {worker=}")
                        if worker == self.n_workers: # no match is always possible
                            matching[worker,firm] = 1
                            matched_firms[firm] = True
                            break
                        else:
                            if not matched_workers[worker]:
                                # match
                                matching[worker,firm] = 1
                                matched_workers[worker] = True
                                matched_firms[firm] = True
                                break

        return matching
    
class UniformRandomSerialDictatorship(RandomSerialDictatorship):
    def __init__(self, n_workers: int, n_firms: int) -> None:
        super().__init__(n_workers, n_firms)

    def init_rankings(self) -> None:
        self.rankings = list(permutations(range(self.n_workers+self.n_firms)))
        self.n_rankings = len(self.rankings)

class DeferredAcceptance(MatchingAlgorithm):
    def __init__(self, n_workers: int, n_firms: int, first:bool=True) -> None:
        super().__init__(n_workers, n_firms)
        self.first = first

    def match(self, worker_preferences: List[List[int]], firm_preferences: List[List[int]]) -> np.ndarray:
        if not self.first:
            # firm proposal
            self.n_workers, self.n_firms = self.n_firms, self.n_workers
            worker_preferences, firm_preferences = firm_preferences, worker_preferences
        
        # Initialize matchings matrix with zeros (unmatched)
        matching = np.zeros((self.n_workers+1, self.n_firms+1), dtype=float)
        
        # Track the current proposals of each worker (initialize to the first preference)
        worker_proposals = [0] * self.n_workers
        
        # Keep track of which workers are still seeking a match
        unmatched_workers = deque(range(self.n_workers))
        
        # Track the current match of firms (initialize to be unmatched)
        unmatched_firms = [True] * self.n_firms
        firm_matchings = [self.n_workers] * self.n_firms

        while unmatched_workers:
            # New round of proposals
            worker = unmatched_workers.popleft()
            
            if worker_proposals[worker] < self.n_firms:
                # There is a firm to apply to
                firm = worker_preferences[worker][worker_proposals[worker]]

                # If this worker prefers to be unmatched
                if firm == self.n_firms:
                    matching[worker, self.n_firms] = 1  # Mark as unmatched
                else:
                    if unmatched_firms[firm]:
                        # Firm is unmatched, make the match
                        matching[worker, firm] = 1
                        unmatched_firms[firm] = False
                        firm_matchings[firm] = worker
                    else:
                        current_match = firm_matchings[firm]
                        # If the firm prefers this worker over the current match, update the matching
                        if firm_preferences[firm].index(worker) < firm_preferences[firm].index(current_match):
                            matching[current_match, firm] = 0
                            matching[worker, firm] = 1
                            firm_matchings[firm] = worker
                            unmatched_workers.append(current_match)
                        else:
                            unmatched_workers.append(worker)
                worker_proposals[worker] += 1
            else:
                # No more firms to apply to 
                matching[worker, self.n_firms] = 1  # Mark as unmatched
        
        # Ensure that firms' unmatched column is correctly set
        for firm in range(self.n_firms):
            if matching[:, firm].sum() == 0:
                matching[self.n_workers, firm] = 1  # Mark as unmatched
                
        return matching


class RankHungarian(MatchingAlgorithm):
    def __init__(self, n_workers: int, n_firms: int, worker_weights: np.ndarray, firm_weights: np.ndarray) -> None:
        assert len(worker_weights) == n_workers
        assert len(firm_weights) == n_firms
        super().__init__(n_workers, n_firms)
        self.worker_weights = np.append(worker_weights, 1.0)
        self.firm_weights = np.append(firm_weights, 1.0)
        
        self.model = gp.Model("assignment")
        self.model.setParam('OutputFlag', 0)
        self.M = self.model.addVars(n_workers+1, n_firms+1, vtype=GRB.BINARY, name="M")
        self.model.addConstrs((gp.quicksum(self.M[i,j] for j in range(n_firms+1)) == 1 for i in range(n_workers)), "worker sum-to-1")
        self.model.addConstrs((gp.quicksum(self.M[i,j] for i in range(n_workers+1)) == 1 for j in range(n_firms)), "firm sum-to-1")
        self.model.addConstr(self.M[n_workers, n_firms] == 0, "do not match bot with bot")

    def __compute_rank(self, l:List[int]) -> List[int]:
        n = len(l)
        ranks = [0] * n
        for i, value in enumerate(l):
            ranks[value] = n - i
        return ranks
        
    def match(self, worker_preferences: List[List[int]], firm_preferences: List[List[int]]) -> np.ndarray:
        worker_ranks = [self.__compute_rank(preference) for preference in worker_preferences] + [[0] * (self.n_firms+1)]
        firm_ranks = [self.__compute_rank(preference) for preference in firm_preferences] + [[0] * (self.n_workers+1)]

        reward_matrix = (self.worker_weights.reshape(-1,1) * np.asarray(worker_ranks)) + (self.firm_weights.reshape(-1,1) * np.asarray(firm_ranks)).T

        self.model.setObjective(gp.quicksum(self.M[i,j] * reward_matrix[i,j] for i in range(self.n_workers+1) for j in range(self.n_firms+1)), GRB.MAXIMIZE)
        self.model.optimize()
        
        matching = np.zeros((self.n_workers+1, self.n_firms+1), dtype=float)
        for i in range(self.n_workers+1):
            for j in range(self.n_firms+1):
                matching[i, j] = float(self.M[i, j].X)
        
        return matching
        
    def compute_reward(self, worker_preferences: List[List[int]], firm_preferences: List[List[int]], matching: np.array):
        worker_ranks = [self.__compute_rank(preference) for preference in worker_preferences] + [[0] * (self.n_firms+1)]
        firm_ranks = [self.__compute_rank(preference) for preference in firm_preferences] + [[0] * (self.n_workers+1)]

        reward_matrix = (self.worker_weights.reshape(-1,1) * np.asarray(worker_ranks)) + (self.firm_weights.reshape(-1,1) * np.asarray(firm_ranks)).T
        
        return (reward_matrix * matching).sum()
    
class EqualWeightedHungarian(RankHungarian):
    def __init__(self, n_workers: int, n_firms: int) -> None:
        worker_weights = np.ones(n_workers)
        firm_weights = np.ones(n_firms)
        super().__init__(n_workers, n_firms, worker_weights, firm_weights)

class MinorityHungarian(RankHungarian):
    def __init__(self, n_workers: int, n_firms: int) -> None:
        worker_weights = np.ones(n_workers)
        worker_weights[np.random.choice(n_workers, size=n_workers//3, replace=False)] = 2.0
        firm_weights = np.ones(n_firms)
        super().__init__(n_workers, n_firms, worker_weights, firm_weights)
