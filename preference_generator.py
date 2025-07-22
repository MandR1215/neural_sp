import itertools
import numpy as np
from typing import List, Tuple
from abc import ABCMeta, abstractmethod

class PreferenceGenerator(metaclass=ABCMeta):
    def __init__(self, n_agents:int, n_counterparts:int) -> None:
        self.n_agents = n_agents
        self.n_counterparts = n_counterparts
        self.preferences = []
        self.initialize()

    @abstractmethod
    def initialize(self) -> None:
        raise NotImplementedError
    
    def get(self, agent_id:int) -> List[int]:
        return self.preferences[agent_id]
    
    def set(self, agent_id:int, preference:List[int]) -> None:
        self.preferences[agent_id] = preference
    
    def get_all(self) -> List[List[int]]:
        return self.preferences
    
    def get_misreports(self, agent_id:int) -> List[List[int]]:
        misreports = []
        for report in itertools.permutations(list(range(self.n_counterparts))):
            if list(report) != list(self.preferences[agent_id]):
                misreports.append(list(report))

        return misreports
    
    def get_misreports_all(self) -> List[List[List[int]]]:
        return [self.get_misreports for agent in range(self.n_agents)]
    
class AttributePreferenceGenerator(PreferenceGenerator):
    def __init__(self, n_agents: int, n_counterparts: int, attributes: None) -> None:
        self.n_agents = n_agents
        self.n_counterparts = n_counterparts
        self.attributes = attributes
        self.preferences = []
        self.validate_attributes(attributes)
        self.initialize(attributes)

    @abstractmethod
    def validate_attributes(self, attributes) -> None:
        raise NotImplementedError

    @abstractmethod
    def initialize(self, attributes) -> None:
        raise NotImplementedError
    
class UniformRandom(PreferenceGenerator):
    def __init__(self, n_agents: int, n_counterparts: int) -> None:
        super().__init__(n_agents, n_counterparts)

    def initialize(self) -> None:
        for _ in range(self.n_agents):
            self.preferences.append(list(np.random.permutation(self.n_counterparts)))

class Euclidean(AttributePreferenceGenerator):
    def __init__(self, n_agents: int, n_counterparts: int, attributes: None, unmatch_threshold:float = None) -> None:
        self.unmatch_threshold = unmatch_threshold
        super().__init__(n_agents, n_counterparts, attributes)

    def validate_attributes(self, attributes) -> None:
        assert isinstance(attributes, tuple)
        assert len(attributes) == 2
        assert attributes[0].ndim == attributes[1].ndim == 2 # n * d
        assert attributes[0].shape[1] == attributes[1].shape[1]

    def initialize(self, attributes) -> None:
        att_agent, att_counterpart = attributes[0], attributes[1]
        diff = att_agent[:,None,:] - att_counterpart[None,:,:]
        dist = None
        if att_agent.shape[1] == 1: # dimension of attribute is 1
            dist = np.abs(diff)
        else:
            dist = diff ** 2
        
        dist = np.sqrt(dist.sum(axis=2))
        if self.unmatch_threshold is not None:
            dist = np.insert(dist, self.n_counterparts, self.unmatch_threshold, axis=1)
            assert dist.shape == (self.n_agents, self.n_counterparts+1)
            self.preferences = np.argsort(dist).tolist()
        else:
            self.preferences = np.argsort(dist).tolist()
            for agent in range(self.n_agents):
                self.preferences[agent].insert(np.random.randint(low=0, high=self.n_counterparts+1), self.n_counterparts)
