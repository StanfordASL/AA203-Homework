from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import torch

class Transition():
    def __init__(self, s, a, r, sp) -> None: # (s, a, r, s') tuple from environment
        self.state = s
        self.action = a
        self.reward = r
        self.next_state = sp

class Agent(ABC):
    def __init__(self, state_dim : Optional[int], action_dim : Optional[int], hidden_dim : Optional[int], use_gpu : bool=False) -> None:
        self.state_dim, self.action_dim, self.hidden_dim = state_dim, action_dim, hidden_dim # defines width of downstream policy network
        
        self.gamma = 0.99 # discount factor
        self.learning_rate = 0.001
        self.batch_size = 128
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def policy(self, state : np.ndarray, train : bool=False) -> torch.Tensor:
        pass