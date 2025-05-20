from .Agent import Agent
import numpy as np
import torch

class Basic(Agent):
    def __init__(self, state_dim : int, action_dim : int, hidden_dim : int=24) -> None:
        super().__init__(state_dim, action_dim, hidden_dim)
        self.agent_name = "basic"

    def policy(self, state : np.ndarray, train : bool=False) -> torch.Tensor: # basic strategy: move cart in the direction that the pole is leaning
        _, _, theta, _ = state # extract pole angle from environment
        left, right = torch.tensor([0]), torch.tensor([1])

        return left if theta < 0 else right
    