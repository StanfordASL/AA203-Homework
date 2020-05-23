import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
# Cart Pole
parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=203, metavar='N',
                    help='random seed (default: 203)')
parser.add_argument('--render', default=False,
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')

# based on:
# https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py

args = parser.parse_args()
env = gym.make('LunarLanderContinuous-v2')

env.seed(args.seed)
torch.manual_seed(args.seed)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(state_dim, 128)
        # actor's layer
        self.action_mean = nn.Linear(128, action_dim)
        self.action_var = nn.Linear(128, action_dim)
        # critic's layer
        self.value_head = nn.Linear(128, 1)
        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        
    def forward(self, x):
        """
        forward of both actor and critic
        """
        # TODO map input to:
        # mean of action distribution,
        # variance of action distribution (pass this through a non-negative function),
        # state value
        
        return 0.5*action_mean, 0.5*action_var, state_values
    
model = Policy().float()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    state = torch.from_numpy(state).float()
    mu, sigma, state_value = model(state)
    
    # create a normal distribution over the continuous action space
    m = Normal(loc=mu,scale=sigma)
    
    # and sample an action using the distribution
    action = m.sample()
    
    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    
    # the action to take (left or right)
    return action.data.numpy()

def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values
    
    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # TODO compute the value at state x
        # via the reward and the discounted tail reward

        
        
        returns.insert(0, R)
        
    # whiten the returns
    returns = torch.tensor(returns).float()
    returns = (returns - returns.mean()) / (returns.std() + eps)
    
    for (log_prob, value), R in zip(saved_actions, returns):
        # TODO compute the advantage via subtracting off value
        
        
        # TODO calculate actor (policy) loss, from log_prob (saved in select action)
        # and from advantage
        
        # append this to policy_losses
        
        # TODO calculate critic (value) loss
        
    # reset gradients
    optimizer.zero_grad()
    
    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    
    # perform backprop
    loss.backward()
    optimizer.step()
    
    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]
    
def main():
    running_reward = -100
    
    # run infinitely many episodes, until performance criteria met
    episodic_rewards = []
    
    for i_episode in count(1):
        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0

        for t in range(1, 2500):
            # select action from policy
            action = select_action(state)
            
            # take the action
            state, reward, done, _ = env.step(action)
            
            if args.render and i_episode % 100 == 0:
                env.render()
    
            model.rewards.append(reward)
            ep_reward += reward
            if done:
                episodic_rewards.append(ep_reward)
                break
                
        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        
        # perform backprop
        finish_episode()
        
        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
            
        # check if we have "solved" the problem
        if running_reward > 200:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))

            # TODO plot episodic_rewards --- submit this plot with your code
            
            break
            
if __name__ == '__main__':
    main()