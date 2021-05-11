from model import dynamics, cost
import numpy as np


stochastic_dynamics = False # set to True for stochastic dynamics
dynfun = dynamics(stochastic=stochastic_dynamics)
costfun = cost()

T = 100 # episode length
N = 100 # number of episodes
gamma = 0.95 # discount factor

total_costs = []

for n in range(N):
    costs = []
    
    x = dynfun.reset()
    for t in range(T):
        

        # TODO compute policy
        
        # TODO compute action
        
        # get reward
        c = costfun.evaluate(x,u)
        costs.append((gamma**t)*c)
        
        # dynamics step
        xp = dynfun.step(u)
        
        # TODO implement recursive least squares update


        x = xp.copy()
        
    total_costs.append(sum(costs))