from model import dynamics, cost
import numpy as np


stochastic_dynamics = False # set to True for stochastic dynamics
dynfun = dynamics(stochastic=stochastic_dynamics)
costfun = cost()

T = 100
N = 10000 
gamma = 0.95 # discount factor

total_costs = []

for n in range(N):
    costs = []
    
    x = dynfun.reset()
    for t in range(T):

        # TODO compute action

        # get reward
        c = costfun.evaluate(x,u)
        costs.append((gamma**t)*c)
        
        # dynamics step
        xp = dynfun.step(u)


        x = xp.copy()

    # TODO update policy



    total_costs.append(sum(costs))