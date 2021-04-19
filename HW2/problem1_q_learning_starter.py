from problem1_q_learning_env import *
import numpy as np
sim = simulator()

T = 5*365 # simulation duration
gamma = 0.95 # discount factor


# get historical data
data = generate_historical_data(sim)
# historical dataset: 
# shape is 3*365 x 4
# k'th row contains (x_k, u_k, r_k, x_{k+1})

# TODO: write Q-learning to yield Q values,
# use Q values in policy (below)

def policy(state,Q):
    # TODO fill in 
    
# Forward simulating the system 
s = sim.reset()
for t in range(T):
    a = policy(s,Q)
    sp,r = sim.step(a)
    s = sp
    # TODO add logging of rewards for plotting

# TODO: write value iteration to compute true Q values
# use functions:
# - sim.transition (dynamics)
# - sim.get_reward 
# plus sim.demand_probs for the probabilities associated with each demand value
