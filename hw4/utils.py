from agents.Agent import Agent
import gymnasium as gym
import numpy as np
from tqdm import tqdm

def environment_info(env: gym.wrappers) -> None:
    ''' Print basic information on the requested environment. '''
    print(f'Observation space: {env.observation_space}')
    print(f'Action space: {env.action_space}')

def policy_rollout(env: gym.wrappers, agent : Agent, visualize : bool=False) -> None:
    ''' Rollout a particular policy in the cartpole environment. '''
    rewards = []
    for _ in tqdm(range(10)):  # Observe policy in 10 independent rollouts for better visualization
        episode_reward = 0
        obs, info = env.reset()
        terminated, truncated = False, False

        while not terminated and not truncated:
            if visualize:
                env.render() # prints a visualization of the environment 

            action = agent.policy(obs)
            obs, reward, terminated, truncated, info = env.step(action.item())
            episode_reward += reward

        rewards.append(episode_reward)
    
    print("--- Reward Statistics ---")
    print(f'Average: {np.mean(rewards)}')
    print(f'Standard deviation: {np.std(rewards)}')
    print(f'Minimum: {np.min(rewards)}')
    print(f'Maximum: {np.max(rewards)}')