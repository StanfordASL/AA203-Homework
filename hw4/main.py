import gymnasium as gym
from utils import policy_rollout, environment_info
from agents.basic import Basic
from agents.qlearning import QLearning
from agents.ReinforceAgent import ReinforceAgent
import argparse, torch, datetime
    
def main(args: argparse.Namespace) -> None:
    env = gym.make(args.env_name, render_mode="human")

    if args.agent_name == "basic":
        agent = Basic(state_dim=args.state_dim, action_dim=args.action_dim)
    elif args.agent_name == "reinforce":
        agent = ReinforceAgent(state_dim=args.state_dim, action_dim=args.action_dim)
    elif args.agent_name == "qlearning":
        agent = QLearning(state_dim=args.state_dim, action_dim=args.action_dim, use_gpu=args.use_gpu)
    else:
        raise ValueError(f"Unsupported agent type: {args.agent}")

    environment_info(env)  # characterize the environment

    if args.mode == "train":
        agent.train(env)
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(agent.policy_network.state_dict(), f"{agent.agent_name}_agent_policy_{current_time}.pth")  # Save the policy network
    
    elif args.mode == "test" and args.checkpoint_path:
        agent.policy_network.load_state_dict(torch.load(args.checkpoint_path))
        print(f'Loaded policy network from {args.checkpoint_path}')
        
        policy_rollout(env, agent, visualize=args.visualization)
    
    elif args.mode == "test" and not args.checkpoint_path: # for testing the base policy before training
        policy_rollout(env, agent, visualize=args.visualization)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="CartPole-v1", help="the openai gym environment in which we will instantiate the RL agent.")
    parser.add_argument("--agent-name", type=str, default="qlearning", help="the policy you wish to instantiate e.g., 'basic', 'reinforce'.")
    parser.add_argument("--state-dim", type=int, required=True, help="the dimension of the cartpole state space.")
    parser.add_argument("--action-dim", type=int, required=True, help="the dimension of the cartpole action space.")
    parser.add_argument("--use-gpu", action='store_true', help="whether to run training on your system's gpu, if available, (a gpu is not necessary to complete this assignment!).")
    parser.add_argument("--mode", type=str, choices=['train', 'test'], default="test", help="mode to run: 'train' or 'test'")
    parser.add_argument("--visualization", action='store_true', help="whether to visualize the cartpole environment during the rollout.")
    parser.add_argument("--checkpoint-path", type=str, help="path to the checkpoint of a trained policy network (for test mode)")
    
    main(parser.parse_args())