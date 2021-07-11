''' An example of training a reinforcement learning agent on the environments in RLCard
'''
import os
import argparse

import torch

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import get_device, tournament_rl, set_seed, reorganize, Logger
from rlcard.agents.multitype_agent import LoosePassive

import sys
sys.path.append('../rlcard/agents')
from lstm_dqn_agent import DQNAgent
from copy import deepcopy

def train(args):

    # Check whether gpu is available
    # device = get_device()
    device = torch.device("cpu")
    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(args.env, config={'seed': args.seed})

    # Initialize the agent and use random agents as opponents
    if args.algorithm == 'dqn':
        # from rlcard.agents import DQNAgent
        agent = DQNAgent(num_actions=env.num_actions,
                         state_shape=env.state_shape[0],
                         mlp_layers=[512, 512],
                         device=device)
    elif args.algorithm == 'nfsp':
        from rlcard.agents import NFSPAgent
        agent = NFSPAgent(num_actions=env.num_actions,
                          state_shape=env.state_shape[0],
                          hidden_layers_sizes=[64,64],
                          q_mlp_layers=[64,64],
                          device=device)
    agents = [agent]
    for _ in range(env.num_players - 1):
        agents.append(RandomAgent(num_actions=env.num_actions))
    env.set_agents(agents)

    # Start training

    for episode in range(args.num_episodes):

        # Evaluate the performance. Play with random agents.
        # if episode % args.evaluate_every == 0:
            # env1 = rlcard.make(args.env, config={'seed': None})
            # env1.set_agents([agent1,RandomAgent(num_actions=env1.num_actions)])
            # env1.set_agents(agents)
            # reward = tournament_rl(env, args.num_eval_games)[0]
        # if args.algorithm == 'nfsp':
        #     agents[0].sample_episode_policy()
        #
        # # Generate data from the environment
        trajectories, payoffs = env.run(is_training=True)

        # Reorganaize the data to be state, action, reward, next_state, done
        trajectories = reorganize(trajectories, payoffs)

        # Feed transitions into agent memory, and train the agent
        # Here, we assume that DQN always plays the first position
        # and the other players play randomly (if any)
        for ts in trajectories[0]:
            agent.feed(ts)
        print(episode)




        # Plot the learning curve
        # logger.plot(args.algorithm)

    # Save model
    # save_path = os.path.join(args.log_dir, 'model.pth')
    # torch.save(agent, save_path)
    # print('Model saved in', save_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN/NFSP example in RLCard")
    parser.add_argument('--env', type=str, default='no-limit-holdem',
            choices=['blackjack', 'leduc-holdem', 'limit-holdem', 'doudizhu', 'mahjong', 'no-limit-holdem', 'uno', 'gin-rummy'])
    parser.add_argument('--algorithm', type=str, default='dqn', choices=['dqn', 'nfsp'])
    parser.add_argument('--cuda', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_episodes', type=int, default=40000)
    parser.add_argument('--num_eval_games', type=int, default=500)
    parser.add_argument('--evaluate_every', type=int, default=100)
    parser.add_argument('--log_dir', type=str, default='experiments/vs_multitpye/HUNL_dqn_result')

    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    #
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    train(args)

