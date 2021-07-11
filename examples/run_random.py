''' An example of playing randomly in RLCard
'''
import argparse
import pprint

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import set_seed, reorganize

def run(args):
    # Make environment
    env = rlcard.make(args.env, config={'seed': 42})

    # Seed numpy, torch, random
    set_seed(42)

    # Set agents
    agent = RandomAgent(num_actions=env.num_actions)
    env.set_agents([agent for _ in range(env.num_players)])
    n=0
    # Generate data from the environment
    while(n<10):
        trajectories, player_wins = env.run(is_training=False)
        # Print out the trajectories
        # print('\nTrajectories:')
        # print(trajectories)
        print('\nSample raw observation:')
        pprint.pprint(trajectories[0][-1]['raw_obs'])
        print('\nSample raw legal_actions:')
        pprint.pprint(trajectories[0][-1]['raw_legal_actions'])
        # trajectories = reorganize(trajectories, player_wins)
        # print(trajectories)
        n+=1

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Random example in RLCard")
    parser.add_argument('--env', type=str, default='no-limit-holdem',
            choices=['blackjack', 'leduc-holdem', 'limit-holdem', 'doudizhu', 'mahjong', 'no-limit-holdem', 'uno', 'gin-rummy'])

    args = parser.parse_args()

    run(args)

