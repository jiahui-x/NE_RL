''' An example of training a reinforcement learning agent on the environments in RLCard
'''
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch


from algos.erl_trainer import ERL_Trainer



if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN/NFSP example in RLCard")
    parser.add_argument('--env', type=str, default='no-limit-holdem',
            choices=['blackjack', 'leduc-holdem', 'limit-holdem', 'doudizhu', 'mahjong', 'no-limit-holdem', 'uno', 'gin-rummy'])
    parser.add_argument('--algorithm', type=str, default='dqn', choices=['dqn', 'nfsp'])
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_eval_games', type=int, default=500)
    parser.add_argument('--evaluate_every_gen', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default='experiments/vs_multitpye/HUNL_ERL_result/')

    #ERL_setting_append
    parser.add_argument('--pop_size', type=int, help='#Policies in the population',  default=8)
    parser.add_argument('--rollout_size', type=int, help='#Policies in rollout size',  default=0)
    parser.add_argument('--num_test', type=int, help='#Test envs to average on', default=0)
    parser.add_argument('--elite_fraction', type=float, help='#fraction of elitism', default=0.2)
    parser.add_argument('--crossover_prob', type=float, help='#fraction of elitism', default=0.15)
    parser.add_argument('--mutation_prob', type=float, help='#fraction of elitism', default=0.90)
    parser.add_argument('--extinction_prob', type=float, help='#fraction of elitism', default=0.005)
    parser.add_argument('--extinction_magnituide', type=float, help='#fraction of elitism', default=0.5)
    parser.add_argument('--weight_magnitude_limit', type=float, help='#fraction of elitism', default=10000000)
    parser.add_argument('--mut_distribution', type=float, help='#fraction of elitism', default=1)
    parser.add_argument('--max_gen', type=int, help='#max training generation', default=40)
    parser.add_argument('--writer', default=SummaryWriter(log_dir='experiments/vs_multitpye/HUNL_ERL_result/tensorboard/'))
    parser.add_argument('--type_choice', type=int, help='#Opponent choice', default=0)



    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    ai = ERL_Trainer(args)
    ai.train()

