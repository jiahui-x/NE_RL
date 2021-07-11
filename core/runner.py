import sys
sys.path.append("../rlcard/agents")
sys.path.append("../rlcard/utils")
from multitype_agent import LoosePassive,LooseAggressive,TightAggressive,TightPassive
import torch
# from rlcard.agents import RandomAgent
from utils import tournament
import rlcard

# Rollout evaluate an agent in a complete game
@torch.no_grad()
def rollout_worker(id, type, task_pipe, result_pipe, store_data, model_bucket, rollout_info):
    env = rlcard.make(rollout_info[1], config={'seed': None})
    fitness = 0.0
    total_frame = rollout_info[2]
    rollout_trajectory = []
    ###LOOP###
    while True:
        identifier = task_pipe.recv()  # Wait until a signal is received  to start rollout
        if identifier == 'TERMINATE': exit(0) #Exit

        # Get the requisite network
        agent = model_bucket[identifier]
        agents = [agent]
        opponent_list = [LoosePassive(num_actions=env.num_actions),LooseAggressive(num_actions=env.num_actions),TightAggressive(num_actions=env.num_actions),TightPassive(num_actions=env.num_actions)]
        for _ in range(env.num_players):
            agents.append(opponent_list[rollout_info[0]])
        env.set_agents(agents)

        # Generate data from the environment
        fitness, ts = tournament(env, rollout_info[2], type)

        # Reorganaize the data to be state, action, reward, next_state, done
        if store_data:
            rollout_trajectory = ts




        # Send back id, fitness, total length and shaped fitness using the result pipe
        result_pipe.send([identifier, fitness, total_frame, rollout_trajectory])
