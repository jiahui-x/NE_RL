
import numpy as np, os, time, random, torch, sys
from algos.neuroevolution import SSNE
from core import utils
from core.runner import rollout_worker
from torch.multiprocessing import Process, Pipe, Manager

import torch
import rlcard
sys.path.append('../rlcard/agents')
from lstm_dqn_agent import DQNAgent
from lstm_dqn_agent import Estimator
import copy



class ERL_Trainer:

	def __init__(self, args):

		self.args = args
		env = rlcard.make(args.env, config={'seed': args.seed})
		rollout_info = [args.type_choice, args.env, args.num_eval_games]
		self.manager = Manager()
		self.device = torch.device("cpu")

		#Evolution
		self.evolver = SSNE(self.args)

		# PG Learner
		self.learner = DQNAgent(num_actions=env.num_actions,
								state_shape=env.state_shape[0],
								mlp_layers=[512, 512],
								device=self.device)

		#Initialize population
		self.population = self.manager.list()
		for _ in range(args.pop_size):
			self.population.append(Estimator(num_actions=env.num_actions, state_shape=env.state_shape[0], \
            mlp_layers=[512, 512], device=torch.device("cpu")))

		#Save best policy
		# self.best_policy = Estimator(num_actions=env.num_actions, state_shape=env.state_shape[0], \
        #     mlp_layers=[512, 512], device=torch.device("cpu"))



		#Replay Buffer
		# self.replay_buffer = Buffer(args.buffer)

		#Initialize Rollout Bucket
		self.rollout_bucket = self.manager.list()
		for _ in range(args.rollout_size):
		# self.rollout_bucket.append(DQNAgent(num_actions=env.num_actions,
		# 						state_shape=env.state_shape[0],
		# 						mlp_layers=[512, 512],
		# 						device=self.device))
			self.rollout_bucket.append(Estimator(num_actions=env.num_actions, state_shape=env.state_shape[0], \
            mlp_layers=[512, 512], device=torch.device("cpu")))

		############## MULTIPROCESSING TOOLS ###################
		#Evolutionary population Rollout workers
		self.evo_task_pipes = [Pipe() for _ in range(args.pop_size)]
		self.evo_result_pipes = [Pipe() for _ in range(args.pop_size)]
		self.evo_workers = [Process(target=rollout_worker, args=(id, 'evo', self.evo_task_pipes[id][1], self.evo_result_pipes[id][0], args.rollout_size > 0, self.population, rollout_info)) for id in range(args.pop_size)]
		for worker in self.evo_workers: worker.start()
		self.evo_flag = [True for _ in range(args.pop_size)]

		#Learner rollout workers
		self.task_pipes = [Pipe() for _ in range(args.rollout_size)]
		self.result_pipes = [Pipe() for _ in range(args.rollout_size)]
		self.workers = [Process(target=rollout_worker, args=(id, 'pg', self.task_pipes[id][1], self.result_pipes[id][0], True, self.rollout_bucket, rollout_info)) for id in range(args.rollout_size)]
		for worker in self.workers: worker.start()
		self.roll_flag = [True for _ in range(args.rollout_size)]

		#Test bucket
		self.test_bucket = self.manager.list()
		self.test_bucket.append(Estimator(num_actions=env.num_actions, state_shape=env.state_shape[0], \
            mlp_layers=[512, 512], device=torch.device("cpu")))
		# self.test_bucket.append(DQNAgent(num_actions=env.num_actions,
		# 						state_shape=env.state_shape[0],
		# 						mlp_layers=[512, 512],
		# 						device=self.device))

		# Test workers
		self.test_task_pipes = [Pipe() for _ in range(args.num_test)]
		self.test_result_pipes = [Pipe() for _ in range(args.num_test)]
		self.test_workers = [Process(target=rollout_worker, args=(id, 'test', self.test_task_pipes[id][1], self.test_result_pipes[id][0], False, self.test_bucket, rollout_info)) for id in range(args.num_test)]
		for worker in self.test_workers: worker.start()
		self.test_flag = False

		#Trackers
		self.best_score = -float('inf'); self.gen_frames = 0; self.total_frames = 0; self.test_score = None; self.test_std = None


	def forward_generation(self, gen, tracker):

		gen_max = -float('inf')
		total_ts = []

		#Start Evolution rollouts
		if self.args.pop_size > 1:
			for id, actor in enumerate(self.population):
				self.evo_task_pipes[id][0].send(id)

		#Sync all learners actor to cpu (rollout) actor and start their rollout
		# self.learner.q_estimator.cpu()
		for rollout_id in range(self.args.rollout_size):
			# utils.hard_update(self.rollout_bucket[rollout_id], self.learner)
			self.rollout_bucket[rollout_id] = copy.copy(self.learner.q_estimator)
			self.task_pipes[rollout_id][0].send(0)
		# self.learner.q_estimator.to(device=self.device)

		#Start Test rollouts
		if gen % self.args.evaluate_every_gen == 0:
			self.test_flag = True
			for pipe in self.test_task_pipes: pipe[0].send(0)


		# ############# UPDATE PARAMS USING GRADIENT DESCENT ##########
		# if self.replay_buffer.__len__() > self.args.learning_start: ###BURN IN PERIOD
		# 	for _ in range(int(self.gen_frames * self.args.gradperstep)):
		# 		ts = self.replay_buffer.sample(self.args.batch_size)
		# 		self.learner.feed(ts)
		#
		# 	self.gen_frames = 0


		########## JOIN ROLLOUTS FOR EVO POPULATION ############
		all_fitness = []; all_eplens = []
		if self.args.pop_size > 1:
			for i in range(self.args.pop_size):
				_, fitness, frames, trajectory = self.evo_result_pipes[i][1].recv()

				all_fitness.append(fitness); all_eplens.append(frames)

				self.gen_frames+= frames; self.total_frames += frames
				total_ts.append(trajectory)
				self.best_score = max(self.best_score, fitness)
				gen_max = max(gen_max, fitness)

		#Only learn from champion
		champ_index = all_fitness.index(max(all_fitness))
		total_ts = [total_ts[champ_index]]

		########## JOIN ROLLOUTS FOR LEARNER ROLLOUTS ############
		rollout_fitness = []; rollout_eplens = []
		if self.args.rollout_size > 0:
			for i in range(self.args.rollout_size):
				_, fitness, pg_frames, trajectory = self.result_pipes[i][1].recv()
				total_ts.append(trajectory)
				self.gen_frames += pg_frames; self.total_frames += pg_frames
				self.best_score = max(self.best_score, fitness)
				gen_max = max(gen_max, fitness)
				rollout_fitness.append(fitness); rollout_eplens.append(pg_frames)

		######################### END OF PARALLEL ROLLOUTS ################

		############ FIGURE OUT THE CHAMP POLICY AND SYNC IT TO TEST #############
		if self.args.pop_size > 1:

			# utils.hard_update(self.test_bucket[0], self.population[champ_index])
			self.test_bucket[0] = copy.copy(self.population[champ_index])
			# if max(all_fitness) > self.best_score:
			# 	self.best_score = max(all_fitness)
			# 	utils.hard_update(self.best_policy, self.population[champ_index])
			# 	torch.save(self.population[champ_index], self.args.log_dir + 'model_pop' + str(self.args.pop_size)+'.pth')
			# 	print("Best policy saved with score", '%.2f'%max(all_fitness))

		else: #If there is no population, champion is just the actor from policy gradient learner
			# utils.hard_update(self.test_bucket[0], self.rollout_bucket[0])
			self.test_bucket[0] = copy.copy(self.learner.q_estimator)

		###### TEST SCORE ######
		if self.test_flag:
			self.test_flag = False
			test_scores = []
			for pipe in self.test_result_pipes: #Collect all results
				_, fitness, _, _ = pipe[1].recv()
				self.best_score = max(self.best_score, fitness)
				gen_max = max(gen_max, fitness)
				test_scores.append(fitness)
			test_scores = np.array(test_scores)
			test_mean = np.mean(test_scores)
			tracker.update([test_mean], gen)

		else:
			test_mean = None


		#NeuroEvolution's probabilistic selection and recombination step
		if self.args.pop_size > 1:
			self.evolver.epoch(gen, self.population, all_fitness, self.rollout_bucket)

		#RL training
		if self.args.rollout_size >0:
			for trajectory in total_ts:
				for tj in trajectory:
					for j in tj:
						self.learner.feed(j)
		#Compute the champion's eplen
		# champ_len = all_eplens[all_fitness.index(max(all_fitness))] if self.args.pop_size > 1 else rollout_eplens[rollout_fitness.index(max(rollout_fitness))]


		return gen_max, test_mean, rollout_fitness


	def train(self):
		# Define Tracker class to track scores
		if self.args.rollout_size==0:
			test_tracker = utils.Tracker(self.args.log_dir, ['NE_pop' + str(self.args.pop_size)], '.csv')  # Tracker class to log progress
		else:
			test_tracker = utils.Tracker(self.args.log_dir, ['NE_RL_pop' + str(self.args.pop_size)], '.csv')
		for gen in range(self.args.max_gen):

			# Train one iteration
			max_fitness, test_mean, rollout_fitness = self.forward_generation(gen, test_tracker)
			if test_mean: self.args.writer.add_scalar('reward', test_mean, gen)

			# print('Gen/Frames:', gen,'/',self.total_frames,
			# 	  ' Gen_max_score:', '%.2f'%max_fitness,
			#  ' Test_score u/std', utils.pprint(test_mean), utils.pprint(test_std),
			# 	  ' Rollout_u/std:', utils.pprint(np.mean(np.array(rollout_fitness))), utils.pprint(np.std(np.array(rollout_fitness))))

			#EA only
			print('Gen:', gen, ' Gen_max_score:', '%.2f'%max_fitness)
			test_tracker.update([max_fitness], gen)
			# if test_mean: print('\n', gen, test_mean)

			# if gen % 5 == 0:
			# 	print('Best_score_ever:''/','%.2f'%self.best_score)
			# 	print()

		###Kill all processes
		try:
			for p in self.task_pipes: p[0].send('TERMINATE')
			for p in self.test_task_pipes: p[0].send('TERMINATE')
			for p in self.evo_task_pipes: p[0].send('TERMINATE')
		except:
			None




