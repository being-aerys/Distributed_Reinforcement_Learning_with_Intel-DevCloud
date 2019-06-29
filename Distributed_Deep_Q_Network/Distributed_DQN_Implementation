import torch
import time
import os
import ray
from tqdm import tqdm
from random import uniform, randint

from dqn_model import DQNModel
from dqn_model import _DQNModel
from memory import ReplayBuffer

import matplotlib.pyplot as plt

FloatTensor = torch.FloatTensor

ENV_NAME = 'Distributed_CartPole'



def plot_result(total_rewards, learning_num, legend):
    print("\nLearning Performance:\n")
    episodes = []
    for i in range(len(total_rewards)):
        episodes.append(i * learning_num + 1)

    plt.figure(num=1)
    fig, ax = plt.subplots()
    plt.plot(episodes, total_rewards)
    #plt.title('Performance')
    plt.legend(legend)
    plt.xlabel("Episodes")
    plt.ylabel("Total Rewards")
    plt.savefig("Distributed_DQN_4_Collectors_4_Workers.png")
    plt.show()


ray.shutdown()
ray.init(include_webui=False, ignore_reinit_error=True, redis_max_memory=500000000, object_store_memory=5000000000)

from memory_remote import ReplayBuffer_remote
from dqn_model import _DQNModel
import torch
from custom_cartpole import CartPoleEnv
from collections import deque

simulator = CartPoleEnv()
result_folder = ENV_NAME
result_file = ENV_NAME + "/results4.txt"

if not os.path.isdir(result_folder):
    os.mkdir(result_folder)
torch.set_num_threads(12)

Memory_Server = ReplayBuffer_remote.remote(2000)


@ray.remote
class DQN_Model_Server():
    def __init__(self, env, hyper_params, batch_size, update_steps, memory_size, beta, model_replace_freq,
                 learning_rate, use_target_model=True, memory=Memory_Server, action_space=2,
                 training_episodes=7000, test_interval=50):
        # super().__init__(update_steps, memory_size, model_replace_freq, learning_rate, beta=0.99, batch_size = 32, use_target_model=True)
        self.batch_size = batch_size

        state = env.reset()
        input_len = len(state)
        output_len = action_space
        self.eval_model = DQNModel(input_len, output_len, learning_rate=0.0003)
        self.target_model = DQNModel(input_len, output_len)
        self.steps = 0
        self.memory = memory
        # self.memory = ReplayBuffer(hyper_params['memory_size'])
        self.prev = 0
        self.next = 0
        self.model_dq = deque()
        self.result = [0] * ((training_episodes // test_interval) + 1)
        self.previous_q_networks = []
        self.result_count = 0
        self.learning_episodes = training_episodes
        self.episode = 0
        self.is_collection_completed = False
        self.evaluator_done = False
        self.batch_num = training_episodes // test_interval
        self.use_target_model = True
        self.beta = 0.99
        self.test_interval = test_interval


    def get_evaluation_model(self):

        if self.episode >= self.learning_episodes:
            self.is_collection_completed = True

        return self.is_collection_completed


    def replace(self):
        self.target_model.replace(self.eval_model)

    def get_total_steps(self):
        return self.steps


    def predict_next(self, state, e_model):
        return e_model.predict(state)

    def get_predict(self, state):
        return self.eval_model.predict(state)

    def set_collect_count(self):
        self.next += 1

    def set_collector_count(self):
        self.episode += 1


    def get_evaluation_count(self):
        return self.result_count

    def get_evaluator_count(self):
        return self.episode

    def ask_evaluation(self):
        if len(self.previous_q_networks) > self.result_count:
            num = self.result_count
            evluation_q_network = self.previous_q_networks[num]
            self.result_count += 1
            self.episode += 50
            return evluation_q_network, False, num
        else:
            if self.episode >= self.learning_episodes:
                self.evaluator_done = True
            return [], self.evaluator_done, None

    def update_batch(self):

        self.steps += 10

        if ray.get(self.memory.__len__.remote()) < self.batch_size:  # or self.steps % self.update_steps != 0:
            return

        if self.is_collection_completed:
            return

        batch = ray.get(self.memory.sample.remote(self.batch_size))

        (states, actions, reward, next_states,
         is_terminal) = batch

        states = states
        next_states = next_states
        terminal = FloatTensor([1 if t else 0 for t in is_terminal])
        reward = FloatTensor(reward)
        batch_index = torch.arange(self.batch_size,
                                   dtype=torch.long)

        _, q_values = self.eval_model.predict_batch(states)
        q_values = q_values[batch_index, actions]

        if self.use_target_model:
            actions, q_next = self.target_model.predict_batch(next_states)
        else:
            actions, q_next = self.eval_model.predict_batch(next_states)#dont need though

        q_targets = []

        for i in range(0, len(terminal), 1):
            if terminal[i] == 1:
                q_targets.append(reward[i])
            else:
                q_targets.append(reward[i] + (self.beta * torch.max(q_next, 1).values[i].data))

        q_target = FloatTensor(q_targets)

        self.eval_model.fit(q_values, q_target)

        if self.episode // self.test_interval + 1 > len(self.previous_q_networks):
            model_id = ray.put(self.eval_model)
            self.previous_q_networks.append(model_id)
        return self.steps

    def set_results(self, result, num):
        self.result[num] = result

    def get_results(self):
        return self.result


@ray.remote
def collecting_worker(model_server, env, update_steps, max_episode_steps, training_episodes, test_interval,
                      model_replace_freq, memory=Memory_Server, action_space=2):
    initial_epsilon = 1
    final_epsilon = 0.1

    def greedy_policy(curr_state):
        return ray.get(model_server.get_predict.remote(curr_state))

    def linear_decrease(initial_value, final_value, curr_steps, final_decay_steps):
        decay_rate = curr_steps / final_decay_steps
        if decay_rate > 1:
            decay_rate = 1
        return initial_value - (initial_value - final_value) * decay_rate

    def explore_or_exploit_policy(curr_state):

        p = uniform(0, 1)
        self_steps = ray.get(model_server.get_total_steps.remote())
        epsilon = linear_decrease(initial_value=initial_epsilon, final_value=final_epsilon, curr_steps=self_steps,
                                  final_decay_steps=100000)

        if p < epsilon:
            return randint(0, action_space - 1)
        else:
            return greedy_policy(curr_state)


    while True:
        collect_done = ray.get(model_server.get_evaluation_model.remote())
        if collect_done:
            break

        for episode in tqdm(range(test_interval), desc="Training"):
            state = env.reset()
            done = False
            steps = 0

            model_replace_freq = 2000

            while steps < max_episode_steps and not done:

                action = explore_or_exploit_policy(state)
                next_state, reward, done, info = env.step(action)
                memory.add.remote(state, action, reward, next_state, done)
                state = next_state
                steps += 1

                if steps % update_steps == 0:
                    model_server.update_batch.remote()

                total_step = ray.get(model_server.get_total_steps.remote())
                if total_step % model_replace_freq == 0:
                    model_server.replace.remote()



@ray.remote
def evaluation_worker(model_server, env, training_episodes, test_interval, eval_worker, trials=30):
    best_reward = 0

    def greedy_policy(curr_state, eval_model):
        return ray.get(model_server.predict_next.remote(curr_state, eval_model))

    while True:

        model_id, done, num = ray.get(model_server.ask_evaluation.remote())
        eval_model = ray.get(model_id)

        eval_num = ray.get(model_server.get_evaluation_count.remote())

        if done:
            break
        total_reward = 0
        if eval_model == []:  #
            continue
        for _ in tqdm(range(trials), desc="Evaluating"):
            state = env.reset()
            done = False
            steps = 0

            while steps < env._max_episode_steps and not done:
                steps += 1
                action = greedy_policy(state, eval_model)  # predicted action of the eval model on specific state
                # action = eval_model.predict(state)
                state, reward, done, _ = env.step(action)
                total_reward += reward

        avg_reward = total_reward / trials

        model_server.set_results.remote(avg_reward, num)

        print(avg_reward)
        f = open(result_file, "a+")
        f.write(str(avg_reward) + "\n")
        f.close()


    return avg_reward


class distributed_DQN_agent():
    def __init__(self, env, hyper_params, cw_num, ew_num, epsilon_decay_steps=100000, final_epsilon=0.1, batch_size=32,
                 update_steps=10, memory_size=2000, beta=0.99, model_replace_freq=2000,
                 learning_rate=0.003, do_test=True, memory=Memory_Server, action_space=2,
                 training_episodes=7000, test_interval=50):
        self.env = env
        self.max_episode_steps = env._max_episode_steps
        self.update_steps = update_steps
        self.model_server = DQN_Model_Server.remote(env, hyper_params, batch_size, update_steps, memory_size, beta,
                                                    model_replace_freq,
                                                    learning_rate, use_target_model=True, memory=Memory_Server,
                                                    action_space=2)
        self.workers_id = []
        self.final_epsilon = final_epsilon
        self.batch_size = batch_size
        self.cw_num = cw_num
        self.ew_num = ew_num
        self.beta = beta
        self.do_test = do_test
        self.memory = Memory_Server
        self.action_space = action_space
        self.training_episodes = training_episodes
        self.test_interval = test_interval
        self.model_replace_freq = model_replace_freq


    def learn_and_evaluate(self):

        workers_id = []

        for i in range(self.cw_num):
            cw_id = collecting_worker.remote(self.model_server, self.env, self.update_steps, self.max_episode_steps,
                                             training_episodes, test_interval, self.model_replace_freq, Memory_Server,
                                             self.action_space)
            workers_id.append(cw_id)
        for i in range(self.ew_num):
            ew_id = evaluation_worker.remote(self.model_server, self.env, self.training_episodes, self.test_interval,
                                             self.ew_num)
            workers_id.append(ew_id)

        ray.wait(workers_id, len(workers_id))
        return ray.get(self.model_server.get_results.remote())


hyperparams_CartPole = {
    'epsilon_decay_steps': 100000,
    'final_epsilon': 0.1,
    'batch_size': 32,
    'update_steps': 10,
    'memory_size': 2000,
    'beta': 0.99,
    'model_replace_freq': 2000,
    'learning_rate': 0.0003,
    'use_target_model': True
}

cw_nums = [4]
ew_nums = [4]

for cw_num, ew_num in zip(cw_nums, ew_nums):
    start_time = time.time()
    training_episodes, test_interval = 5500, 50#note 5500
    distributed_dqn_agent = distributed_DQN_agent(simulator, hyperparams_CartPole, cw_num=cw_num, ew_num=ew_num,
                                                  epsilon_decay_steps=100000, final_epsilon=0.1, batch_size=32,
                                                  update_steps=10, memory_size=2000,
                                                  beta=0.99, model_replace_freq=140, learning_rate=0.003, do_test=True,
                                                  memory=Memory_Server, action_space=2,
                                                  training_episodes=training_episodes, test_interval=test_interval)
    result = distributed_dqn_agent.learn_and_evaluate()
    run_time = time.time() - start_time


    print("Learning time:\n",run_time)
    #plot_result(result, test_interval, ["Distributed DQN"])


