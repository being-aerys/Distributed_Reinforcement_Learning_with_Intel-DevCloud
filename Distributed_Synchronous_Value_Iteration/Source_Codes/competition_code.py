
import ray
import time
from copy import deepcopy
import matplotlib.pyplot as plt
from random import randint, choice
get_ipython().run_line_magic('matplotlib', 'inline')
import pickle




def evaluate_policy(env, policy, trials = 1000):
    total_reward = 0
    for _ in range(trials):
        env.reset()
        done = False
        observation, reward, done, info = env.step(policy[0])
        total_reward += reward
        while not done:
            observation, reward, done, info = env.step(policy[observation])
            total_reward += reward
    return total_reward / trials
def evaluate_policy_discounted(env, policy, discount_factor, trials = 1000):
    total_reward = 0
    #INSERT YOUR CODE HERE
    for _ in range(trials):
        env.reset()
        done = False
        observation, reward, done, info = env.step(policy[0])
        total_reward += reward
        beta = 1
        while not done:
            observation, reward, done, info = env.step(policy[observation])
            
            beta = beta*discount_factor
            total_reward = total_reward + beta*reward
    return total_reward / trials
def print_results(v, pi, map_size, env, beta, name):
    v_np, pi_np  = np.array(v), np.array(pi)
    print("\nState Value:\n")
    print(np.array(v_np[:-1]).reshape((map_size,map_size)))
    print("\nPolicy:\n")
    print(np.array(pi_np[:-1]).reshape((map_size,map_size)))
    print("\nAverage reward: {}\n".format(evaluate_policy(env, pi)))
    print("Avereage discounted reward: {}\n".format(evaluate_policy_discounted(env, pi, discount_factor = beta)))
    print("State Value image view:\n")
    plt.imshow(np.array(v_np[:-1]).reshape((map_size,map_size)))
    
    pickle.dump(v, open(name + "_" + str(map_size) + "_v.pkl", "wb"))
    pickle.dump(pi, open(name + "_" + str(map_size) + "_pi.pkl", "wb"))


# In[3]:


ray.shutdown()
ray.init(include_webui=False, ignore_reinit_error=True, redis_max_memory=500000000, object_store_memory=5000000000)


# In[10]:



@ray.remote
class VI_server_v2(object):
    #INSERT YOUR CODE HERE
    def __init__(self,size):
        self.v_current=[0]*size
        self.v_new = [0]*size
        self.pi = [0]*size
    def get_value_and_policy(self):
        return self.v_current, self.pi
    
    def update(self, start_state,end_state,max_actions,max_values):
        
        for state in range(start_state,end_state):
            self.v_new[state] = max_values[state-start_state]
            self.pi[state] = max_actions[state-start_state]
        #print("called by a worker")
    
    def get_error_and_update(self):
        max_error = 0
        for i in range(len(self.v_current)):
            error = abs(self.v_new[i] - self.v_current[i])
            if error > max_error:
                max_error = error
            self.v_current[i] = self.v_new[i]
            
        return max_error
    
@ray.remote
def VI_worker_v2(VI_server, data, start_state, end_state):
        env, workers_num, beta, epsilon = data
        A = env.GetActionSpace()
        S = env.GetStateSpace()
        
        #INSERT YOUR CODE HERE
        V, _ = ray.get(VI_server.get_value_and_policy.remote())
        
        action_chosen = [0]*(end_state-start_state+1)
        values_for_state =[0]*(end_state-start_state+1)
        #print("beta is",beta)
        for state in range(start_state,end_state):
            max_v = float('-inf')
            max_a = 0
            for action in range(A):
                succ = env.GetSuccessors(state,action)
                tp_score = 0
                for st,prob in succ:
                    tp_score+= (prob*V[st])

                tp_score = env.GetReward(state,action) + beta*tp_score
                if max_v < tp_score:
                    max_v = tp_score
                    max_a = action
            action_chosen[state-start_state]= max_a
            values_for_state[state-start_state] = max_v
            
        VI_server.update.remote(start_state,end_state,action_chosen,values_for_state)
        return data
                    
def fast_value_iteration(env, beta = 0.999, epsilon = 0.0001, workers_num = 4):
   
    S = env.GetStateSpace()
    VI_server = VI_server_v2.remote(S)
    
    start_and_last = []
    data_id = ray.put((env, workers_num, beta, epsilon))
    
   
    first = None
    last = 0
    batch_size = int(S/workers_num)
    
    for i in range(workers_num):
        first = last
        last = min(first+batch_size, S)
        start_and_last.append([first,last])
        
        
    
    error = float('inf')
    
    
    while error > epsilon:
    
        workers_list = []
        for i in range(workers_num):
            w_id = VI_worker_v2.remote(VI_server, data_id,start_and_last[i][0],start_and_last[i][1])
            workers_list.append(w_id)
            
        results,_ = ray.wait(workers_list, num_returns = workers_num, timeout = None)
        error = ray.get(VI_server.get_error_and_update.remote())
    
    v, pi = ray.get(VI_server.get_value_and_policy.remote())
    return v, pi




