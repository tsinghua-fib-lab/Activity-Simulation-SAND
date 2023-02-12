from collections import deque
import random
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import mlflow

class RolloutStorage(object):
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.memory = []
        self.dis_memory = []
    
    def store(self, state, action, reward, done, value, action_log_prob,action_dist_entropy,z0):
        self.memory.append([state, action, reward, done, value, action_log_prob,action_dist_entropy,z0])
        self.dis_memory.append([state, action])
        
    
    def compute_returns(self):
        gae = 0
        value_previous = 0
        for step in reversed(list(self.memory)):

            reward = step[2]
            done = step[3]
            mask = 1-done
            value = step[4]
            delta = reward + self.args.gamma * value_previous * mask - value
            gae = delta + self.args.gamma * self.args.lmbda * mask * gae
            returns = gae + value
            
            step.append(gae)
            step.append(returns)
            value_previous = value
        
        
    def sample(self,batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, reward, done, value, action_log_prob,action_dist_entropy,z0, returns,advantages = zip(*batch)

        return state, action, reward, done, value, action_log_prob,action_dist_entropy,z0, returns,advantages

        
    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()
