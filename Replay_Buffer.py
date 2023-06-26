#Importing Libraries

import os
import time
import random
from torch.autograd import Variable
from collections import deque
import pandas as pd
import numpy as np

#Defining the replay buffer class

class ReplayBuffer_pretrain(object):

  #Size of the replay buffer
  def __init__(self, max_size= 1e6):
    self.storage = []
    self.max_size = max_size
    self.ptr = 292 * 24   #number of total timesteps in training days

  #Add function for adding a transition (also making sure not overriding expert demonstrations)
  def add(self, transition):

    if len(self.storage)== self.max_size:
        if self.ptr==(self.max_size-1):
            self.ptr=292*24
        else:
            self.storage[int (self.ptr)]= transition
            self.ptr +=1
            
    else:
      self.storage.append(transition)

  #Sampling a batch of transitions

  def sample(self, batch_size):
    ind = np.random.randint(0, len(self.storage), size= batch_size)
    batch_states, batch_next_states, batch_actions, batch_next_actions, batch_rewards, batch_costs_1, batch_costs_2, batch_costs_3 , batch_costs_4, batch_dones = [],[], [], [], [], [], [], [], [], []
    for i in ind:
      state, next_state, action, next_action, reward, cost_1,cost_2,cost_3, cost_4, done = self.storage[i]
      batch_states.append(np.array(state, copy=False))
      batch_next_states.append(np.array(next_state, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_next_actions.append(np.array(next_action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_costs_1.append(np.array(cost_1, copy=False))
      batch_costs_2.append(np.array(cost_2, copy=False))
      batch_costs_3.append(np.array(cost_3, copy=False))
      batch_costs_4.append(np.array(cost_4, copy=False))

      batch_dones.append(np.array(done, copy=False))
    return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions),np.array(batch_next_actions),np.array(batch_rewards).reshape(-1,1),np.array(batch_costs_1).reshape(-1,1),np.array(batch_costs_2).reshape(-1,1),np.array(batch_costs_3).reshape(-1,1) ,np.array(batch_costs_4).reshape(-1,1) ,np.array(batch_dones).reshape(-1,1)
    
    
