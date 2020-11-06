import numpy as np
import pickle
import collections as namedtuple
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_buffer import ReplayBuffer
from dqn_net import DQN_NET
import copy
batch_size = 32
lr = 0.05
eps = 0.9
gamma = 0.9
target_update = 100
buffer_size = 3000
env = gym.make('CartPole-v0')
env=env.unwrapped
action_space = 2
state_space = 4

class dqn_learner():
    def __init__(self):
        self.agent = DQN_NET()
        self.target_agent = copy.deepcopy(self.agent)
        self.buffer = ReplayBuffer(buffer_size)
        self.optimizer = torch.optim.Adam(params=self.agent.parameters(),lr=lr)
        self.loss_func = nn.MSELoss()
        self.step=0
    def select_action(self,x):
        rand = np.random.uniform()
        #print("x1 = ",x)
        x = torch.FloatTensor(x)
        #print("x2 = ",x)
        #x = torch.unsqueeze(x,0)
        #print("x3 = ",x)
        if rand<eps:
            values = self.agent.forward(x)
            #print("value:",values)
            action = torch.max(values,0)[1].numpy()
            action = action
        else:
            action = np.random.randint(0,action_space)
        return action

    def train(self):
        if self.step % target_update == 0:
            self.target_agent.load_state_dict(self.agent.state_dict())
        self.step += 1
        
        batch = np.array(self.buffer.sample(batch_size))
        #print(batch)
        b_st = torch.FloatTensor(batch[:,:state_space])
        b_at = torch.LongTensor(batch[:,state_space:state_space+1])
        b_rt = torch.FloatTensor(batch[:,state_space+1:state_space+2])
        b_stt = torch.FloatTensor(batch[:,-state_space:])

        q_agent = self.agent(b_st).gather(1,b_at)
        q_target_agent = self.target_agent(b_stt).detach()
        target = b_rt + gamma * q_target_agent.max(1)[0].view(batch_size,1)
        loss = self.loss_func(q_agent,target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

