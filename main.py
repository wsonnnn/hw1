import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import datetime
from dqn_learner import dqn_learner
from tensorboardX import SummaryWriter
now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
tenboard_dir = './tensorboard/'+now_time+'/'
writer = SummaryWriter(log_dir = tenboard_dir)

if __name__ == '__main__':
    dqn = dqn_learner()
    env = gym.make('CartPole-v0')
    env=env.unwrapped
    for episode in range(5000):
        st = env.reset()
        r_t = 0
        while True:
            env.render()
            #print(st)
            a = dqn.select_action(st)
            stt , r, done, info = env.step(a)
            trans = np.hstack((st, [a, r], stt))
            #print(trans)
            dqn.buffer.push(trans)
            r_t += r
            if dqn.buffer.__len__() == dqn.buffer.size:
                dqn.train()
                if done :
                    print("episode: ",episode," done with reward ",r_t)
                
            if done:
                break
            st = stt
        writer.add_scalar('reward/graph',r_t,episode)
    torch.save(dqn.agent.state_dict,"./agent.th")