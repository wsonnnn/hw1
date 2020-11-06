from collections import namedtuple
import random

class ReplayBuffer():
    def __init__(self,size):
        self.size=size
        self.buffer=[]
        self.position=0
        
    def push(self, trans):
        if len(self.buffer) < self.size:
            self.buffer.append(trans)
        else:
            self.buffer[self.position] = trans
        self.position = ( self.position + 1 ) % self.size
        #print(self.buffer)


    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
