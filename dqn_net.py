import torch.nn as nn
import torch.nn.functional as F

class DQN_NET(nn.Module):
    def __init__(self):
        super(DQN_NET, self).__init__()
        self.input_shape = 4
        self.action_space = 2
        self.fc1 = nn.Linear(self.input_shape, 32)
        self.fc1.weight.data.normal_(0,0.2)
        self.fc2 = nn.Linear(32, 64)
        self.fc2.weight.data.normal_(0,0.2)
        self.fc3 = nn.Linear(64, 64)
        self.fc3.weight.data.normal_(0,0.2)
        self.fc4 = nn.Linear(64, self.action_space)
        self.fc4.weight.data.normal_(0,0.2)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x
