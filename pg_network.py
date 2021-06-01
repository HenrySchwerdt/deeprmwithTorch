import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class DQN(nn.Module):

    def __init__(self,w,outdims):
        super(DQN, self).__init__()
        self.input = nn.Linear(w,400,bias=True)
        self.hidden_1 = nn.Linear(400,800, bias=True)
        self.hidden_2 = nn.Linear(800,1200, bias=True)
        self.hidden_3 = nn.Linear(1200,400, bias=True)
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.output = nn.Linear(400,outdims)

    def forward(self, x):
        x = F.relu(self.hidden_1(self.input(x)))
        x = F.relu(self.hidden_2(x))
        x = F.relu(self.hidden_3(x))
        x = self.dropout(x)
        x = F.softmax(self.output(x),dim=0)
        return x
