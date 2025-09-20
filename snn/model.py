import torch
import torch.nn as nn
from layer import *

class MNISTNet(nn.Module):
    def __init__(self,T=6):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc1_s = tdLayer(self.fc1)
        self.fc2 = nn.Linear(256, 128)
        self.fc2_s = tdLayer(self.fc2)
        self.fc3 = nn.Linear(128, 10)
        self.fc3_s = tdLayer(self.fc3)
        self.relu = nn.ReLU()
        self.act = LIFSpike()
        self.T = T


    def forward(self, x):
        x = x.view(-1, 784)
        x= add_dimention(x, self.T)
        x = self.act(self.fc1_s(x))

        x = self.act(self.fc2_s(x))

        x = self.fc3_s(x)
        x= x.mean(1)
        return x