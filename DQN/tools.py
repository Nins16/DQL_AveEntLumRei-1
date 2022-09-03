import time
import threading
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim
import numpy as np
import traci
import optparse
import os
import sys
import pandas as pd
import json
from tqdm import tqdm
from collections import deque
import random
import torch.nn.functional as F

from sumolib import checkBinary  # noqa
import traci  # noqa

class Net:
    def __init__(self,states_length,total_phases):
        super().__init__()
        self.states_length = states_length
        self.total_phases = total_phases
        self.fc1 = nn.Linear(states_length,1000)
        self.fc2 = nn.Linear(1000,500)
        self.fc3 = nn.Linear(500,self.total_phases)
    
    def forward(self, x):
        #Neural Network Forward Pass Layers
        x = F.relu(self.fc1(x))
        # x = nn.BatchNorm1d(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

class SmartTLS:
    def __init__(self, tlsID):
        self.tlsID = tlsID
        self.model = Net()
    def get_reward(self):
        pass
    def get_state(self):
        pass
    def train(self):
        pass
    def net(self):
        pass
    def total_phases(self):
        pass

