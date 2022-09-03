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
        return F.sigmoid(x)

class SmartTLS:
    def __init__(self, tlsID):
        self.tlsID = tlsID
    def init_model(self,neighbors):
        pass
    def get_reward(self):
        pass
    def get_state(self):
        pass
    def train(self):
        pass
    def total_phases(self):
        pass

class SumoEnvironment:
    def __init__(self, gui = True, buffer_size = 15, buffer_yellow = 6, train=False):
        #Set Buffer Size
        self.buffer_yellow = buffer_yellow
        self.buffer_size = buffer_size
        if self.buffer_size < self.buffer_yellow:
            raise ValueError("Buffer size must be greater than yellow buffer")

        #Set GUI boolean condition
        self.gui = gui

        #initialize program
            # this script has been called from the command line. It will start sumo as a server, then connect and run
        if not self.gui:
            self.sumoBinary = checkBinary('sumo')
        else:
            self.sumoBinary = checkBinary('sumo-gui')
            # we need to import python modules from the $SUMO_HOME/tools directory
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")
        traci.start([self.sumoBinary, "-c", "Simulation_Environment\Main Route Simulation\osm.sumocfg",
                             "--tripinfo-output", "Data\\tripinfo.xml",  "--start"])
    def take_action(self, tlsID, phase_no):
        pass
    
    def get_neighbors(self):
        pass

    def get_reward(self):
        pass
    
    def get_state(self, tlsID):
        pass
    def get_joint_action(self, neighbors):
        pass

def train(M=5):
    ### Please start from scratch. Use this as guide.
    # for m in range(M):
    #     for j in agents.keys():
    #         state = env.get_state(j)
    #         neighbors = env.get_neighbors(j)
    #         joint_actions = env.get_joint_action(neighbors)
    #         q_values = net(state, joint_actions)
    #         if random.random() < epsilon:
    #             action = np.random.randint(0,j.total_phases)
    #         else:
    #             action = torch.argmax(q_values)

    #Initialize environment
    env = SumoEnvironment(gui=False, buffer_size=10, buffer_yellow=3, train=True)

    #initialize agents
    agents = {}
    