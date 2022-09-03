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
        
        #Create dictionary for e2 detectors in TLS using trafficlightID as key
        self.get_e2_detectors()
    
    def take_action(self, tlsID, phase_no):
        pass
    
    def get_neighbors(self):
        pass

    def get_reward(self):
        pass

    def get_e2_detectors(self):
        all_tls = traci.trafficlight.getIDList()
        all_e2_detectors = traci.lanearea.getIDList()
        for tls in all_tls:
            tls_pos = traci.getNode
            controlled_lanes = traci.trafficlight.getControlledLanes(tls)
            detectors_in_controlled_lanes = [i for i in all_e2_detectors if i in controlled_lanes]

    def get_state(self, trafficlight):
        #Get ID list of detectors
        e2_detectors = self.detectors

        queues = [traci.lanearea.getLastStepVehicleNumber(detector) for detector in e2_detectors] #gets the queus in detectors

        tl_phase = traci.trafficlight.getPhase(trafficlight)
        one_hot_vector_tl_phase = np.eye(self.total_phases)[tl_phase]
        arry = np.hstack([queues, one_hot_vector_tl_phase])

        return arry
        
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
    