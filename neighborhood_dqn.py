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
import sumolib
from DQN import tools

class SumoEnvironment:
    def __init__(self, gui = True, buffer_size = 10, buffer_yellow = 3, train=False,
                dir=Path("Simulation_Environment\Main DQN"),neighbor_limit = 2):
        #Set directory of environment
        self.dir = dir

        #Create Num
        self.net = sumolib.net.readNet(Path(f"{self.dir}\\osm.net.xml"))

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
        traci.start([self.sumoBinary, "-c", f"{self.dir}\osm.sumocfg",
                             "--tripinfo-output", f"{self.dir}\\Results\\tripinfo.xml",  "--start"])
        
        #Get Phase Data of Each Traffic Light
        self.get_phase_data()

        #Create dictionary for e2 detectors in TLS using trafficlightID as key
        self.get_e2_detectors()

        #Set neighbor Limit
        self.neighbor_limit = neighbor_limit

        #Create Dict for each trafficlight
        self.init_tls_properties()


        #Create Network for each traffic light if in training mode
        self.train = train
        if train == True:
            self.init_neural_net()


    def get_phase_data(self):
        """Convert Phase SUMO XML TL Program phase data to native python string phase data
        """
        self.total_phases = {}
        self.previous_tl_state = {}
        self.phases = {}
        for traffic_light in traci.trafficlight.getIDList():
            phases_objects=traci.trafficlight.getCompleteRedYellowGreenDefinition(traffic_light)[0].getPhases()
            self.phases[traffic_light] = [phase.state for phase in phases_objects]
            self.total_phases[traffic_light] = len(self.phases)
            self.previous_tl_state[traffic_light] = traci.trafficlight.getRedYellowGreenState(traffic_light)

    def take_action(self, tlsID, phase_no):
        pass

    def get_reward(self):
        pass
    
    def get_e2_detectors(self):
        #Get all tls and e2 detectors
        all_tls = traci.trafficlight.getIDList()
        all_e2_detectors = traci.lanearea.getIDList()
        
        #initialize data storage for e2 detectors of each lane
        #Note that there should be only one e2 detector per lane
        self.e2_detectors = {}
        for tls in all_tls:
            controlled_lanes = traci.trafficlight.getControlledLanes(tls)
            detectors_in_controlled_lanes = [detector for detector in all_e2_detectors if traci.lanearea.getLaneID(detector) in controlled_lanes]
            self.e2_detectors[tls] = detectors_in_controlled_lanes

    def get_state(self, trafficlight, joint_action):
        #Get ID list of detectors
        e2_detectors = self.e2_detectors[trafficlight]
        
        #Get Queues of each detector
        queues = [traci.lanearea.getLastStepVehicleNumber(detector) for detector in e2_detectors] #gets the queus in detectors

        #get current Phase ID
        tl_phase = traci.trafficlight.getPhase(trafficlight)

        #return state (queues, ohe of tl phase, ordinal joint action of neighbors)
        one_hot_vector_tl_phase = np.eye(self.total_phases[trafficlight])[tl_phase]
        arry = np.hstack([queues, one_hot_vector_tl_phase, joint_action])

        return arry
        
    def get_joint_action(self, trafficlight):
        #What is the joint action? Up for them to decide
        pass
    
    def init_tls_properties(self):
        """initializes the properties of each tls"""
        self.tls = {}
        all_tls = traci.trafficlight.getIDList()
        
        #Create a dictionary for each tls
        for tls in all_tls:
            self.tls[tls] = {}

        #Add neighbour list to tls dict
        neighbors = tools.get_neighbors(2,self.net)
        for key,val in neighbors.items(): 
            item = self.tls[key]
            item['neighbor'] = val
        
    def init_neural_net(self):
        """Initializes the neural network of each ITS"""
        #Create dictionary for tls neighbors
        self.get_neighbors(self.neighbor_limit)
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
    