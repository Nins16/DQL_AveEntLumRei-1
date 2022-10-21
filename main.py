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
                dir=Path("Simulation_Environment\Main DQN"),neighbor_limit = 2, cycle_length=120):
        #Set directory of environment
        self.dir = dir

        #Create Num
        self.net = sumolib.net.readNet(Path(f"{self.dir}\\osm.net.xml"))

        #Set Buffer Size
        self.buffer_yellow = buffer_yellow
        self.buffer_size = buffer_size
        if self.buffer_size < self.buffer_yellow:
            raise ValueError("Buffer size must be greater than yellow buffer")

        #Set Cycle Length
        self.cycle_length = cycle_length

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

        #Create dictionary for e2 detectors in TLS using trafficlightID as key
        self.get_e2_detectors()

        #Set neighbor Limit
        self.neighbor_limit = neighbor_limit

        #Create Dict for each trafficlight
        self.init_tls_properties()

        #Get Phase Data of Each Traffic Light
        self.get_phase_data()

        #Randomize state
        self.randomize_state()

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
            tls_dict = self.tls[traffic_light]
            phases_objects=traci.trafficlight.getCompleteRedYellowGreenDefinition(traffic_light)[0].getPhases()
            #Gets the phase index where there is no transition phase(basically, no "yellow phase")
            tls_dict['phases'] = [idx for idx, phase in enumerate(phases_objects) if 'y' not in phase.state]
            #Gets the numer of phases
            tls_dict['total_phases'] = len(tls_dict['phases'])

    def get_reward(self):
        pass
    
    def get_e2_detectors(self):
        #Get all tls and e2 detectors
        all_tls = traci.trafficlight.getIDList()
        all_e2_detectors = traci.lanearea.getIDList()
        
        #initialize data storage for e2 detectors of each lane
        #Note that there should be only one e2 detector per lane(this should be true even for network application)
        self.e2_detectors = {}
        for tls in all_tls:
            controlled_lanes = traci.trafficlight.getControlledLanes(tls)
            detectors_in_controlled_lanes = [detector for detector in all_e2_detectors if traci.lanearea.getLaneID(detector) in controlled_lanes]
            self.e2_detectors[tls] = detectors_in_controlled_lanes
        
        #Duplicate Error Checking
        for idx_1 in len(all_tls):
            for idx_2, tls_2 in enumerate(all_tls):
                if idx_1 == idx_2:
                    continue
                for detector in self.e2_detectors[idx_1]:
                    if detector in self.e2_detector[idx_2]:
                        raise AttributeError("Detector shared in multple lanes, add node in between or remove detector.")
    
    def get_phase_duration(self, trafficlight):
        """Gets the phase duration of a traffic light(excluding transition phase e.g. 'yellow')"""
        phases = traci.trafficlight.getCompleteRedYellowGreenDefinition(trafficlight)[0].getPhases()
        duration = [i.duration for i in phases if 'y' not in i.state]
        return duration

    def get_state(self, trafficlight):
        """Gets the state of the trafficlight"""

        #Get action of neighbors(joint-action)
        tls_dict = self.tls[trafficlight]
        neighbors = tls_dict['neighbors']
        joint_action = []
        for neighbor in neighbors:
            joint_action.append(self.get_phase_duration(neighbor))
        joint_action = np.hstack(joint_action)/self.cycle_length   

        #Get ID list of detectors
        e2_detectors = self.e2_detectors[trafficlight]
        
        #Get Queues of each detector
        queues = [traci.lanearea.getLastStepVehicleNumber(detector) for detector in e2_detectors] #gets the queus in detectors

        #get current Phase ID
        tl_phase = traci.trafficlight.getPhase(trafficlight)

        #return state (queues, ohe of tl phase, ordinal joint action of neighbors)
        phase_durations = np.array(self.get_phase_duration(trafficlight))/self.cycle_length

        arry = np.hstack([queues, phase_durations, joint_action])

        return arry

    def take_action(self, trafficlight, q_values):
        #create a distribution of phase duration based from the q_values
        sum_q_values = np.sum(q_values)
        percentage = q_values/sum_q_values
        phase_time = percentage*self.cycle_length
        
        #set the phase duration
        tls_dict = self.tls[trafficlight]
        for idx, duration in enumerate(phase_time):
            phase_idx = tls_dict['phases'][idx]
            traci.trafficlight.setPhase(trafficlight, phase_idx)
            traci.trafficlight.setPhaseDuration(trafficlight, duration)
        traci.trafficlight.setPhase(trafficlight, 0) #resets the trafficlight
    
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
            item['neighbors'] = val
        
            
    def randomize_state(self):
        for trafficlight in traci.trafficlight.getIDList():
            randomized_action = random.randint(0,self.tls[trafficlight]['total_phases'] - 1)
            traci.trafficlight.setPhase(trafficlight,randomized_action)
            
    def init_neural_net(self):
        """Initializes the neural network of each ITS"""
        for trafficlight in traci.trafficlight.getIDList():
            states_length = len(self.get_state(trafficlight))
            tls_dict = self.tls[trafficlight]
            total_phases = tls_dict['total_phases']
            tls_dict['agent'] = tools.Net(states_length,total_phases)


    