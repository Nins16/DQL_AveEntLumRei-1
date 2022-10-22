from pathlib import Path
import torch
import torch.nn as nn
import torch.optim
import numpy as np
import traci
import os
import sys
import pandas as pd
from tqdm import tqdm
import random

from sumolib import checkBinary  # noqa
import traci  # noqa
import sumolib
from aux_files import tools


class SumoEnvironment:
    def __init__(self, gui = True, buffer_size = 10, buffer_yellow = 3, train=False,
                dir=Path("Simulation_Environment\Main MADDPG"),neighbor_limit = 2, cycle_length=120):
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

        #Init yellow transition
        self.init_yellow_transition()

        #Randomize state
        self.randomize_state()
        
        #inits vehicle counter
        for trafficlight in traci.trafficlight.getIDList():
            tls_dict = self.tls[trafficlight]
            tls_dict['vehicle speed'] = []
            tls_dict['lane queue'] = {}
    
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
            tls_dict['transition phases'] = [idx for idx, phase in enumerate(phases_objects) if 'y' in phase.state]
            #Gets the numer of phases
            tls_dict['total_phases'] = len(tls_dict['phases'])
    
    def init_yellow_transition(self):
        """Iterates through every transition phase and sets it to the user indicated yellow transition length"""
        if self.buffer_yellow is None:
            pass
        for trafficlight in traci.trafficlight.getIDList():
            tls_dict = self.tls[trafficlight]
            yellow_phases = tls_dict['transition phases']
            for phase in yellow_phases:
                traci.trafficlight.setPhase(trafficlight, phase)
                traci.trafficlight.setPhaseDuration(trafficlight, self.buffer_yellow)
            traci.trafficlight.setPhase(trafficlight)

    def randomize_state(self):
        for trafficlight in traci.trafficlight.getIDList():
            randomized_action = random.randint(0,self.tls[trafficlight]['total_phases'] - 1)
            traci.trafficlight.setPhase(trafficlight,self.tls[trafficlight]['phases'][randomized_action])
    
    def get_phase_duration(self, trafficlight):
        """Gets the phase duration of a traffic light(excluding transition phase e.g. 'yellow')"""
        phases = traci.trafficlight.getCompleteRedYellowGreenDefinition(trafficlight)[0].getPhases()
        duration = [i.duration for i in phases if 'y' not in i.state]
        return duration

    def get_reward(self, trafficlight):
        durations = self.get_phase_duration(trafficlight)
        exceed_min_green = False
        for duration in durations:
            if duration < 15:
                exceed_min_green = True
        
        if exceed_min_green:
            return -10
        
        tls_dict    = self.tls[trafficlight]
        all_speed   = tls_dict["vehicle_speed"]
        population = len(all_speed)
        all_speed = np.array(all_speed)
        speed_max = all_speed.max()
        if speed_max == 0:
            return 0
        speed_max_ratio = all_speed/speed_max
        speed_max_sum = speed_max_ratio.sum()

        #reset vehicle speed counter
        tls_dict = self.tls
        tls_dict['vehicle speed'] = []
        return (1/population)*speed_max_sum
        
    
    def record(self,trafficlight):
        tls_dict = self.tls[trafficlight]
        all_speed = []
        e2_detectors = self.e2_detectors[trafficlight]

        for detector in e2_detectors:
            speed = traci.lanearea.getLastStepMeanSpeed(detector)
            if speed < 0:
                speed = 0
            all_speed.append(speed)
        average_speed = sum(all_speed)/len(all_speed)
        tls_dict['vehicle speed'].append(average_speed)

        detector_dct = tls_dict['lane queue']
        for detector in e2_detectors:
            lane_occupancy = traci.lanearea.getLastStepOccupancy(detector)
            
           #If not initialized, initialize detector dct
            if len(detector_dct) == 0:
                for detector in e2_detectors:
                    detector_dct[detector] = []
            
            detector_dct[detector].append(lane_occupancy)


    def get_state(self, trafficlight):
        """Gets the state of the trafficlight"""  

        tls_dict = self.tls[trafficlight]
        
        #Get the average queue
        # queues = [traci.lanearea.getLastStepVehicleNumber(detector) for detector in e2_detectors] #gets the queus in detectors
        queues = []
        for vals in tls_dict['lane queue'].values():
            queues.append(vals)
        queues = np.array(queues).mean(axis=0)

        #return state (queues, ohe of tl phase, ordinal joint action of neighbors)
        phase_durations = np.array(self.get_phase_duration(trafficlight))/self.cycle_length

        arry = np.hstack([queues, phase_durations])

        return arry

    def set_action(self, trafficlight, q_values):
        """Actions must be in sigmoid (e.g. [0.1, 0.6,0.9,0.4])"""

        #create a distribution of phase duration based from the q_values
        sum_q_values = np.sum(q_values.cpu().detach().numpy())
        percentage = q_values/sum_q_values
        phase_time = percentage*self.cycle_length
        
        #set the phase duration
        tls_dict = self.tls[trafficlight]
        for idx, duration in enumerate(phase_time):
            phase_idx = tls_dict['phases'][idx]
            traci.trafficlight.setPhase(trafficlight, phase_idx)
            traci.trafficlight.setPhaseDuration(trafficlight, duration-self.buffer_yellow)
        traci.trafficlight.setPhase(trafficlight, 0) #resets the trafficlight
    
    def is_done(self):
        return traci.simulation.getMinExpectedNumber() == 0


###Please Edit
    def step(self):
        """Simulates the environment by the cycle length and records the 
        observation to the tls dict"""
        for seconds in range(self.cycle_length):
            traci.simulation.step()
            for trafficlight in traci.trafficlight.getIDList():
                self.record(trafficlight)
        
    def obs(self, trafficlight):
        new_state = self.get_state(trafficlight)
        reward = self.get_reward(trafficlight)
        done = self.is_done()

        return new_state, reward, done