from pathlib import Path
import torch
import torch.nn as nn
import torch.optim
import numpy as np
import traci
import os
import sys
import random

from sumolib import checkBinary  # noqa
import traci  # noqa
import sumolib
from aux_files import tools

DEBUG = False
class SumoEnvironment:
    """Environment for MADDPG"""
    def __init__(self, gui = True, buffer_size = 10, buffer_yellow = 3, train=False,
                dir=Path("Simulation_Environment\Main MADDPG"), cycle_length=120, simulation_time=57600, simulation_time_tol=0.2):
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

        #Set Simulation limit
        self.simulation_time = simulation_time
        self.simulation_time_tol = simulation_time_tol + 1

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

        # #Set neighbor Limit #TODO: Delete this
        # self.neighbor_limit = neighbor_limit

        #Create Dict for each trafficlight
        self.init_tls_properties()

        #Get Phase Data of Each Traffic Light
        self.get_phase_data()

        #Init yellow transition
        self.init_yellow_transition()

        #Randomize state
        self.reset_phase()
        
        #inits vehicle counter
        for trafficlight in traci.trafficlight.getIDList():
            tls_dict = self.tls[trafficlight]
            tls_dict['vehicle speed'] = []
            tls_dict['lane queue'] = {}
        #     tls_dict['initial waiting time'] = {}
        #     tls_dict['current waiting time'] = {}
        
        #Init Values for e2 detectors record
        self.init_e2_records()
        
        print("Environment initialized")
    
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
    
    def init_tls_properties(self):
        """initializes the properties of each tls"""
        self.tls = {}
        all_tls = traci.trafficlight.getIDList()
        
        #Create a dictionary for each tls
        for tls in all_tls:
            self.tls[tls] = {}

        # #Add neighbour list to tls dict
        # neighbors = tools.get_neighbors(2,self.net)
        # for key,val in neighbors.items(): 
        #     item = self.tls[key]
        #     item['neighbors'] = val
    
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
            traci.trafficlight.setPhase(trafficlight, 0)

    def reset_phase(self):
        for trafficlight in traci.trafficlight.getIDList():
            traci.trafficlight.setPhase(trafficlight,self.tls[trafficlight]['phases'][0])
    
    def init_e2_records(self):
        """inits the e2 records of all tls"""
        all_tls = traci.trafficlight.getIDList()
        for trafficlight in all_tls:
            tls_dict = self.tls[trafficlight]
            detector_dct = tls_dict['lane queue']
            e2_detectors = self.e2_detectors[trafficlight]
            
            for detector in e2_detectors:
                detector_dct[detector] = [0]
    
    def get_phase_duration(self, trafficlight):
        """Gets the phase duration of a traffic light(excluding transition phase e.g. 'yellow')"""
        phases = traci.trafficlight.getCompleteRedYellowGreenDefinition(trafficlight)[0].getPhases()
        duration = [i.duration for i in phases if 'y' not in i.state]

        # if trafficlight == traci.trafficlight.getIDList()[0] and DEBUG:
        #     print('Phases', duration, trafficlight)
        return duration

    def get_reward(self, trafficlight):
        # Average Speed
        # durations = self.get_phase_duration(trafficlight)
        # punishment = 0
        
        # for duration in durations:
        #     if duration < 15:   #below minimum green time
        #         punishment += -2
        
        tls_dict    = self.tls[trafficlight]
        # all_speed   = tls_dict["vehicle speed"]
        
        # population = len(all_speed)
        # all_speed = np.array(all_speed)
        # speed_max = all_speed.max()
        # speed_max_ratio = all_speed/speed_max
        # speed_max_sum = speed_max_ratio.sum()

        # #reset vehicle speed counter
        # tls_dict = self.tls
        tls_dict['vehicle speed'] = []
        # reward = (1/population)*speed_max_sum + punishment

        #Waiting Time
        tls_dict = self.tls[trafficlight]
        waiting_time_dct = tls_dict['waiting time']

        waiting_times = list(waiting_time_dct.values())
        waiting_time_normalized = np.array(waiting_times)/max(waiting_times)
        waiting_time_normalized = waiting_time_normalized.sum()
        reward = -waiting_time_normalized

        #reset waiting time counter
        waiting_time_dct = {}

        return reward
    
    def record(self,trafficlight):
        tls_dict = self.tls[trafficlight]
        all_speed = []
        e2_detectors = self.e2_detectors[trafficlight]
        detector_dct = tls_dict['lane queue']

        #Records average speed, lane occupancy, and waiting time
        for detector in e2_detectors:
            #Average speed
            speed = traci.lanearea.getLastStepMeanSpeed(detector)
            if speed < 0:
                speed = 0
            all_speed.append(speed)

            #Average speed
            lane_occupancy = traci.lanearea.getLastStepOccupancy(detector)
            #If not initialized, initialize detector dct
            if len(detector_dct) == 0:
                for detector in e2_detectors:
                    detector_dct[detector] = []
            
            #Waiting Time #TODO: Check if it works in the simulation

            all_vehicles = traci.lanearea.getLastStepVehicleIDs(detector)
            
            try:        #Create waiting time dictionary in tls_dict if not available
                waiting_time_dict = tls_dict['waiting time']
            except KeyError:
                tls_dict['waiting time'] = {}
                waiting_time_dict = tls_dict['waiting time'] = {}

            for vehicle in all_vehicles:
                current_waiting_time = traci.vehicle.getWaitingTime(vehicle)
                try:
                    previous_time = waiting_time_dict[vehicle]
                except KeyError:
                    previous_time = 0
                
                if previous_time <= current_waiting_time:
                    waiting_time = current_waiting_time
                    waiting_time_dict[vehicle] = waiting_time
                else:
                    continue    #does not add the vehicle if the previous time is less than the current (Either it left the simulation or it is already moving. It keeps the max waiting time)


            detector_dct[detector].append(lane_occupancy)

        average_speed = sum(all_speed)/len(all_speed)
        tls_dict['vehicle speed'].append(average_speed)
            

    def get_state(self, trafficlight):
        """Gets the state of the trafficlight"""  

        tls_dict = self.tls[trafficlight]
        
        #Get the average queue
        queues = np.array([])
        for vals in tls_dict['lane queue'].values():
            mean_vals = np.array(vals).mean()
            queues = np.hstack([mean_vals, queues])

        #return state (queues, ohe of tl phase, ordinal joint action of neighbors)
        phase_durations = np.array(self.get_phase_duration(trafficlight))/self.cycle_length
        # if trafficlight == traci.trafficlight.getIDList()[0] and DEBUG:
        #     print("Problematic", phase_durations, trafficlight)
        #     print("Problematic Logic", traci.trafficlight.getAllProgramLogics(trafficlight)[0], trafficlight)
        arry = np.hstack([queues, phase_durations])

        return arry.astype('float64')

    def set_action(self, trafficlight, q_values):
        """Actions must be in sigmoid (e.g. [0.1, 0.6,0.9,0.4])"""

        if trafficlight == traci.trafficlight.getIDList()[0] and DEBUG:
            print("Q Value", q_values, trafficlight)

        tls_dict = self.tls[trafficlight]
        no_of_phases = tls_dict['total_phases']
        q_values = np.reshape(q_values,-1)

        tls_dict = self.tls[trafficlight]

        #create a distribution of phase duration based from the q_values
        sum_q_values = np.sum(q_values)
        percentage = q_values/sum_q_values
        total_avail_green = self.cycle_length - len(tls_dict['transition phases'])*self.buffer_yellow   #subtracts the cycle length by the total time of the transition phases
        phase_time = percentage*total_avail_green


        #Edit the Program Logic
        logic = traci.trafficlight.getAllProgramLogics(trafficlight)[0]
        for idx, duration in enumerate(phase_time):
            phase_idx = tls_dict['phases'][idx]
            phase = logic.phases[phase_idx]
            phase.duration = duration
            phase.minDur = duration
            phase.maxDur = duration
        traci.trafficlight.setProgramLogic(trafficlight, logic)
        
        if trafficlight == traci.trafficlight.getIDList()[0] and DEBUG:
            print("Phase Time", phase_time, trafficlight)
            # print("Logic", traci.trafficlight.getAllProgramLogics(trafficlight)[0], logic, trafficlight)

        traci.trafficlight.setPhase(trafficlight, 0) #resets the trafficlight
    
    def is_done(self):
        if traci.simulation.getMinExpectedNumber() == 0:
            return True
        elif traci.simulation.getTime() > self.simulation_time*self.simulation_time_tol:
            return True
        else:
            return False

    def step(self, actions, all_tls):
        #  """Simulates the environment by the cycle length and records the 
        # observation to the tls dict"""
        
        for action, trafficlight in zip(actions, all_tls):
            self.set_action(trafficlight,action)

        for seconds in range(self.cycle_length):
            traci.simulation.step()
            for trafficlight in all_tls:
                self.record(trafficlight)
    
    def obs(self, trafficlight):
        """Returns the state, reward, and done"""
        new_state = self.get_state(trafficlight)
        reward = self.get_reward(trafficlight)
        done = self.is_done()
        tls_dict = self.tls[trafficlight]

        #resets the detector queue and vehicle speed
        tls_dict['vehicle speed'] = []
        queue = tls_dict['lane queue']
        for key, val in queue.items():
            queue[key] = []

        return new_state, reward, done
    
    def reset(self):
        traci.close()
        traci.start([self.sumoBinary, "-c", f"{self.dir}\osm.sumocfg",
                             "--tripinfo-output", f"{self.dir}\\Results\\tripinfo.xml",  "--start"])
        #Create dictionary for e2 detectors in TLS using trafficlightID as key
        self.get_e2_detectors()

        #Create Dict for each trafficlight
        self.init_tls_properties()

        #Get Phase Data of Each Traffic Light
        self.get_phase_data()

        #Init yellow transition
        self.init_yellow_transition()

        for trafficlight in traci.trafficlight.getIDList():
            tls_dict = self.tls[trafficlight]
            tls_dict['vehicle speed'] = []
            tls_dict['lane queue'] = {}
            tls_dict['initial waiting time'] = {}
            tls_dict['current waiting time'] = {}

        #Init Values for e2 detectors record
        self.init_e2_records()

        #Randomize state
        self.reset_phase()
    
    def get_phase_no(self, tls):
        tls_dict = self.tls[tls]
        phase_no = tls_dict['total_phases']
        return phase_no

    def close(self):
        traci.close()
