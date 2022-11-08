
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim
import numpy as np
import traci

from collections import deque

import torch.nn.functional as F

from sumolib import checkBinary  # noqa
import traci  # noqa
import sumolib
from aux_files.MADDPG import MADDPG, ActorNetwork, CriticNetwork, Agent, MultiAgentReplayBuffer
from aux_files.SUMOEnvironment import SumoEnvironment

#Hyperparameters of Model
ALPHA = 0.01
BETA =  0.01
FC1 = 64
FC2 = 64
GAMMA = 0.95
CHKPT_DIR = Path("DDPG_Models")
TAU = 0.01
EPOCH = 50

#Buffer Parameters
BATCH_SIZE=64
MAX_SIZE=100

#Parameters for Sumo Environment
BUFFER_YELLOW = 5
DIR = Path("Simulation_Environment\Main MADDPG")
CYCLE_LENGTH = 120

#Train or Evaluate bool
EVALUATE = False

env = SumoEnvironment()

def get_all_states(env):
    traffic_lights = traci.trafficlight.getIDList()
    states = np.array([])
    for tls in traffic_lights:
        state = env.get_state(tls)
        states = np.concatenate([states, state])
    
    return states

def init_agent(env):
    """Returns TLS IDs and MADDPG Learner Class"""
    all_actors = []
    all_target_actors =[]
    all_critics = []
    all_target_critics = []
    all_tls = traci.trafficlight.getIDList()

    all_actions = 0
    for trafficlight in all_tls:
        tls_dict = env.tls[trafficlight]
        input_dims = len(env.get_state(trafficlight))
        n_actions = len(tls_dict['phases'])
        tls_actor = ActorNetwork(alpha=ALPHA, input_dims=input_dims, fc1_dims=FC1,
                                fc2_dims=FC2, n_actions=n_actions, name=trafficlight+'_actor', 
                                chkpt_dir=CHKPT_DIR)
        target_tls_actor = ActorNetwork(alpha=ALPHA, input_dims=input_dims, fc1_dims=FC1,
                                fc2_dims=FC2, n_actions=n_actions, name=trafficlight+'_targetActor', 
                                chkpt_dir=CHKPT_DIR)
        
        all_actors.append(tls_actor)
        all_target_actors.append(target_tls_actor)
        all_actions += n_actions
    
    for trafficlight in all_tls:
        tls_dict = env.tls[trafficlight]
        input_dims = len(env.get_state(trafficlight)) + all_actions
        tls_critic = CriticNetwork(beta=BETA, input_dims=input_dims, fc1_dims=FC1,
                                fc2_dims=FC2, name=trafficlight+'_critic', chkpt_dir=CHKPT_DIR)
        target_tls_critic = CriticNetwork(beta=BETA, input_dims=input_dims, fc1_dims=FC1,
                                fc2_dims=FC2, name=trafficlight+'_targetCritic', chkpt_dir=CHKPT_DIR)
        all_critics.append(tls_critic)
        all_target_critics.append(target_tls_critic)

    all_models = zip(all_actors, all_target_actors, all_critics, all_target_critics)
    agents = []
    for actor, target_actor, critic, target_critic in all_models:
        agent = Agent(actor, target_actor, critic, target_critic, gamma=GAMMA, tau=TAU)
        agents.append(agent)
    
    maddpg_agents = MADDPG(agents)
    return all_tls, maddpg_agents

def get_obs(env, trafficlights):
    """returns the observation"""
    obs = []
    for trafficlight in trafficlights:
        tls_obs = env.get_state(trafficlight)
        obs.append(tls_obs)
    
    obs = np.array(obs, dtype=object)
    return obs

def simulation_step(env, all_tls):
    "Moves the simulation and returns the new_state, reward, done"
    

if __name__ == '__main__':
    env = SumoEnvironment(gui=False, buffer_yellow=BUFFER_YELLOW, dir=DIR, cycle_length=CYCLE_LENGTH)
    all_tls, maddpg_agent = init_agent(env)
    
    actor_dims = [len(env.tls[trafficlight]['phases']) for trafficlight in all_tls]
    all_states_length = sum(actor_dims)
    n_agents = len(actor_dims)
    
    memory = MultiAgentReplayBuffer(MAX_SIZE, all_states_length, actor_dims, all_states_length,
                                    n_agents, BATCH_SIZE)
    
    if EVALUATE:
        maddpg_agent.load_checkpoint()
        EPOCH = 1
        env.close()
        env = SumoEnvironment(gui=True, buffer_yellow=BUFFER_YELLOW, dir=DIR, cycle_length=CYCLE_LENGTH)

    for i in range(EPOCH):
        env.reset()
        obs = get_obs(env,all_tls)
        score = 0
        done = env.is_done()
        episode_step = 0
        
        while not done:
            actions = maddpg_agent.choose_action(obs)




        




    

        


    






    # def init_neural_net(self):
    #     """Initializes the neural network of each ITS"""
    #     for trafficlight in traci.trafficlight.getIDList():
    #         states_length = len(self.get_state(trafficlight))
    #         tls_dict = self.tls[trafficlight]
    #         total_phases = tls_dict['total_phases']
    #         tls_dict['agent'] = tools.Net(states_length,total_phases)


    