import time
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
from aux_files.custom_maddpg import MADDPG, ActorNetwork, CriticNetwork, Agent
from aux_files.SUMOEnvironment import SumoEnvironment

torch.autograd.set_detect_anomaly(True)

#Hyperparameters of Model
ALPHA = 0.01    #Learning Rate of Actor
BETA =  0.01    #Learning Rate of Critic
FC1 = 64        #First Hidden Layer Size
FC2 = 80        #Second Hidden Layer Size
GAMMA = 0.95
CHKPT_DIR = Path("DDPG_Models")
TAU = 0.01
EPOCH = 50
LEARN_INT = 10

#Buffer Parameters
BATCH_SIZE=64
MAX_SIZE=100

#Parameters for Sumo Environment
BUFFER_YELLOW = 4   #4 seconds for transition
DIR = Path("Simulation_Environment\Main MADDPG")
CYCLE_LENGTH = 120  #Cycle Length

#Train or Evaluate bool
EVALUATE = False

#Other Params
PRINT_INTERVAL = 100
GUI = False

def get_all_states(env, trafficlights):
    states = np.array([])
    for tls in trafficlights:
        state = env.get_state(tls)
        states = np.concatenate([states, state])
    
    return states

def init_agent_old(env):
    """Returns TLS IDs and MADDPG Learner Class"""
    all_actors = []
    all_target_actors =[]
    all_critics = []
    all_target_critics = []
    all_tls = traci.trafficlight.getIDList()
    max_no_phases = env.get_max_phase_no()

    for trafficlight in all_tls:
        tls_dict = env.tls[trafficlight]

        state = env.get_state(trafficlight)
        tls_dict = env.tls[trafficlight]
        phase_no = tls_dict['total_phases']
        if phase_no != max_no_phases:
            padded_state = np.pad(state, (0, max_no_phases-phase_no)).tolist()
            input_dims = len(padded_state)
        else:
            input_dims = len(state)

        n_actions = max_no_phases
        tls_actor = ActorNetwork(alpha=ALPHA, input_dims=input_dims, fc1_dims=FC1,
                                fc2_dims=FC2, n_actions=n_actions, name=trafficlight+'_actor', 
                                chkpt_dir=CHKPT_DIR)
        target_tls_actor = ActorNetwork(alpha=ALPHA, input_dims=input_dims, fc1_dims=FC1,
                                fc2_dims=FC2, n_actions=n_actions, name=trafficlight+'_targetActor', 
                                chkpt_dir=CHKPT_DIR)
        
        all_actors.append(tls_actor)
        all_target_actors.append(target_tls_actor)
    
    all_actions = int(env.get_max_phase_no() * len(all_tls))
    
    critic_dims=0
    for actor in all_actors:
        critic_dims = actor.input_dims + critic_dims

    for trafficlight in all_tls:
        #Includes the padded verision as input
        state = env.get_state(trafficlight)
        tls_dict = env.tls[trafficlight]
        phase_no = tls_dict['total_phases']
        if phase_no != max_no_phases:
            padded_state = np.pad(state, (0, max_no_phases-phase_no)).tolist()
            input_dims = len(padded_state) + all_actions
        else:
            input_dims = len(state) + all_actions


        tls_critic = CriticNetwork(beta=BETA, input_dims=critic_dims, fc1_dims=FC1,
                                fc2_dims=FC2,all_actions=all_actions, name=trafficlight+'_critic', chkpt_dir=CHKPT_DIR)
        target_tls_critic = CriticNetwork(beta=BETA, input_dims=critic_dims, fc1_dims=FC1,
                                fc2_dims=FC2,all_actions=all_actions, name=trafficlight+'_targetCritic', chkpt_dir=CHKPT_DIR)
        all_critics.append(tls_critic)
        all_target_critics.append(target_tls_critic)

    all_models = zip(all_actors, all_target_actors, all_critics, all_target_critics)
    agents = []
    for actor, target_actor, critic, target_critic in all_models:
        agent = Agent(actor, target_actor, critic, target_critic, gamma=GAMMA, tau=TAU)
        agents.append(agent)
    
    maddpg_agents = MADDPG(agents)
    return all_tls, maddpg_agents

def init_agent(env):
    all_tls = traci.trafficlight.getIDList()
    n_agents = len(all_tls)
    
    actor_dims = [env.get_state(tls).shape[0] for tls in all_tls]
    critic_dims = sum(actor_dims)
    n_actions = [env.get_phase_no(tls) for tls in all_tls]
    

    maddpg_agent = MADDPG(actor_dims, critic_dims, n_agents, n_actions, CHKPT_DIR, all_tls, BATCH_SIZE,ALPHA,
                        BETA, FC1, FC2, GAMMA, TAU)

    memory = deque(maxlen=MAX_SIZE)

    return all_tls, maddpg_agent, memory
    

def get_obs(env, trafficlights):
    """returns the observation of each actor 
        dim = [actor, state]"""
    obs = []
    for trafficlight in trafficlights:
        tls_obs = env.get_state(trafficlight)
        obs.append(tls_obs)
    obs = np.array(obs, dtype='object')
    return obs


def get_rewards(env, trafficlights):
    rewards = []
    for trafficlight in trafficlights:
        tls_rewards = env.get_reward(trafficlight)
        rewards.append(tls_rewards)
    
    rewards = np.array(rewards)
    return rewards

def simulation_step(env, actions, all_tls, n_agents):
    "Moves the simulation and returns the new_state, reward, done"
    env.step(actions, all_tls)
    obs = get_obs(env, all_tls)
    reward = get_rewards(env, all_tls)
    reward = reward.astype('float64')
    done = [env.is_done()]*n_agents

    return obs, reward, done

def state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs]).astype('float64')

    return state

if __name__ == '__main__':
    #init envi, maddpg agent, and memory
    env = SumoEnvironment(gui=GUI, buffer_yellow=BUFFER_YELLOW, dir=DIR, cycle_length=CYCLE_LENGTH)
    all_tls, maddpg_agent, memory = init_agent(env)
    
    total_steps = 0
    score_history = []
    best_score = 0
    n_agents = len(all_tls)
    lst_of_rewards = [] #debugging

    if EVALUATE:
        maddpg_agent.load_checkpoint()
        EPOCH = 1
        env.close()
        print("RESTARTING IN EVAL MODE...")
        time.sleep(1)
        env = SumoEnvironment(gui=True, buffer_yellow=BUFFER_YELLOW, dir=DIR, cycle_length=CYCLE_LENGTH)
        print("APPLICATION IN EVAL MODE")
        done = [False]*n_agents

    for i in range(EPOCH):
        env.reset()
        obs = get_obs(env,all_tls)
        score = 0
        done = [env.is_done()]*n_agents
        episode_step = 0
        print("\n\nNow running game no. ", i)
        
        while not any(done):
            if episode_step%10 == 0:
                print(f"Currently in episode {episode_step}")
            
            actions = maddpg_agent.choose_action(obs)
            obs_ , reward, done = simulation_step(env,actions,all_tls,n_agents)

            states = state_vector(obs)
            states_ = state_vector(obs_)

            memory.append([obs, states, actions, reward, obs_, states_, done])

            if total_steps % LEARN_INT == 0 and not EVALUATE:
                maddpg_agent.learn(memory)
                cur_vehicles = traci.vehicle.getIDCount()
                print(f"Vehicles in simulation:{cur_vehicles}\n")

            obs = obs_

            score += sum(reward)
            total_steps += 1
            episode_step += 1
            lst_of_rewards.append(reward)


        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        with open("rewards.csv", "w") as f:
            for idx, score in enumerate(lst_of_rewards):
                f.write(f"{idx},{score}\n")
        if not EVALUATE:
            if avg_score > best_score:
                maddpg_agent.save_checkpoint()
                best_score = avg_score
            with open("average.csv", "w") as f:
                for idx, score in enumerate(score_history):
                    f.write(f"{idx},{score}\n")
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))