import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim
import numpy as np
import optparse
import os
import sys
import pandas as pd
from tqdm import tqdm
from collections import deque
import random
import torch.nn.functional as F

from sumolib import checkBinary  # noqa
import traci  # noqa
import sumolib
from collections import deque

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
    def __init__(self, tlsID, memsize):
        self.tlsID = tlsID
        self.replay = deque(maxlen=memsize) #Instant's replay experience
    
    def init_model(self,neighbors):
        #state length is 
        pass
    def train(self):
        pass

#This is used for the DDOG Network
class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2,dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()
    
    def __call__(self):
        x = self.x_prev + self.theta*(self.mu - self.x_prev) * self.dt + \
            self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        self.x_prev = x
    
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir = 'tmp/ddpg'):
        super(CriticNetwork,self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir,name+'ddpg')
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        f3 = 0.003
        self.q = nn.Linear(self.fc2_dims,1)
        torch.nn.init.uniform_(self.q.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.q.bias.data, -f3, f3)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)
    
    def forward(self,state,action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)

        state_value = self.bn2(state_value)
        state_value = self.fc2(state_value)

        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(torch.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value
    
    def save_checkpoint(self):
        print('Saving neural net to checkpoint...')
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        print('Loading neural net...')
        self.load_state_dict(torch.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir,name+'ddpg')
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        f3 = 0.003
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        torch.nn.init.uniform_(self.mu.weight.data(), -f3, f3)
        torch.nn.init.uniform_(self.mu.bias.data(), -f3, f3)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.sigmoid(self.mu(x))

    def save_checkpoint(self):
        print('Saving neural net to checkpoint...')
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        print('Loading neural net...')
        self.load_state_dict(torch.load(self.checkpoint_file))  

class DDPGAGent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, n_actions=4, max_size=1e6, layer1_size=400, layer2_size=300, batch_size=24):
        self.gamma= gamma
        self.tau = tau
        self.replay = deque(maxlen=max_size)
        self.batch_size = batch_size
        
        self.actor          = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions=n_actions, name="Actor")
        self.target_actor   = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions=n_actions, name="TargetActor")
        self.critic         = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=n_actions, name="Critic")
        self.target_critic  = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=n_actions, name="TargetCritic")

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.update_network_parameters(tau=1)
    
    def choose_action(self, observation):
        self.actor.eval()
        # self.target_actor.eval()
        # self.critic.eval()
        # self.target_critic.eval()

        observation = torch.tensor(observation, dtype=torch.float)
        mu = self.actor(observation).to(self.actor.device)
        mu_prime = mu + torch.tensor(self.noise(), dtype=torch.float).to(self.actor.device)

        self.actor.train()
        return mu_prime.cpu().detach().numpy()
    
    def add_memory(self, state, action, reward, new_state, done):
        exp = [state, action, reward, new_state, done]
        self.replay.append(exp)
    
    def learn(self):
        if len(self.replay) > self.batch_size:
            minibatch = random.sample(self.replay, self.batch_size)
            state1_batch    = torch.cat([s1 for (s1,a,r,s2,d) in minibatch]).to(self.critic.device)
            action_batch    = torch.Tensor([a for (s1,a,r,s2,d) in minibatch]).to(self.critic.device)
            reward_batch    = torch.Tensor([r for (s1,a,r,s2,d) in minibatch]).to(self.critic.device)
            state2_batch    = torch.cat([s2 for (s1,a,r,s2,d) in minibatch]).to(self.critic.device)
            done_batch      = torch.Tensor([d for (s1,a,r,s2,d) in minibatch]).to(self.critic.device)

            self.target_actor.eval()
            self.target_crtic.eval()
            self.crtic.eval()

            target_actions = self.target_actor.forward(state2_batch)
            critic_value_ = self.target_critic.forward(state2_batch, target_actions)
            critic_value = self.critic.forward(state1_batch, action_batch)

            target = reward_batch + self.gamma*critic_value_*done_batch
            target = torch.tensor(target).to(self.critic.device)
            target = target.view(self.batch_size,1)

            self.critic.train()
            self.critic.optimizer.zero_grad()
            critic_loss = F.mse_loss(target, critic_value)
            critic_loss.backward()
            self.critic.optimizer.step()

            self.critic.eval()
            self.actor.optimizer.zero_grad()
            mu = self.actor.forward(state1_batch)
            self.actor.train()
            actor_loss = -self.critic.forward(state1_batch, mu)
            actor_loss = torch.mean(actor_loss)
            actor_loss.backward()
            self.actor.optimizer.step()

            self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + (1-tau)*target_critic_state_dict[name].clone()
        
        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)*target_actor_state_dict[name].clone()
        
        self.target_actor.load_state_dict(actor_state_dict)
    
    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()
    
    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()




#Tools
def get_neighbors(neighbour_limit,net):
    """Returns the neighbouring nodes"""
    def distance(x,y):  #Distance between two points in a cartesea plane (No Elevation)
        x_a = x[0]
        x_b = x[1]
        y_a = y[0]
        y_b = y[1]
        c   = np.sqrt(((x_a-y_a)**2)+((x_b-y_b)**2))
        return c

    all_nodes   = net.getNodes()
    tls_nodes   = [node for node in all_nodes if node.getType() == 'traffic_light']
    array       = np.zeros([len(tls_nodes)]*2)
    column_names    = [i.getID() for i in tls_nodes]
    df              = pd.DataFrame(array, columns=column_names, index=column_names)
    for row in column_names:
        for column in column_names:
            row_node    = net.getNode(row)
            column_node = net.getNode(column)
            row_coords  = row_node._coord
            column_coords = column_node._coord
            df.loc[row,column] = distance(row_coords,column_coords)
    distance_dict = {}
    for node in column_names:
        node_tl_name = node+'_tl'
        lst=[i+'_tl' for i in list(df[node].sort_values()[1:neighbour_limit+1].index)]
        distance_dict[node_tl_name] = lst
    
    return distance_dict

