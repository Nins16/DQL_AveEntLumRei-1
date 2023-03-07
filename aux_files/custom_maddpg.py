import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from pathlib import Path

DEBUG = False

#Actor Network
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, 
                 n_actions, name, chkpt_dir):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = F.relu(self.pi(x))

        return pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))
    pass

#Critic Network
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, 
                total_actions,name, chkpt_dir):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims+total_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
 
        self.to(self.device)

    def forward(self, state, action):
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class Agent:
    """actor_dims:  List of states per agent
    critic_dims: get the dims for all critics. Note: all critics have same no of """
    def __init__(self, actor_dims, critic_dims, n_actions, total_actions, name, chkpt_dir,
                    alpha=0.01, beta=0.01, fc1=64, 
                    fc2=64, gamma=0.95, tau=0.01):
        self.n_actions      = n_actions
        self.chkpt_dir      = chkpt_dir
        self.gamma          = gamma
        self.tau            = tau
        self.name           = name


        self.actor          = ActorNetwork(alpha,actor_dims,fc1,fc2,n_actions,f"actor_{self.name}",chkpt_dir)
        self.target_actor   = ActorNetwork(alpha,actor_dims,fc1,fc2,n_actions,f"target_actor_{self.name}",chkpt_dir)

        self.critic         = CriticNetwork(beta,critic_dims,fc1,fc2,total_actions,f"critic_{self.name}",chkpt_dir)
        self.target_critic  = CriticNetwork(beta,critic_dims,fc1,fc2,total_actions,f"target_critic_{self.name}",chkpt_dir)
        self.update_network_parameters(self.tau)
        self.MSELoss = nn.MSELoss()

    
    def choose_action(self, observation):
        observation = np.array([observation])   #TODO: Possible error below
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)

        actions = self.actor.forward(state)
        noise = T.rand(self.n_actions).to(self.actor.device)
        action = actions + noise

        return action.detach().cpu().numpy()[0]

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
    


class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, chkpt_dir, names, batch_size, alpha=0.01, beta=0.01, fc1=64, fc2=64, gamma=0.9, tau=0.01):
        self.batch_size = batch_size
        self.agents = []

        for idx in range(n_agents):
            agent = Agent(actor_dims[idx], critic_dims, n_actions[idx], sum(n_actions), names[idx], chkpt_dir, alpha, beta, fc1 ,fc2 , gamma, tau)
            self.agents.append(agent)

            loss_dir = Path(f'loss_logs/{agent.name}_loss.csv')
            with open(loss_dir, 'w') as f:
                f.write('Actor Loss,Critic Loss\n')
                agent.loss_dir = loss_dir

        self.MSELoss = nn.MSELoss()
        

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)
        return actions
    
    def three_d_memory_processor(self, arry, agent_idx):
        """Isolates the data solely for the respective agent \n
        e.g.\n
        [0,0,0,1,0,0],\n
        [0,0,0,1,0,0],\n
        [0,0,0,1,0,0]\n
        ->\n
        [1,1,1]"""
        agent_arry = []
        for episode in arry:
            agent_arry.append(episode[agent_idx])
        return np.array(agent_arry)
        
    #Problem with gradients. Use new to individually set the tensors
    def learn(self, memory):
        if len(memory) < self.batch_size:
            return

        #obs -> states per agent [list of np arrays]
        #state -> all states [np array] 
        #actions -> actions per Agent [list of np arrays]
        #done -> done of all [np array]

        # obs, states, actions, reward, obs_, states_, done = random.sample(memory,self.batch_size)
        minibatch = random.sample(memory,self.batch_size)
        obs = [o for o,s,a,r,o_,s_,d in minibatch]
        states = [s for o,s,a,r,o_,s_,d in minibatch]
        actions = [a for o,s,a,r,o_,s_,d in minibatch]
        reward = [r for o,s,a,r,o_,s_,d in minibatch]
        obs_ = [o_ for o,s,a,r,o_,s_,d in minibatch]
        states_ = [s_ for o,s,a,r,o_,s_,d in minibatch]
        dones = [d for o,s,a,r,o_,s_,d in minibatch]

        device = self.agents[0].actor.device

        
        states  = T.tensor(states, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones   = T.tensor(dones).to(device)
        rewards = T.tensor(reward, dtype=T.float).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            #get new actions
            #Very problematic, reworking this to get actions based on new just the action batch
            #such that it pass through the network only once
            with T.no_grad():
                #new states and new pi is okay
                new_states = self.three_d_memory_processor(obs_, agent_idx)
                new_states = T.tensor(new_states, dtype=T.float).to(device)
                
                new_pi = agent.target_actor.forward(new_states)
                all_agents_new_actions.append(new_pi)

                action = self.three_d_memory_processor(actions, agent_idx)
                action = T.tensor(action, dtype=T.float).to(device)
                old_agents_actions.append(action)

                mu_states = self.three_d_memory_processor(obs, agent_idx)
                mu_states = T.tensor(mu_states, dtype=T.float).to(device)
                pi = agent.actor.forward(mu_states)
                all_agents_new_mu_actions.append(pi)


        for agent_idx, agent in enumerate(self.agents):
            new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1).to(device)
            mu = T.cat([mu for mu in all_agents_new_mu_actions], dim=1).to(device)
            old_actions  = T.cat([acts for acts in old_agents_actions], dim=1).to(device)

            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
            critic_value_[dones[:,0]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten()

            target = rewards[:,agent_idx] + agent.gamma*critic_value_
            critic_loss = self.MSELoss(target.detach(), critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss)

            if DEBUG and agent_idx==0:
                print('States and Mu\n', states, mu)
                print("Actor Loss", actor_loss)

            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            with open(agent.loss_dir, 'a') as f:
                f.write(f"{actor_loss},{critic_loss}\n")

            agent.update_network_parameters()
        print(f"Update finished in Agents")

    
    # def learn(self, memory):
    #     minibatch = random.sample(memory,self.batch_size)
    #     obs = [o for o,s,a,r,o_,s_,d in minibatch]
    #     states = [s for o,s,a,r,o_,s_,d in minibatch]
    #     actions = [a for o,s,a,r,o_,s_,d in minibatch]
    #     reward = [r for o,s,a,r,o_,s_,d in minibatch]
    #     obs_ = [o_ for o,s,a,r,o_,s_,d in minibatch]
    #     states_ = [s_ for o,s,a,r,o_,s_,d in minibatch]
    #     dones = [d for o,s,a,r,o_,s_,d in minibatch]

    #     for agent


        

        
            
        