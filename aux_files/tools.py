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

