#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:49:23 2019

@author: picost
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 16, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.n_lin = 56 * 56 * 16
        self.lin1 = nn.Linear(self.n_lin, 1000)
        self.lin2 = nn.Linear(1000, 108)
        self.drop = nn.Dropout(p=0.1)
        self.drop2d = nn.Dropout2d(p=0.2)
    
    def forward(self, x):
        ## Define forward behavior
        # first conv and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop2d(x)
        # second conv layer
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.n_lin)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.log_softmax(x, dim=1)
        return x