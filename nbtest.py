from batch_env import batchEnv
import numpy as np
from gym_connect import gym_connect
import torch
from strategies import SoftMaxStrategy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



config = {'rows':6,'columns':7,'inarow':4,'agent':'rule'}
def printstate(state):
    print(state[0]+2*state[1])

env = gym_connect(config)


for e in range(10):
    s = env.reset()
    done = False
    while not done:
        printstate(s)
        action = input('Action Joueur 0 ? ')
        s, r, done, _ = env.step(int(action))
