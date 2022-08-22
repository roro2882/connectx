from batch_env import batchEnv
import numpy as np
from gym_connect import gym_connect
import torch
from strategies import SoftMaxStrategy
from networks import FCQ
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



config = {'rows':6,'columns':7,'inarow':4,'agent':None}


def printstate(state):
    print(state[0]+2*state[1])

def bagent(state):
    printstate(state[0])
    if path is not None:
        q_values = network(state).detach().cpu().data.numpy()
        print(q_values)
        action = strategy.select_action_from_q_values(q_values,batch=True, debug=True)[0]
    else:
        action = input('Action Joueur 1 ? ')
    print(action)
    return np.array([[int(action)]])

env = batchEnv(lambda : gym_connect(config),1,bagent)
path = "checkpoints/model.67soft_hardrelu.tar"
if path is not None:
    dummy_env = gym_connect(config)
    network = FCQ(dummy_env.observation_space,config['columns'],(0,128,64), activation_fc=F.relu)
    print('retrieving checkpoint',path)
    dictionarysave = torch.load(path)
    network.load_state_dict(dictionarysave['state_dict'])
    print('total_step : ',dictionarysave['total_step'])
    strategy = SoftMaxStrategy(0.05,0.05,1,20000000)



for e in range(10):
    s = env.reset()
    done = False
    while not done:
        printstate(s[0])
        action = input('Action Joueur 0 ? ')
        s, r, done, _ = env.step(np.array([[int(action)]]))
