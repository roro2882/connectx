import numpy as np
import utils
import torch

class SoftMaxStrategy():
    def __init__(self, 
                 init_temp=1.0, 
                 min_temp=0.3, 
                 exploration_ratio=0.8, 
                 max_steps=25000):
        self.t = 0
        self.init_temp = init_temp
        self.exploration_ratio = exploration_ratio
        self.min_temp = min_temp
        self.max_steps = max_steps
        self.exploratory_action_taken = False
        
    def _update_temp(self):
        temp = 1 - self.t / (self.max_steps * self.exploration_ratio)
        temp = (self.init_temp - self.min_temp) * temp + self.min_temp
        temp = np.clip(temp, self.min_temp, self.init_temp)
        self.t += 1
        self.temp = temp

    def choice(self,probs, batch=False):
        if not batch:
            x = np.random.rand()
            cum = 0
            i=0
            for i,p in enumerate(probs):
                cum += p
                if x < cum:
                    break
            return i
        else:
            pass

    def select_action_from_q_values(self, q_values, batch=False, debug=False):
        self._update_temp()
        exp = np.exp(q_values/self.temp).astype(float)
        probs = exp/np.sum(exp,axis=1,keepdims=True)
        if debug:
            print(probs)
        actions = utils.chooseActions(probs)
        if batch:
            return actions
        else:
            return actions[0]


    def select_action(self, model, state, batch=False, debug=False):
        if not batch:
            if debug:
                print(state[0]*1+state[1]*2)
            with torch.no_grad():
                q_values = model(state, drop=False).cpu().detach().data.numpy()
                if debug:
                    print(q_values)

                actions = self.select_action_from_q_values(q_values, debug=debug)
                if debug:
                    print("Action : ",actions[0])
                return actions
        else:
            with torch.no_grad():
                q_values = model(state, drop=False).cpu().detach().data.numpy()
                actions = self.select_action_from_q_values(q_values, batch=True)
                return actions


class GreedyStrategy():
    def __init__(self):
        self.exploratory_action_taken = False

    def select_action(self, model, state, debug=False):
        if debug:
            print(state[0]*1+state[1]*2)
        with torch.no_grad():
            q_values = model(state, drop=False).cpu().detach().data.numpy().squeeze()
            if debug:
                print(q_values)
            action = np.argmax(q_values)
            if debug:
                print("Action : ",action)
            return action

class eGreedyStrategy():
    temp = 0
    def __init__(self, epsilon, select_random_action):
        self.exploratory_action_taken = False
        self.epsilon = epsilon
        self.select_random_action = select_random_action

    def select_action(self, model, state, batch=False, debug=False):
        if not batch:
            if debug:
                print(state[0]*1+state[1]*2)
            r = np.random.random()
            if r>self.epsilon:
                with torch.no_grad():
                    q_values = model(state, drop=False).cpu().detach().data.numpy().squeeze()
                    if debug:
                        print(q_values)
                    action = np.argmax(q_values)
                    if debug:
                        print("Action : ",action)
                    return action
            else:
                return self.select_random_action(state)
        else:
            r = np.random.random()
            if r>self.epsilon:
                with torch.no_grad():
                    q_values = model(state, drop=False).cpu().detach().data.numpy().squeeze()
                    return np.argmax(q_values, axis=1,keepdims=True)
            else:
                ractions = [self.select_random_action(state[i]) for i in range(state.shape[0])]
                return np.array(ractions).reshape(-1,1)

