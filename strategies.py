import numpy as np
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
    def choice(self,probs):
        x = np.random.rand()
        cum = 0
        i=0
        for i,p in enumerate(probs):
            cum += p
            if x < cum:
                break
        return i
    def select_action(self, model, state):
        self.exploratory_action_taken = False
        
        self._update_temp()
        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy().squeeze()
            scaled_qs = q_values/self.temp
            norm_qs = scaled_qs - scaled_qs.max()            
            e = np.exp(norm_qs)
            probs = e / np.sum(e)

        action = self.choice(probs)
        self.exploratory_action_taken = action != np.argmax(q_values)
        return action


class GreedyStrategy():
    def __init__(self):
        self.exploratory_action_taken = False

    def select_action(self, model, state):
        with torch.no_grad():
            q_values = model(state, drop=False).cpu().detach().data.numpy().squeeze()
            return np.argmax(q_values)

class eGreedyStrategy():
    temp = 0
    def __init__(self, epsilon, select_random_action):
        self.exploratory_action_taken = False
        self.epsilon = epsilon
        self.select_random_action = select_random_action

    def select_action(self, model, state, batch=False):
        if not batch:
            r = np.random.random()
            if r>self.epsilon:
                with torch.no_grad():
                    q_values = model(state, drop=False).cpu().detach().data.numpy().squeeze()
                    return np.argmax(q_values)
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

