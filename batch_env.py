import numpy as np

class batchEnv():
    envs = []
    batch_agent = None
    def __init__(self, make_env_fn, num = 16, batch_agent = None):
        for i in range(num):
            self.envs.append(make_env_fn())
        self.states = np.zeros((num,)+self.envs[0].observation_space,dtype = np.uint8)
        self.rewards= np.zeros((num,1),dtype = np.float32)
        self.dones= np.zeros((num,1),dtype = np.bool8)
        self.infos= np.zeros((num,1),dtype = np.str0)
        self.batch_agent = batch_agent

    def reset(self):
        for i in range(len(self.envs)):
            self.states[i] = self.envs[i].reset()
        return self.states

    def make_batch_agent(self, single_agent):
        def batch_agent(states):
            actions = np.array((len(self.envs),1),dtype=np.int8)
            for i in range(len(self.envs)):
                actions[i,0] = single_agent(states[i])
            return actions
        return batch_agent
            

    def step(self, actions):
        if self.batch_agent is not None:
            for i in range(len(self.envs)):
                self.states[i], self.rewards[i,0], self.dones[i,0], self.infos[i,0] = self.envs[i].playerstep(0, actions[i,0])
               
            oactions = self.batch_agent(self.states)
            for i in range(len(self.envs)):
                if self.dones[i]:
                    self.states[i] = self.envs[i].reset()
                    continue
                self.states[i], oreward, odone, oinfos = self.envs[i].playerstep(1,oactions[i,0])
                self.infos[i,0]+=oinfos
                if oreward==1:
                    self.rewards[i,0]=-1
                if odone:
                    self.dones[i,0] = True
                    self.states[i] = self.envs[i].reset()
        else : 
            for i in range(len(self.envs)):
                self.states[i], self.rewards[i], self.dones[i], self.infos[i] = self.envs[i].step(actions[i])

        return self.states, self.rewards, self.dones, self.infos


