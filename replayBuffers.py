import numpy as np
class ReplayBuffer():
    def __init__(self, 
                 state_size,
                 max_size=10000, 
                 batch_size=64):
        self.ss_mem = np.empty(shape=(max_size,)+tuple(state_size), dtype=np.int8)
        self.as_mem = np.empty(shape=(max_size,1), dtype=np.int8)
        self.rs_mem = np.empty(shape=(max_size,1), dtype=np.float32)
        self.ps_mem = np.empty(shape=(max_size,)+tuple(state_size), dtype = np.int8)
        self.ds_mem = np.empty(shape=(max_size,1), dtype=np.bool8)

        self.max_size = max_size
        self.batch_size = batch_size
        self._idx = 0
        self.size = 0
    
    def store(self, sample):
#        if self._idx==self.max_size-1:
#            return
        s, a, r, p, d = sample
        self.ss_mem[self._idx] = s
        self.as_mem[self._idx,0] = a
        self.rs_mem[self._idx,0] = r
        self.ps_mem[self._idx] = p
        self.ds_mem[self._idx,0] = d
        
        self._idx += 1
        self._idx = self._idx % self.max_size

        self.size += 1
        self.size = min(self.size, self.max_size)

    def bstore(self, samples):
        s, a, r, p, d = samples
        n = len(a)
        if n<=self.max_size-self._idx:
            self.ss_mem[self._idx:self._idx+n] = s
            self.as_mem[self._idx:self._idx+n,] = a
            self.rs_mem[self._idx:self._idx+n,] = r
            self.ps_mem[self._idx:self._idx+n] = p
            self.ds_mem[self._idx:self._idx+n,] = d
        else : 
            ni = self.max_size-self._idx
            self.ss_mem[self._idx:] = s[:ni]
            self.as_mem[self._idx:,] = a[:ni]
            self.rs_mem[self._idx:,] = r[:ni]
            self.ps_mem[self._idx:] = p[:ni]
            self.ds_mem[self._idx:,] = d[:ni]
            self._idx = 0
            n -= ni
            self.ss_mem[self._idx:self._idx+n] = s[ni:]
            self.as_mem[self._idx:self._idx+n,] = a[ni:]
            self.rs_mem[self._idx:self._idx+n,] = r[ni:]
            self.ps_mem[self._idx:self._idx+n] = p[ni:]
            self.ds_mem[self._idx:self._idx+n,] = d[ni:]



        self._idx += n
        self._idx = self._idx % self.max_size
        self.size += n
        self.size = min(self.size, self.max_size)


    def sample(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size

        idxs = np.random.randint(0,
            self.size, batch_size)
        experiences = self.ss_mem[idxs], \
                      self.as_mem[idxs], \
                      self.rs_mem[idxs], \
                      self.ps_mem[idxs], \
                      self.ds_mem[idxs]
        
        return experiences

    def __len__(self):
        return self.size


