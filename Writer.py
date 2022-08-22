import os

import numpy as np
import time
import torch
import os.path


class Writer():
    path = ''
    infofile = ''
    datafiles = {}
    monitoredscalars = {}

    def __init__(self, name, rootd = './', cont=False):
        self.path = os.path.join(rootd, name)
        exists = os.path.exists(self.path)
        i=0
        while exists:
            self.path = os.path.join(rootd, name+str(i))
            exists = os.path.exists(self.path)
            i+=1
        os.mkdir(self.path)
        self.infopath = os.path.join(self.path, 'infos.txt')
        self.datapath = os.path.join(self.path, 'data')
        os.mkdir(self.datapath)
        self.infofile = open(self.infopath,'a')

    def add_fscalar(self, name, value, step, average_over=-1):
        if name not in self.monitoredscalars:
            self.monitoredscalars[name]=[[],[],average_over, 0]
        self.monitoredscalars[name][0].append(step)
        self.monitoredscalars[name][1].append(value)

    def flush_fscalars(self):
        scalars = {}
        for index, name in enumerate(self.monitoredscalars):
            x,y, average_over, last_index = self.monitoredscalars[name]
            average = 0
            if average_over == -1:
                average = np.mean(y[last_index:len(y)])
            else:
                if len(y)<=average_over:
                    average = np.mean(y[-len(y):])
                else:
                    average = np.mean(y[-average_over:])
            scalars[name]=average
            self.monitoredscalars[name][3] = len(y)
            self.add_scalar(name,average,x[-1])
        return scalars


    def add_scalar(self, name, value, step):
        if name not in self.datafiles:
            npath = os.path.join(self.datapath, name)
            self.datafiles[name] = open(npath, 'a')

        file = self.datafiles[name]
        file.write(str(step)+','+str(value)+'\n')
        file.flush()

    def add_text(self, name, value):
        if name not in self.datafiles:
            npath = os.path.join(self.datapath, name)
            self.datafiles[name] = open(npath, 'a')

        file = self.datafiles[name]
        file.write(str(value)+'\n')
        file.flush()
