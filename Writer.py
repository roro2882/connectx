import os

import numpy as np
import time
import torch
import os.path


class Writer():
    path = ''
    infofile = ''
    datafiles = {}

    def __init__(self, name, rootd = './'):
        self.path = os.path.join(rootd, name)
        os.mkdir(self.path)
        self.infopath = os.path.join(self.path, 'infos.txt')
        self.datapath = os.path.join(self.path, 'data')
        os.mkdir(self.datapath)
        self.infofile = open(self.infopath,'a')

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


