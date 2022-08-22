#!/usr/bin/python
from enum import auto
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys
import os
import datetime

root = 'runs'
def print_plots():
    path = os.path.join(root,lnames[0],'data')
    files = os.listdir(path)
    for file in files:
        print(file)

def get_names():
    path = os.path.join(root)
    files = os.listdir(root)
    lnames = []
    ltimes = []
    for file in files:
        #print(file)
        sfiles = os.listdir(os.path.join(root,file))
        wtime = 0
        for sfile in sfiles:
            wtime = max(wtime,os.path.getmtime(os.path.join(root,file,sfile)))
        i=0
        for i in range(len(ltimes)):
            if ltimes[i]<wtime:
                break
        lnames.insert(i,file)
        ltimes.insert(i,wtime)
        #print(date)
    return lnames

lnames = get_names()
print(lnames)
print_plots()
args = []
maxtime = -1
autoremove = False
xmin = 10000
i=0
while i < len(sys.argv):
    arg = sys.argv[i]
    if arg == '-r':
        root = sys.argv[i+1]
        i+=1
    elif arg == '-auto-remove':
        autoremove = True
    else:
        args.append(arg)
    i+=1

def removeRec(path):
    if os.path.isdir(path):
        files = os.listdir(path)
        for file in files:
            removeRec(os.path.join(path,file))
        os.rmdir(path)
    else:
        os.remove(path)
test = 'value_loss'
if autoremove:
    for name in lnames:
        path = os.path.join(root, name, 'data', test)
        if name != 'trash':
            if not os.path.exists(path):
                removeRec(os.path.join(root,name))


