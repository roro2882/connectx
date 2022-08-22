#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys
import os

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
        if file == "trash":
            continue
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
xmin = 10000
i=0
while i < len(sys.argv):
    arg = sys.argv[i]
    if arg == '-r':
        root = sys.argv[i+1]
        i+=1
    elif arg == '-t':
        maxtime = int(sys.argv[i+1])
        i+=1
    elif arg == '-xmin':
        xmin = int(sys.argv[i+1])
        i+=1
    else:
        args.append(arg)
    i+=1


current = 'name'
plots = []
names = []
isarg = False
for i, arg in enumerate(args):
    if "Visualize" in arg:
        isarg=True
        continue
    elif not isarg:
        continue
    if arg == 'plot':
        current = 'plot'
    else:
        if current=='name':
            names.append(arg)
        elif current=='plot':
            plots.append(arg)

if maxtime != -1:
    names = lnames[:maxtime]

files = [[open(os.path.join(root,name,'data',plot),'r') for plot in plots] for name in names]

data = [([ ([],[]) for _ in range(len(names)) ]) for _ in range(len(plots))]

ymin, ymax = [10.0 for i in range(len(plots))],[-10.0 for i in range(len(plots))]
xmax = [0.0 for i in range(len(plots))]
needDraw = [False for i in range(len(plots))]
def update_data():
    global xmax, ymin, ymax,needDraw
    for i, name in enumerate(names):
        for j, file in enumerate(files[i]):
            nline = file.readline()
            while nline!='':
                x, y = nline.split(',')
                x = int(x)
                if x>xmax[j]:
                    xmax[j] = 1.1*x
                    needDraw[j] = True
                y = float(y)
                if y<ymin[j] and x>xmin:
                    ymin[j] = y-0.1*(ymax[j]-ymin[j])
                    needDraw[j] = True
                if y>ymax[j] and x>xmin:
                    ymax[j] = y+0.1*(ymax[j]-ymin[j])
                    needDraw[j] = True
                data[j][i][0].append(x)
                data[j][i][1].append(y)
                nline = file.readline()

update_data()
rows = 3
columns = (len(plots)-1)//3+1
itorc=[(i%3,i//3) for i in range(len(plots))]
fig, axs = plt.subplots(nrows=rows,ncols=columns, sharex=False)

axs = axs.reshape((rows,columns))
for i in range(len(plots)):
    axs[itorc[i][0], itorc[i][1]].set_title(plots[i])


lines = []
for i in range(len(plots)):
    lines.append([])
    for j in range(len(names)):
        if i==0:
            lines[i].append(axs[itorc[i][0], itorc[i][1]].plot(data[i][j][0], data[i][j][1], label=names[j])[0])
        else:
            lines[i].append(axs[itorc[i][0], itorc[i][1]].plot(data[i][j][0], data[i][j][1])[0])

rartists = []
for j in range(len(names)):
    for i in range(len(plots)): 
        rartists.append(lines[i][j]) 

#for i in range(len(plots)): 
#    rartists.append(axs[itorc[i][0], itorc[i][1]])


fig.legend()

def smooth(x, y,interval):
    if len(y)%interval>0:
        ny = np.zeros(len(y)//interval+1)
    else:
        ny = np.zeros(len(y)//interval)
    for i in range(len(y)//interval):
        ny[i] = np.mean(y[i*interval:(i+1)*interval])
    if len(y)%interval>0:
        ny[len(y)//interval] = np.mean(y[len(y)//interval*interval:])
    nx = x[::interval]
    return nx, ny

def animate(i):
    #retrieve data
    update_data()
    for i in range(len(plots)):
        for j in range(len(names)):
            interval = int(len(data[i][j][0])/100)+1
            smoothedx, smoothedy = smooth(data[i][j][0],data[i][j][1],interval)
            lines[i][j].set_data(smoothedx,smoothedy)  # update the data

        if needDraw[i]:
            axs[itorc[i][0], itorc[i][1]].set_xlim(left=xmin, right=xmax[i])
            axs[itorc[i][0], itorc[i][1]].set_ylim(bottom=ymin[i], top=ymax[i])
            needDraw[i] = False
    return rartists


# Init only required for blitting to give a clean slate.
def init():
    return rartists

ani = animation.FuncAnimation(fig, animate, np.arange(1, 200), init_func=init,
                              interval=1000, blit=False)
plt.show()
