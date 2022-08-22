import gym_connect
import time

ptime = time.time()
config = {'rows':4,'columns':4,'inarow':3,'agent':None}
env = gym_connect.gym_connect(config)
steps = 0
for i in range(10000):
    s = env.reset()
    done = False
    while not done:
        steps += 1
        s, r, done, i = env.step(env.randomplay(s))

print(steps)
print(time.time()-ptime)
