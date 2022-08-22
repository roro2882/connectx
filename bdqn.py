import gym_connect as gym_connect
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from itertools import count
from strategies import GreedyStrategy, SoftMaxStrategy, eGreedyStrategy
from replayBuffers import ReplayBuffer
from networks import FCQ
from batch_env import batchEnv


import os.path
import random
import glob
import time
import os

from Writer import Writer
torch.backends.cudnn.benchmark = True

LEAVE_PRINT_EVERY_N_SECS = 2
SAVE_EVERY_N_STEPS = 50000
EVAL_EVERY_N_STEPS = 1000
DEBUG_EVERY_N_STEPS = 1000
DEMO_EVERY_N_STEPS = 2000
DEMOR_EVERY_N_STEPS = 10000
ERASE_LINE = '\x1b[2K'
BEEP = lambda: os.system("printf '\a'")


def get_make_env_fn(**kargs):
    def make_env_fn(rows=4, columns=4, inarow=3, agent=None ,seed=None, debug = False):
        env = gym_connect.gym_connect({'rows':rows, 'columns':columns, 'inarow':inarow,'agent':agent, 'debug':debug})
        return env
    return make_env_fn, kargs



class Agent():
    def __init__(self, model_fn, path):
        self.model = model_fn()
        dictionarysave = torch.load(path)

        self.model.load_state_dict(dictionarysave['state_dict'])
    def play(self, board):
         with torch.no_grad():
            q_values = self.model(board, drop=False).cpu().detach().data.numpy().squeeze()
            return np.argmax(q_values)
       


class DDQN():
    def __init__(self, 
                 replay_buffer, 
                 value_model_fn, 
                 value_optimizer_fn, 
                 value_scheduler_fn,
                 training_strategy_fn,
                 evaluation_strategy_fn,
                 n_warmup_batches,
                 train_every_n_steps,
                 update_target_every_steps):
        self.replay_buffer = replay_buffer
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_scheduler_fn = value_scheduler_fn
        self.training_strategy_fn = training_strategy_fn
        self.evaluation_strategy_fn = evaluation_strategy_fn
        self.n_warmup_batches = n_warmup_batches
        self.train_every_n_steps = train_every_n_steps
        self.update_target_every_steps = update_target_every_steps
        self.target_model = self.value_model_fn()
        self.target_model1 = self.value_model_fn()
        self.online_model = self.value_model_fn()
        self.online_model1 = self.value_model_fn()
        self.total_ep = 0
        self.total_step = 0
        self.update_network()

    def optimize_model(self, experiences):
        r = np.random.random()
        states, actions, rewards, next_states, is_terminals = experiences
        if r>0.5:
            with torch.no_grad():
                nactions = self.online_model1(next_states).max(1)[1].unsqueeze(1)
            max_a_q_sp = self.online_model(next_states).detach().gather(1,nactions)
            q_values = self.online_model(states)
        else:
            with torch.no_grad():
                nactions = self.online_model(next_states).max(1)[1].unsqueeze(1)
            max_a_q_sp = self.online_model1(next_states).detach().gather(1,nactions)
            q_values = self.online_model1(states)

        target_q_sa = rewards + (self.gamma * max_a_q_sp * (1 - is_terminals))
        q_sa = q_values.gather(1, actions)
        td_error = q_sa - target_q_sa
        value_loss = td_error.pow(2).mul(0.5).mean()
        value_loss.backward()

        if self.total_step%50==0:
            writer.add_fscalar('value_loss',value_loss.item(),self.total_step, 3000)
            if r>0.5:
                writer.add_fscalar('value_loss1',value_loss.item(),self.total_step, 3000)
            else:
                writer.add_fscalar('value_loss2',value_loss.item(),self.total_step, 3000)
            writer.add_fscalar('q_value',q_sa.mean().item(), self.total_step, 3000)

        if r>0.5:
            self.value_optimizer.step()
            self.value_optimizer.zero_grad()
        else:
            self.value_optimizer1.step()
            self.value_optimizer1.zero_grad()

        self.scheduer.step()
        self.scheduer1.step()
        self.total_ep+=1


    def interaction_step(self, states, batch_env):
        actions = self.training_strategy.select_action(self.online_model, states, batch=True)
        new_states, rewards, is_terminals, _= batch_env.step(actions)
        experience = states, actions, rewards, new_states, is_terminals

        self.replay_buffer.bstore(experience)
        writer.add_fscalar('step_reward',np.mean(rewards), self.total_step, 10000)
        #self.episode_exploration[-1] += np.sum((self.training_strategy.exploratory_action_taken))
        return new_states, is_terminals
    
    def update_network(self):
        for target, online in zip(self.target_model.parameters(), 
                                  self.online_model.parameters()):
            target.data.copy_(online.data)

    def train(self, make_env_fn, make_env_kargs, make_eval_env_kargs, seed, gamma, 
              max_minutes, max_step, batch_num = 16):
        training_start, last_debug_time = time.time(), float('-inf')
        self.checkpoint_dir = "./checkpoints"
        self.make_env_fn = make_env_fn
        self.make_env_kargs = make_env_kargs
        self.seed = seed
        self.gamma = gamma
        
        #envs... 
        smake_env_fn = lambda : self.make_env_fn(**self.make_env_kargs, seed = self.seed)
        batch_env = batchEnv(smake_env_fn, num = batch_num, batch_agent = self.bagent)
        demo_env = self.make_env_fn(**make_env_kargs, seed = self.seed)
        eval_env = self.make_env_fn(**make_eval_env_kargs, seed = self.seed)

        # random seed
        torch.manual_seed(self.seed) ; np.random.seed(self.seed) ; random.seed(self.seed)
    
        self.value_optimizer = self.value_optimizer_fn(self.online_model)
        self.value_optimizer1 = self.value_optimizer_fn(self.online_model1)

        self.training_strategy = training_strategy_fn()
        #self.training_strategy.t = self.total_step
        self.evaluation_strategy = evaluation_strategy_fn() 
                    
        self.scheduer = self.value_scheduler_fn(self.value_optimizer)
        self.scheduer1 = self.value_scheduler_fn(self.value_optimizer1) # lr sch not functionnal when resuming

        training_time = 0
        states = batch_env.reset()

        min_samples = self.replay_buffer.batch_size * self.n_warmup_batches

        self.total_episodes = 0
        for step in count():
            states, is_terminals = self.interaction_step(states, batch_env)
            self.total_step += 1
            self.total_episodes += np.sum(is_terminals)
                           
            if self.total_step % self.update_target_every_steps == 0:
                self.update_network()

            if len(self.replay_buffer) > min_samples:
                if self.train_every_n_steps>0:
                    if self.total_step%self.train_every_n_steps==0:
                        experiences = self.replay_buffer.sample()
                        experiences = self.online_model.load(experiences)
                        self.optimize_model(experiences)
                else:
                    for _ in range(-self.train_every_n_steps):
                        experiences = self.replay_buffer.sample()
                        experiences = self.online_model.load(experiences)
                        self.optimize_model(experiences)


            
            if self.total_step%EVAL_EVERY_N_STEPS==0:
                evaluation_score, _ = self.evaluate(self.online_model, eval_env)
                _, _ = self.evaluate(self.online_model, demo_env, is_eval_env = False)
                writer.add_fscalar('ev_score',evaluation_score,self.total_step, 40)
            if self.total_step%SAVE_EVERY_N_STEPS==0:
                self.save_checkpoint()
            if self.total_step%DEMO_EVERY_N_STEPS==0:
                self.demo(5,demo_env,self.online_model, self.training_strategy)
                
            if self.total_step%DEMOR_EVERY_N_STEPS==0:
                print('rand demo')
                self.demo(20,eval_env,self.online_model, self.evaluation_strategy)
                
            if self.total_step%DEBUG_EVERY_N_STEPS==0:
                wallclock_elapsed = time.time() - training_start
                
                reached_debug_time = time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS
                reached_max_minutes = wallclock_elapsed >= max_minutes * 60
                reached_max_steps = self.total_step >= max_steps
                training_is_over = reached_max_minutes or \
                                   reached_max_steps
                 
                # LOGGING
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
                debug_message = 'el {}, ep {:04}, ts {:06}, '
                debug_message += 'ev {:05.2f}'
                debug_message += ',   vl {:05.3f}'
                writer.add_scalar('time',wallclock_elapsed,self.total_step)
                writer.add_scalar('total_ep',self.total_ep,self.total_step)
                writer.add_scalar('total_episodes',self.total_episodes,self.total_step)
                scalars = writer.flush_fscalars()
                debug_message = debug_message.format(
                    elapsed_str,self.total_episodes , self.total_step,
                    scalars['ev_score'],
                    scalars['value_loss'])
                print(debug_message, end='\r', flush=True)
                if reached_debug_time or training_is_over:
                    print(ERASE_LINE + debug_message, flush=True)
                    last_debug_time = time.time()
                if training_is_over:
                    if reached_max_minutes: print(u'--> reached_max_minutes \u2715')
                    if reached_max_steps: print(u'--> reached_max_steps \u2715')
                    break
                
        final_eval_score, score_std = self.evaluate(self.online_model, eval_env, n_episodes=5000)
        wallclock_time = time.time() - training_start
        print('Training complete.')
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
              ' {:.2f}s wall-clock time.\n'.format(
                  final_eval_score, score_std, training_time, wallclock_time))
        self.demo(20,demo_env,self.online_model, self.evaluation_strategy)
        return result, final_eval_score, training_time, wallclock_time
    
    def agent(self, state, debug=False):
        r = np.random.random()
        if debug:
            print("agent!")
        if r>0.5:
            return self.training_strategy.select_action(self.online_model,state, debug=debug)
        else:
            return self.training_strategy.select_action(self.online_model1,state, debug=debug)

    def bagent(self, states):
        r = np.random.random()
        if r>0.5:
            return self.training_strategy.select_action(self.online_model,states,batch=True)
        else:
            return self.training_strategy.select_action(self.online_model1,states, batch = True)


    def demo(self, n_demo,env, model, strategy):
        for episode in range(1, n_demo+ 1):
            state, is_terminal = env.reset(), False

            #print(state[0]+2*state[1])
            for step in count():
                a = strategy.select_action(model,state, debug=True)
                q_values1 = self.online_model1(state)
                print("2nd network:",q_values1)
                #print(a)
                state, reward, is_terminal, info = env.step(a)
                #print(state[0]+2*state[1])
                print(reward, info)
                if is_terminal:
                    print('--------------------------')
                    break
         
    def evaluate(self, eval_policy_model, eval_env, is_eval_env=True, n_episodes=10):
        rs = []
        for _ in range(n_episodes):
            s, d = eval_env.reset(), False
            rs.append(0)
            states = []
            actions = []
            rewards = []
            next_states = []
            is_terminals = []
            step = 0
            for step in count():
                if is_eval_env:
                    a = self.evaluation_strategy.select_action(eval_policy_model, s)
                else :
                    a = self.training_strategy.select_action(eval_policy_model,s)
                states.append(s)
                actions.append(a)
                s, r, d, _ = eval_env.step(a)
                rewards.append(r)
                is_terminals.append(d)
                next_states.append(s)
                rs[-1] += r
                if d: break
            with torch.no_grad():
                experiences = np.array(states), np.array(actions).reshape((-1,1)), np.array(rewards).reshape((-1,1)) , np.array(next_states), np.array(is_terminals).reshape((-1,1))
                states, actions, rewards, next_states, is_terminals = self.online_model.load(experiences)
                nactions = self.online_model1(next_states).max(1)[1].unsqueeze(1)
                max_a_q_sp = self.online_model(next_states).detach().gather(1,nactions)
                target_q_sa = rewards + (self.gamma * max_a_q_sp * (1 - is_terminals))
                q_values = self.online_model(states)
                q_values1 = self.online_model1(states)
                numpy_q_values = q_values.cpu().detach().data.numpy()
                writer.add_fscalar('nq_dist',np.std(numpy_q_values),self.total_step,1000)
                writer.add_fscalar('nq_value',np.mean(numpy_q_values),self.total_step,1000)
                q_sa = q_values.gather(1, actions)
                q_sa1 = q_values1.gather(1, actions)
                numpy_q_sa = q_sa.cpu().detach().data.numpy()
                numpy_q_sa1 = q_sa1.cpu().detach().data.numpy()
                
                td_error = q_sa - target_q_sa
                value_loss = td_error.pow(2).mul(0.5).mean()
                monte_carlo_returns = np.zeros((step+1,1))
                monte_carlo_returns[-1,0] = rewards[-1,0]
                for s in range(2,step+1):
                    monte_carlo_returns[-s,0] = rewards[-s,0] + self.gamma*monte_carlo_returns[-s+1,0]
                #print('test monte carlo')
                #print(rewards)
                #print(monte_carlo_returns)
                #print(numpy_q_sa)
                errors = (numpy_q_sa - monte_carlo_returns)**2*0.5
                errors1 = (numpy_q_sa1 - monte_carlo_returns)**2*0.5
                if is_eval_env:
                    writer.add_fscalar("ev_value_loss",value_loss.item(),self.total_step,1000)
                    writer.add_fscalar("ev_episode_length",step,self.total_step,1000)
                    writer.add_fscalar("ev_mc_loss",errors.mean(), self.total_step, 1000)
                    writer.add_fscalar("ev_mc_loss1",errors1.mean(), self.total_step, 1000)
                    for s in range(1,step+1):
                        writer.add_fscalar("ev_mc_loss-"+str(s),errors[-s,0],self.total_step,1000)
                        writer.add_fscalar("ev_mc_loss1-"+str(s),errors1[-s,0],self.total_step,1000)
                else:
                    writer.add_fscalar('nev_value_loss',value_loss.item(), self.total_step,1000)
                    writer.add_fscalar("nev_episode_length",step,self.total_step,1000)
                    writer.add_fscalar("nev_mc_loss",errors.mean(), self.total_step, 1000)
                    writer.add_fscalar("ev_mc_loss1",errors1.mean(), self.total_step, 1000)
                    for s in range(1,step+1):
                        writer.add_fscalar("nev_mc_loss-"+str(s),errors[-s,0],self.total_step,1000)
                        writer.add_fscalar("nev_mc_loss1-"+str(s),errors1[-s,0],self.total_step,1000)


        return np.mean(rs), np.std(rs)

    def retrieve_checkpoint(self, path):
        print('retrieving checkpoint',path)
        dictionarysave = torch.load(path)

        self.online_model.load_state_dict(dictionarysave['state_dict'])
        self.online_model1.load_state_dict(dictionarysave['state_dict_1'])
        self.value_optimizer.load_state_dict(dictionarysave['optim_state_dict'])
        self.value_optimizer1.load_state_dict(dictionarysave['optim_state_dict_1'])
        self.update_network()
        self.total_step = dictionarysave['total_step']
        self.total_ep = dictionarysave['total_ep']

    def save_checkpoint(self):
        dictionarysave = {
                'state_dict':self.online_model.state_dict(),
                'state_dict_1':self.online_model1.state_dict(),
                'optim_state_dict':self.value_optimizer.state_dict(),
                'optim_state_dict_1':self.value_optimizer1.state_dict(),
                'total_step':self.total_step,
                'total_ep':self.total_ep,
                }
        torch.save(dictionarysave, 
                   os.path.join(self.checkpoint_dir, 'model.{}.tar'.format(environment_settings['name'])))
    def save_games(self, savingpath):
        dictionarysave = {
                'states':self.replay_buffer.ss_mem,
                'actions':self.replay_buffer.as_mem,
                'nstates':self.replay_buffer.ps_mem,
                'rewards':self.replay_buffer.rs_mem,
                }
        torch.save(dictionarysave, 
                   os.path.join(savingpath, 'games.{}.tar'.format(environment_settings['name'])))



environment_settings = {
    'gamma': 0.9,
    'lr':1e-4,
    'batch_size':128,
    'replay_buffer_size':200000,
    'update_target_every_n_steps':200000,
    'lr_scheduler_min':5e-6,
    'lr_scheduler_epochs':50000,
    'goal_mean_100_reward': 1.1,
    'n_warmup_batches':100,
    'train_every_n_steps':-2,
    'temp':0.2,
    'max_minutes': 1800,
    'max_steps':10000000,
    'env_batch_size':64,
}
environment_settings['name'] = '67soft_hardrelu'

rows, columns, inarow = 6,7,4
seed = np.random.randint(10,100)
nS = [2,rows,columns]
nA = columns

#value model
value_model_fn = lambda :FCQ(nS, nA, hidden_dims=(0,128,64), activation_fc=F.relu)
value_optimizer_fn = lambda net: optim.Adam(net.parameters(), lr=environment_settings['lr'])

#scheduler
sch_gamma = np.exp(-1/environment_settings['lr_scheduler_epochs'])
sch = lambda epoch: sch_gamma**epoch + environment_settings['lr_scheduler_min']/environment_settings['lr']
value_scheduler_fn = lambda value_optimizer, last_ep=-1 : torch.optim.lr_scheduler.LambdaLR(value_optimizer, sch,last_epoch=last_ep)

dummy_env = gym_connect.gym_connect(config={'agent': None, 'rows':rows, 'columns':columns, 'inarow':inarow})

#strategies
training_strategy_fn = lambda: SoftMaxStrategy(environment_settings['temp'],environment_settings['temp'],1,200200000000)
evaluation_strategy_fn = lambda: GreedyStrategy()

#replay_buffer
replay_buffer = ReplayBuffer(nS,max_size=environment_settings['replay_buffer_size'], batch_size=environment_settings['batch_size'])
n_warmup_batches = environment_settings['n_warmup_batches']

gamma, max_minutes, max_steps, goal_mean_100_reward = \
        environment_settings['gamma'], environment_settings['max_minutes'], environment_settings['max_steps'], environment_settings['goal_mean_100_reward']

#analyse d'arguments
path = ''
opponentpath = ''
savingpath = ''
generation = False
for i,arg in enumerate(sys.argv):
    if arg == '-o':
        path = sys.argv[i+1]
        environment_settings['n_warmup_batches'] = 1000
    if arg == '-s':
        savingpath = sys.argv[i+1]
    if arg == '-e':
        opponentpath = sys.argv[i+1]
    if arg == '-g':
        generation = True

#agent evaluation
agent_eval = 'rule'
if opponentpath != '':
    agent_eval = Agent(value_model_fn, opponentpath ).play 

if generation:
    SAVE_EVERY_N_EPISODES = 999999999
    n_warmup_batches = 100000000
    environment_settings['replay_buffer_size'] = environment_settings['max_steps']
    environment_settings['name'] = 'generation'

    dummy_env = gym_connect.gym_connect(config={'agent': None, 'rows':rows, 'columns':columns, 'inarow':inarow})
    training_strategy_fn = lambda: eGreedyStrategy(1, dummy_env.randomplay)
    SAVE_EVERY_N_EPISODES = 5000
    EVAL_EVERY_N_EPISODES = 100000000
    DEBUG_EVERY_N_EPISODES = 100
    DEMO_EVERY_N_EPISODES = 2000


#environment_settings['name'] =  environment_settings['name'].replace('%t',str(int(time.time())))

writer = Writer(environment_settings['name'], rootd='./runs')

agent = DDQN(replay_buffer,
            value_model_fn,
            value_optimizer_fn,
            value_scheduler_fn,
            training_strategy_fn,
            evaluation_strategy_fn,
            n_warmup_batches,
            environment_settings['train_every_n_steps'],
            environment_settings['update_target_every_n_steps'])


writer.add_text('network',str(agent.online_model))
writer.add_text('params',str(environment_settings))

#-o
if path != "":
    agent.retrieve_checkpoint(path)
    if generation:
        agent.total_step = 0



make_env_fn, make_env_kargs = get_make_env_fn(agent = agent.agent, rows = rows, columns = columns, inarow = inarow)
_, make_eval_env_kargs = get_make_env_fn(agent = agent_eval, rows = rows, columns = columns, inarow = inarow)


result, final_eval_score, training_time, wallclock_time = agent.train(
    make_env_fn, make_env_kargs, make_eval_env_kargs, seed, gamma, max_minutes, max_steps,batch_num=environment_settings['env_batch_size'])
metrics = {'final_eval_score':final_eval_score, 'final value_loss':mean_100_value_loss, 'wallclock_time':wallclock_time}
if savingpath != '':
    agent.save_games(savingpath)
print(metrics)
hparams = environment_settings
#writer.add_hparams(environment_settings,metrics)
_ = BEEP()
