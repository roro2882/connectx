import gym_connect
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from itertools import count
from strategies import GreedyStrategy, SoftMaxStrategy
from replayBuffers import ReplayBuffer


import os.path
import random
import glob
import time
import os

from torch.utils.tensorboard.writer import SummaryWriter


LEAVE_PRINT_EVERY_N_SECS = 2
SAVE_EVERY_N_EPISODES = 5000
EVAL_EVERY_N_EPISODES = 10
DEBUG_EVERY_N_EPISODES = 10
DEMO_EVERY_N_EPISODES = 200
ERASE_LINE = '\x1b[2K'
EPS = 1e-6
BEEP = lambda: os.system("printf '\a'")
RESULTS_DIR = os.path.join('..', 'results')


def get_make_env_fn(**kargs):
    def make_env_fn(rows=4, columns=4, inarow=3, agent=None ,seed=None, debug = False):
        env = gym_connect.gym_connect({'rows':rows, 'columns':columns, 'inarow':inarow,'agent':agent, 'debug':debug})
        return env
    return make_env_fn, kargs

class FCQ(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=[32,32],
                 activation_fc=F.relu):
        super(FCQ, self).__init__()
        self.activation_fc = activation_fc
        self.input_dim = input_dim
        out_channels = 64

        self.input_layer = nn.Conv2d(input_dim[0], out_channels = out_channels, kernel_size=3, padding=1, padding_mode='zeros', stride=1)
        self.input_layer1 = nn.Conv2d(out_channels, out_channels = out_channels, kernel_size=3, padding=1, stride=1)
        self.pool_layer    = nn.MaxPool2d(5)
        self.dropout = nn.Dropout2d(0.5)
        #self.input_layer2 = nn.Conv2d(out_channels, out_channels = out_channels, kernel_size=3, padding=1, stride=1)
        #self.input_layer = nn.Conv2d(2, 16,4)

        self.hidden_layers = nn.ModuleList()
        hidden_dims = list(hidden_dims)
        hidden_dims[0] = (input_dim[1])*(input_dim[2])*out_channels
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x,
                             device=self.device,
                             dtype=torch.float32)
            if len(x.shape)==len(self.input_dim):
                x = x.unsqueeze(0)
        return x

    def forward(self, state, drop =False):
        x = self._format(state)
        x = self.input_layer(x)
        if drop:
            pass
        #    x = self.dropout(x)
        x = self.activation_fc(x)
        #for i in range(1):
        x =self.activation_fc(self.input_layer1(x))
        if drop:
            pass
        #    x = self.dropout(x)
        #x = self.activation_fc(self.input_layer2(x))
        #if drop:
        #    x = self.dropout(x)
        #x = self.pool_layer(x)
        #x = self.pool_layer(x)
        #x = self.input_layer2(x)
        #if drop:
        #    x = self.dropout(x)

        x = torch.flatten(x,start_dim=1)
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        x = self.output_layer(x)
        return x

    def numpy_float_to_device(self, variable):
        variable = torch.from_numpy(variable).float().to(self.device)
        return variable

    def load(self, experiences):
        states, actions, rewards, new_states, is_terminals = experiences
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, rewards, new_states, is_terminals


class Agent():
    def __init__(self, model_fn, path):
        self.model = model_fn()
        dictionarysave = torch.load(path)

        self.model.load_state_dict(dictionarysave['state_dict'])
    def play(self, board):
         with torch.no_grad():
            q_values = self.model(board, drop=False).cpu().detach().data.numpy().squeeze()
            return np.argmax(q_values)
       


class DQN():
    def __init__(self, 
                 replay_buffer, 
                 value_model_fn, 
                 value_optimizer_fn, 
                 value_optimizer_lr,
                 value_scheduler_fn,
                 training_strategy_fn,
                 evaluation_strategy_fn,
                 n_warmup_batches,
                 update_target_every_steps):
        self.replay_buffer = replay_buffer
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.value_scheduler_fn = value_scheduler_fn
        self.training_strategy_fn = training_strategy_fn
        self.evaluation_strategy_fn = evaluation_strategy_fn
        self.n_warmup_batches = n_warmup_batches
        self.update_target_every_steps = update_target_every_steps
        self.total_step = 0
        self.target_model = self.value_model_fn()
        self.online_model = self.value_model_fn()
        self.update_network()

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences
        with torch.no_grad():
            max_a_q_sp = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        #target
        target_q_sa = rewards + (self.gamma * max_a_q_sp * (1 - is_terminals))
        q_values = self.online_model(states)
        self.q_dist.append(np.std(q_values.cpu().detach().data.numpy()))
        q_sa = q_values.gather(1, actions)
        self.q_value.append(q_sa.mean().item())
        
        td_error = q_sa - target_q_sa
        value_loss = td_error.pow(2).mul(0.5).mean()
        self.value_errors.append(value_loss.item())
        value_loss.backward()
        self.value_optimizer.step()
        self.scheduer.step()
        self.value_optimizer.zero_grad()

    def interaction_step(self, state, env):
        action = self.training_strategy.select_action(self.online_model, state)
        new_state, reward, is_terminal, info = env.step(action)
        is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
        is_failure = is_terminal and not is_truncated
        experience = (state, action, reward, new_state, float(is_failure))

        self.replay_buffer.store(experience)
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        self.episode_exploration[-1] += int(self.training_strategy.exploratory_action_taken)
        return new_state, is_terminal
    
    def update_network(self):
        for target, online in zip(self.target_model.parameters(), 
                                  self.online_model.parameters()):
            target.data.copy_(online.data)

    def train(self, make_env_fn, make_env_kargs, make_eval_env_kargs, seed, gamma, 
              max_minutes, max_steps, goal_mean_100_reward):
        training_start, last_debug_time = time.time(), float('-inf')
        self.checkpoint_dir = "./checkpoints"
        self.make_env_fn = make_env_fn
        self.make_env_kargs = make_env_kargs
        self.seed = seed
        self.gamma = gamma
        
        env = self.make_env_fn(**self.make_env_kargs, seed=self.seed)
        eval_env = self.make_env_fn(**make_eval_env_kargs, seed = self.seed)
        torch.manual_seed(self.seed) ; np.random.seed(self.seed) ; random.seed(self.seed)
    
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []        
        self.episode_exploration = []
        self.q_value = []
        self.q_dist = []
        self.value_errors = []
        

        self.value_optimizer = self.value_optimizer_fn(self.online_model, 
                                                       self.value_optimizer_lr)

        self.training_strategy = training_strategy_fn()
        self.training_strategy.t = self.total_step
        self.evaluation_strategy = evaluation_strategy_fn() 
        self.scheduer = self.value_scheduler_fn(self.value_optimizer)
                    
        result = np.empty((max_steps, 5))
        result[:] = np.nan
        training_time = 0
        for episode in range(1, max_steps + 1):
            episode_start = time.time()
            
            state, is_terminal = env.reset(), False
            self.episode_reward.append(0.0)
            self.episode_timestep.append(0.0)
            self.episode_exploration.append(0.0)

            min_samples = self.replay_buffer.batch_size * self.n_warmup_batches
            for step in count():
                state, is_terminal = self.interaction_step(state, env)
                self.total_step += 1
                               
                if self.total_step % self.update_target_every_steps == 0:
                    self.update_network()
                
                if is_terminal:
                    #gc.collect()
                    break

            if len(self.replay_buffer) > min_samples:
                experiences = self.replay_buffer.sample()
                experiences = self.online_model.load(experiences)
                self.optimize_model(experiences)
 
            # stats
            episode_elapsed = time.time() - episode_start
            
            self.episode_seconds.append(episode_elapsed)
            training_time += episode_elapsed

            if episode%EVAL_EVERY_N_EPISODES==0:
                evaluation_score, _ = self.evaluate(self.online_model, eval_env)
                self.evaluation_scores.append(evaluation_score)
            if episode%SAVE_EVERY_N_EPISODES==0:
                self.save_checkpoint()
            if episode%DEMO_EVERY_N_EPISODES==0:
                self.demo(5,env,self.online_model)
            if episode%DEBUG_EVERY_N_EPISODES==0:
                mean_100_reward = np.mean(self.episode_reward[-400:])
                std_100_reward = np.std(self.episode_reward[-400:])
                mean_100_q_value = np.mean(self.q_value[-400:])
                mean_100_q_dist = np.mean(self.q_dist[-400:])
                mean_100_value_loss = np.mean(self.value_errors[-400:])
                mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
                std_100_eval_score = np.std(self.evaluation_scores[-100:])
                lst_100_exp_rat = np.array(
                    self.episode_exploration[-100:])/np.array(self.episode_timestep[-100:])
                mean_100_exp_rat = np.mean(lst_100_exp_rat)
                std_100_exp_rat = np.std(lst_100_exp_rat)
                
                wallclock_elapsed = time.time() - training_start
                result[episode-1] = self.total_step, mean_100_reward, \
                    mean_100_eval_score, training_time, wallclock_elapsed
                
                reached_debug_time = time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS
                reached_max_minutes = wallclock_elapsed >= max_minutes * 60
                reached_max_steps = self.total_step >= max_steps
                reached_goal_mean_reward = mean_100_eval_score >= goal_mean_100_reward
                training_is_over = reached_max_minutes or \
                                   reached_max_steps or \
                                   reached_goal_mean_reward

                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
                debug_message = 'el {}, ep {:04}, ts {:06}, '
                #debug_message += 'ar 10 {:05.1f}\u00B1{:05.1f}, '
                debug_message += '100 {:05.2f}\u00B1{:05.1f}, '
                debug_message += 'ev {:05.2f}\u00B1{:05.1f}'
                debug_message += ',   vl {:05.3f}'
                debug_message = debug_message.format(
                    elapsed_str, episode-1, self.total_step, #mean_10_reward, std_10_reward, 
                    mean_100_reward, std_100_reward,
                    mean_100_eval_score, std_100_eval_score,
                    mean_100_value_loss)
                print(debug_message, end='\r', flush=True)
                writer.add_scalar("value loss", mean_100_value_loss, self.total_step)
                writer.add_scalar("time", int(time.time() - training_start), self.total_step)
                writer.add_scalar("reward", mean_100_reward, self.total_step)
                writer.add_scalar("scheduer", float(self.scheduer.get_last_lr()[0]), self.total_step)
                writer.add_scalar("evaluation score", mean_100_eval_score, self.total_step)
                writer.add_scalar("q_value", mean_100_q_value, self.total_step)
                writer.add_scalar("q_dist", mean_100_q_dist, self.total_step)
                writer.add_scalar("temp", self.training_strategy.temp, self.total_step)
                if reached_debug_time or training_is_over:
                    print(ERASE_LINE + debug_message, flush=True)
                    last_debug_time = time.time()
                if training_is_over:
                    if reached_max_minutes: print(u'--> reached_max_minutes \u2715')
                    if reached_max_steps: print(u'--> reached_max_steps \u2715')
                    if reached_goal_mean_reward: print(u'--> reached_goal_mean_reward \u2713')
                    break
                
        final_eval_score, score_std = self.evaluate(self.online_model, env, n_episodes=1000)
        wallclock_time = time.time() - training_start
        print('Training complete.')
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
              ' {:.2f}s wall-clock time.\n'.format(
                  final_eval_score, score_std, training_time, wallclock_time))
        self.demo(20,env,self.online_model)
        del env
        return result, final_eval_score, mean_100_value_loss, training_time, wallclock_time
    
    def agent(self, state):
        return self.training_strategy.select_action(self.target_model,state)

    def demo(self, n_demo,env, model):
        for episode in range(1, n_demo+ 1):
            episode_start = time.time()
            
            state, is_terminal = env.reset(), False
            self.episode_reward.append(0.0)
            self.episode_timestep.append(0.0)
            self.episode_exploration.append(0.0)

            print(state[0]+2*state[1])
            for step in count():
                with torch.no_grad():
                    q_values = model(state).cpu().detach().data.numpy().squeeze()
                a = np.argmax(q_values)

                state, reward, is_terminal, info = env.step(a)
                print(q_values)
                print(state[0]+2*state[1])
                print(reward, info)
                if is_terminal:
                    #gc.collect()
                    #if reward==0:
                    #    breakpoint()
                    print('--------------------------')
                    break
         
    def evaluate(self, eval_policy_model, eval_env, n_episodes=1):
        rs = []
        for _ in range(n_episodes):
            s, d = eval_env.reset(), False
            rs.append(0)
            for _ in count():
                a = self.evaluation_strategy.select_action(eval_policy_model, s)
                s, r, d, _ = eval_env.step(a)
                rs[-1] += r
                if d: break
        return np.mean(rs), np.std(rs)

    def get_cleaned_checkpoints(self, n_checkpoints=5):
        try: 
            return self.checkpoint_paths
        except AttributeError:
            self.checkpoint_paths = {}

        paths = glob.glob(os.path.join(self.checkpoint_dir, '*.tar'))
        paths_dic = {int(path.split('.')[-2]):path for path in paths}
        last_ep = max(paths_dic.keys())
        # checkpoint_idxs = np.geomspace(1, last_ep+1, n_checkpoints, endpoint=True, dtype=np.int)-1
        checkpoint_idxs = np.linspace(1, last_ep+1, n_checkpoints, endpoint=True, dtype=np.int)-1

        for idx, path in paths_dic.items():
            if idx in checkpoint_idxs:
                self.checkpoint_paths[idx] = path
            else:
                os.unlink(path)

        return self.checkpoint_paths
    
    def retrieve_checkpoint(self, path):
        print('retrieving checkpoint',path)
        dictionarysave = torch.load(path)

        self.online_model.load_state_dict(dictionarysave['state_dict'])
        self.update_network()
        self.total_step = dictionarysave['total_step']
        self.evaluation_scores= dictionarysave['evaluation_scores']
        self.episode_reward = dictionarysave['episode_reward']

    def save_checkpoint(self):
        model = self.online_model
        dictionarysave = {
                'state_dict':model.state_dict(),
                'total_step':self.total_step,
                'evaluation_scores':self.evaluation_scores,
                'episode_reward':self.episode_reward,
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
    'lr':0.00005,
    'batch_size':64,
    'replay_buffer_size':20000,
    'update_target_every_n_steps':5000,
    'lr_scheduler_gamma':1.0,
    'goal_mean_100_reward': 1.1,
    'n_warmup_batches':50,
    'max_minutes': 30,
    'max_steps': 50000,
    'name':'auto',
}

rows, columns, inarow = 5,5,4
seed = np.random.randint(10,100)
nS = [2,rows,columns]
nA = columns


value_model_fn = lambda :FCQ(nS, nA, hidden_dims=(0,128), activation_fc=F.relu)
value_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
value_optimizer_lr = environment_settings['lr']
value_scheduler_fn = lambda value_optimizer : torch.optim.lr_scheduler.ExponentialLR(value_optimizer, environment_settings['lr_scheduler_gamma'])

training_strategy_fn = lambda: SoftMaxStrategy(init_temp=1.0, 
                                                min_temp=0.3, 
                                                exploration_ratio=0.8, 
                                                max_steps=environment_settings['max_steps'])
evaluation_strategy_fn = lambda: GreedyStrategy()

replay_buffer = ReplayBuffer(nS,max_size=environment_settings['replay_buffer_size'], batch_size=environment_settings['batch_size'])
n_warmup_batches = environment_settings['n_warmup_batches']

gamma, max_minutes, \
max_steps, goal_mean_100_reward =  environment_settings['gamma'], environment_settings['max_minutes'], environment_settings['max_steps'], environment_settings['goal_mean_100_reward']

path = ''
opponentpath = ''
savingpath = ''
generation = False
for i,arg in enumerate(sys.argv):
    if arg == '-o':
        path = sys.argv[i+1]
    if arg == '-s':
        savingpath = sys.argv[i+1]
    if arg == '-e':
        opponentpath = sys.argv[i+1]
    if arg == '-g':
        generation = True

agent_eval = None
if opponentpath != '':
    agent_eval = Agent(value_model_fn, opponentpath ).play 

if generation:
    SAVE_EVERY_N_EPISODES = 999999999
    n_warmup_batches = 100000000
    environment_settings['replay_buffer_size'] = environment_settings['max_steps']
    environment_settings['name'] = 'generation'

if environment_settings['name']=='auto':
    environment_settings['name'] = str(time.time())

writer = SummaryWriter('runs/'+environment_settings['name'])

agent = DQN(replay_buffer,
            value_model_fn,
            value_optimizer_fn,
            value_optimizer_lr,
            value_scheduler_fn,
            training_strategy_fn,
            evaluation_strategy_fn,
            n_warmup_batches,
            environment_settings['update_target_every_n_steps'])


if path != "":
    agent.retrieve_checkpoint(path)
    if generation:
        agent.total_step = 0


make_env_fn, make_env_kargs = get_make_env_fn(agent = agent.agent, rows = rows, columns = columns, inarow = inarow)
_, make_eval_env_kargs = get_make_env_fn(agent = agent_eval, rows = rows, columns = columns, inarow = inarow)


result, final_eval_score, mean_100_value_loss, training_time, wallclock_time = agent.train(
    make_env_fn, make_env_kargs, make_eval_env_kargs, seed, gamma, max_minutes, max_steps, goal_mean_100_reward)
metrics = {'final_eval_score':final_eval_score, 'final value_loss':mean_100_value_loss, 'wallclock_time':wallclock_time}
if savingpath != '':
    agent.save_games(savingpath)
print(metrics)
hparams = environment_settings
writer.add_text('network',str(agent.online_model))
writer.add_hparams(environment_settings,metrics)
_ = BEEP()
