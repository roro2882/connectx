
import gym
import gym_connect
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from itertools import count


import os.path
import tempfile
import random
import glob
import time
import os

from torch.utils.tensorboard import SummaryWriter


LEAVE_PRINT_EVERY_N_SECS = 120
SAVE_EVERY_N_EPISODES = 10000
EVAL_EVERY_N_EPISODES = 10
ERASE_LINE = '\x1b[2K'
EPS = 1e-6
BEEP = lambda: os.system("printf '\a'")
RESULTS_DIR = os.path.join('..', 'results')

writer = SummaryWriter()
class kaggleWrapper:
        class observation_space:
            shape = [2,4,4] # rows, columns

        class action_space:
            n = 4
        def __init__(self, env):
            self.env = env
        def reset(self):
            result =  self.env.reset()
            return result
        def step(self, action):
            result = self.env.step(int(action))
            return result


def get_make_env_fn(**kargs):
    def make_env_fn(env_name, seed=None, debug = False):
        env = gym.make(env_name,config={"rows": 4, "columns": 4, "inarow": 3})
        #if seed is not None: env.seed(seed)

        return kaggleWrapper(env)
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
        out_channels = 128

        self.input_layer = nn.Conv2d(input_dim[0], out_channels = out_channels, kernel_size=3, padding=0, stride=1)
        #self.input_layer = nn.Conv2d(2, 16,4)

        self.hidden_layers = nn.ModuleList()
        hidden_dims = list(hidden_dims)
        hidden_dims[0] = (input_dim[1]-2)*(input_dim[2]-2)*out_channels
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

    def forward(self, state):
        #breakpoint()
        x = self._format(state)
        x = self.input_layer(x)
        x = torch.flatten(x,start_dim=1)
        x = self.activation_fc(x)
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
        self.exploratory_action_taken = None
        
    def _update_temp(self):
        temp = 1 - self.t / (self.max_steps * self.exploration_ratio)
        temp = (self.init_temp - self.min_temp) * temp + self.min_temp
        temp = np.clip(temp, self.min_temp, self.init_temp)
        self.t += 1
        return temp
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
        
        temp = self._update_temp()
        writer.add_scalar("temp", temp, self.t)
        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy().squeeze()
            scaled_qs = q_values/temp
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
            q_values = model(state).cpu().detach().data.numpy().squeeze()
            return np.argmax(q_values)

class ReplayBuffer():
    def __init__(self, 
                 state_size,
                 max_size=10000, 
                 batch_size=64):
        self.ss_mem = np.empty(shape=(max_size,)+tuple(state_size), dtype=np.int8)
        self.as_mem = np.empty(shape=(max_size,1), dtype=np.int8)
        self.rs_mem = np.empty(shape=(max_size), dtype=np.float32)
        self.ps_mem = np.empty(shape=(max_size,)+tuple(state_size), dtype = np.int8)
        self.ds_mem = np.empty(shape=(max_size), dtype=np.bool8)

        self.max_size = max_size
        self.batch_size = batch_size
        self._idx = 0
        self.size = 0
    
    def store(self, sample):
        s, a, r, p, d = sample
        self.ss_mem[self._idx] = s
        self.as_mem[self._idx,0] = a
        self.rs_mem[self._idx] = r
        self.ps_mem[self._idx] = p
        self.ds_mem[self._idx] = d
        
        self._idx += 1
        self._idx = self._idx % self.max_size

        self.size += 1
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

class DQN():
    def __init__(self, 
                 replay_buffer_fn, 
                 value_model_fn, 
                 value_optimizer_fn, 
                 value_optimizer_lr,
                 value_scheduler_fn,
                 training_strategy_fn,
                 evaluation_strategy_fn,
                 n_warmup_batches,
                 update_target_every_steps):
        self.replay_buffer_fn = replay_buffer_fn
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.value_scheduler_fn = value_scheduler_fn
        self.training_strategy_fn = training_strategy_fn
        self.evaluation_strategy_fn = evaluation_strategy_fn
        self.n_warmup_batches = n_warmup_batches
        self.update_target_every_steps = update_target_every_steps
        self.total_step = 0
        nS, nA = kaggleWrapper.observation_space.shape, kaggleWrapper.action_space.n
        self.target_model = self.value_model_fn(nS, nA)
        self.online_model = self.value_model_fn(nS, nA)
        self.update_network()

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences
        
        max_a_q_sp = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        target_q_sa = rewards + (self.gamma * max_a_q_sp * (1 - is_terminals))
        q_sa = self.online_model(states).gather(1, actions)

        td_error = q_sa - target_q_sa
        value_loss = td_error.pow(2).mul(0.5).mean()
        self.value_errors.append(value_loss.item())
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

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

    def train(self, make_env_fn, make_env_kargs, seed, gamma, 
              max_minutes, max_episodes, goal_mean_100_reward):
        training_start, last_debug_time = time.time(), float('-inf')

        self.checkpoint_dir = "./checkpoints"
        self.make_env_fn = make_env_fn
        self.make_env_kargs = make_env_kargs
        self.seed = seed
        self.gamma = gamma
        
        env = self.make_env_fn(**self.make_env_kargs, seed=self.seed)
        torch.manual_seed(self.seed) ; np.random.seed(self.seed) ; random.seed(self.seed)
    
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []        
        self.episode_exploration = []
        self.value_errors = []
        

        self.value_optimizer = self.value_optimizer_fn(self.online_model, 
                                                       self.value_optimizer_lr)

        nS, nA = kaggleWrapper.observation_space.shape, kaggleWrapper.action_space.n
        self.replay_buffer = self.replay_buffer_fn(nS)
        self.training_strategy = training_strategy_fn()
        self.training_strategy.t = self.total_step
        self.evaluation_strategy = evaluation_strategy_fn() 
        self.scheduer = self.value_scheduler_fn(self.value_optimizer)
                    
        result = np.empty((max_episodes, 5))
        result[:] = np.nan
        training_time = 0
        for episode in range(1, max_episodes + 1):
            episode_start = time.time()
            
            state, is_terminal = env.reset(), False
            self.episode_reward.append(0.0)
            self.episode_timestep.append(0.0)
            self.episode_exploration.append(0.0)

            for step in count():
                state, is_terminal = self.interaction_step(state, env)
                self.total_step += 1
                min_samples = self.replay_buffer.batch_size * self.n_warmup_batches
                if len(self.replay_buffer) > min_samples:
                    experiences = self.replay_buffer.sample()
                    experiences = self.online_model.load(experiences)
                    self.optimize_model(experiences)
                
                if self.total_step % self.update_target_every_steps == 0:
                    self.update_network()
                
                if is_terminal:
                    #gc.collect()
                    break
            
            # stats
            episode_elapsed = time.time() - episode_start
            
            self.episode_seconds.append(episode_elapsed)
            self.scheduer.step()
            training_time += episode_elapsed

            if episode%EVAL_EVERY_N_EPISODES==0:
                evaluation_score, _ = self.evaluate(self.online_model, env)
                self.evaluation_scores.append(evaluation_score)
            if episode%SAVE_EVERY_N_EPISODES==0:
                self.save_checkpoint()
            
            mean_10_reward = np.mean(self.episode_reward[-10:])
            std_10_reward = np.std(self.episode_reward[-10:])
            mean_100_reward = np.mean(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])
            mean_100_value_loss = np.mean(self.value_errors[-100:])
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
            reached_max_episodes = episode >= max_episodes
            reached_goal_mean_reward = mean_100_eval_score >= goal_mean_100_reward
            training_is_over = reached_max_minutes or \
                               reached_max_episodes or \
                               reached_goal_mean_reward

            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
            debug_message = 'el {}, ep {:04}, ts {:06}, '
            debug_message += 'ar 10 {:05.1f}\u00B1{:05.1f}, '
            debug_message += '100 {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'ex 100 {:02.1f}\u00B1{:02.1f}, '
            debug_message += 'ev {:05.1f}\u00B1{:05.1f}'
            debug_message = debug_message.format(
                elapsed_str, episode-1, self.total_step, mean_10_reward, std_10_reward, 
                mean_100_reward, std_100_reward, mean_100_exp_rat, std_100_exp_rat,
                mean_100_eval_score, std_100_eval_score)
            print(debug_message, end='\r', flush=True)
            writer.add_scalar("value loss", mean_100_value_loss, self.total_step)
            writer.add_scalar("time", int(time.time() - training_start), self.total_step)
            writer.add_scalar("reward", mean_100_reward, self.total_step)
            writer.add_scalar("scheduer", float(self.scheduer.get_last_lr()[0]), self.total_step)
            writer.add_scalar("evaluation score", mean_100_eval_score, self.total_step)
            if reached_debug_time or training_is_over:
                print(ERASE_LINE + debug_message, flush=True)
                last_debug_time = time.time()
            if training_is_over:
                if reached_max_minutes: print(u'--> reached_max_minutes \u2715')
                if reached_max_episodes: print(u'--> reached_max_episodes \u2715')
                if reached_goal_mean_reward: print(u'--> reached_goal_mean_reward \u2713')
                break
                
        final_eval_score, score_std = self.evaluate(self.online_model, env, n_episodes=200)
        wallclock_time = time.time() - training_start
        print('Training complete.')
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
              ' {:.2f}s wall-clock time.\n'.format(
                  final_eval_score, score_std, training_time, wallclock_time))
        self.demo(10,env,self.online_model)
        del env
        return result, final_eval_score, training_time, wallclock_time
    

    def demo(self, n_demo,env, model):
        for episode in range(1, n_demo+ 1):
            episode_start = time.time()
            
            state, is_terminal = env.reset(), False
            self.episode_reward.append(0.0)
            self.episode_timestep.append(0.0)
            self.episode_exploration.append(0.0)

            print(state[0]+2*state[1])
            for step in count():
                a = self.evaluation_strategy.select_action(model, state)
                state, reward, is_terminal, _ = env.step(a)
                print(state[0]+2*state[1])
                print(reward)
                if is_terminal:
                    #gc.collect()
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
                   os.path.join(self.checkpoint_dir, 'model.{}.tar'.format(time.ctime())))


dqn_results = []
best_agent, best_eval_score = None, float('-inf')
environment_settings = {
    'env_name': 'gym_connect',
    'gamma': 0.95,
    'max_minutes': 30,
    'max_episodes': 50000,
    'goal_mean_100_reward': 1.1
}

seed = 13

value_model_fn = lambda nS, nA: FCQ(nS, nA, hidden_dims=(64, 128))
value_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
value_optimizer_lr = 0.00005
value_scheduler_fn = lambda value_optimizer : torch.optim.lr_scheduler.ExponentialLR(value_optimizer, gamma=1.0)

# training_strategy_fn = lambda: EGreedyStrategy(epsilon=0.5)
# training_strategy_fn = lambda: EGreedyLinearStrategy(init_epsilon=1.0,
#                                                      min_epsilon=0.3, 
#                                                      max_steps=20000)
training_strategy_fn = lambda: SoftMaxStrategy(init_temp=1.0, 
                                                min_temp=0.2, 
                                                exploration_ratio=0.8, 
                                                max_steps=100000)
evaluation_strategy_fn = lambda: GreedyStrategy()

replay_buffer_fn = lambda nS: ReplayBuffer(nS,max_size=10000, batch_size=128)
n_warmup_batches = 5
update_target_every_steps = 100

env_name, gamma, max_minutes, \
max_episodes, goal_mean_100_reward = environment_settings.values()
agent = DQN(replay_buffer_fn,
            value_model_fn,
            value_optimizer_fn,
            value_optimizer_lr,
            value_scheduler_fn,
            training_strategy_fn,
            evaluation_strategy_fn,
            n_warmup_batches,
            update_target_every_steps)

make_env_fn, make_env_kargs = get_make_env_fn(env_name=env_name)
import sys
path = ''
for i,arg in enumerate(sys.argv):
    if arg == '-o':
        path = sys.argv[i+1]
if path is not "":
    agent.retrieve_checkpoint(path)
result, final_eval_score, training_time, wallclock_time = agent.train(
    make_env_fn, make_env_kargs, seed, gamma, max_minutes, max_episodes, goal_mean_100_reward)
_ = BEEP()
