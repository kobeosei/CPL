import math
import random
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from custom_env_w import Point

# Set device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)

        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.layer_norm3 = nn.LayerNorm(hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, state):
        x = self.swish(self.layer_norm1(self.linear1(state)))
        x = self.swish(self.layer_norm2(self.linear2(x)))
        x = self.swish(self.layer_norm3(self.linear3(x)))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample().to(device)
        action = torch.tanh(mean + std * z)

        action = action.cpu().detach().numpy()
        return action[0]

# Parameters
goal = np.array([20.0, 20.0])
bad_cnstr = np.array([5.5, 14.5])
very_bad_cnstr = np.array([14.5, 5.5])
r_constr = 3
N_tot_traj = 1500
N_max_episode = 60
N_good_traj = 750

# Initialize environment
env = Point('name', 20, -1, 20, -1, goal, bad_cnstr, very_bad_cnstr, 0, 0)

# Initialize policy networks
action_dim = 2
state_dim = 2
hidden_dim = 256

policy_net_good = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
policy_net_nocnstr = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

# Load pre-trained models
policy_net_good.load_state_dict(torch.load('model/policy_net_cnstr_-10.0_-100.0.pth'))
policy_net_nocnstr.load_state_dict(torch.load('model/policy_net_cnstr_0.0_0.0.pth'))

# Generate trajectories
traj_dict = {}
cum_reward = []
states_list = []
traj_index = []

for iteration in range(N_tot_traj):
    print(iteration)
    states_l = []
    state = np.zeros(2)
    if np.random.rand() < 0.5:
        state[0] = np.random.uniform(9, 19)
        state[1] = 1 * np.random.rand()
    else:
        state[0] = 1 * np.random.rand()
        state[1] = np.random.uniform(9, 19)

    env.set_position(state[0], state[1])
    states_l.append(state)
    cum_r = 0

    for t in range(N_max_episode):
        if iteration <= N_good_traj:
            action = policy_net_good.get_action(state)
            ind = 1
        else:
            action = policy_net_nocnstr.get_action(state)
            ind = 2

        action = np.clip(action, -1, 1)
        state, reward, done, _ = env.step(action)
        states_l.append(state)
        cum_r += reward
        if done or t >= N_max_episode - 1:
            cum_reward.append(cum_r)
            states_list.append(states_l)
            traj_index.append(ind)
            break
#Classify Trajectories Based on Constraints
traj_index = np.zeros(len(states_list))
for cnt, traj in enumerate(states_list):
    traj_index[cnt] = 1
    for tr in traj:
        if np.linalg.norm(tr - bad_cnstr) < r_constr:
            traj_index[cnt] = 2
        if np.linalg.norm(tr - very_bad_cnstr) < r_constr:
            traj_index[cnt] = 3

traj_dict['trajectories'] = states_list
traj_dict['rewards'] = cum_reward
traj_dict['traj_index'] = traj_index

#print(traj_dict['trajectories'],traj_dict['rewards'],traj_dict['traj_index'])
# Save trajectories to pickle file
with open('data/trajectories_m10_m100.pickle', 'wb') as handle:
    pickle.dump(traj_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
