import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

# Training parameters
goal = np.array([20.0, 20.0])
bad_cnstr = np.array([5.5, 14.5])
very_bad_cnstr = np.array([14.5, 5.5])
hidden_dim = 256
num_epochs = 1000
num_episodes = 60

# Initialize environment
env = Point('name', 20, -1, 20, -1, goal, bad_cnstr, very_bad_cnstr, -10, -100)

# Initialize policy network
policy_net = PolicyNetwork(2, 2, hidden_dim).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=3e-4)

def train_policy_network():
    for epoch in range(num_epochs):
        state = env.reset()
        for t in range(num_episodes):
            action = policy_net.get_action(state)
            action = np.clip(action, -1, 1)
            next_state, reward, done, _ = env.step(action)

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)
            mean, log_std = policy_net(state_tensor)
            std = log_std.exp()
            normal = Normal(mean, std)
            log_prob = normal.log_prob(action_tensor).sum()
            loss = -log_prob * reward

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            if done:
                break

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

# Train the network
train_policy_network()

# Save the trained model
torch.save(policy_net.state_dict(), 'model/policy_net_cnstr_-10.0_-100.0.pth')
