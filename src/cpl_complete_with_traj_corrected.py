import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# Define the Point environment
class Point(object):
    def __init__(self, x_max, x_min, y_max, y_min, target, bad_cnstr, very_bad_cnstr, pen_1, pen_2, r_constr=3, r_target=3.0):
        self.x = 0
        self.y = 0
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.target = target
        self.bad_cnstr = bad_cnstr
        self.very_bad_cnstr = very_bad_cnstr
        self.observation_space = np.array([2])
        self.action_space = np.array([2])
        self.icon_w = 0
        self.icon_h = 0
        self.w = np.array([1, pen_1, pen_2, -1])
        self.num_features = 4
        self.w_nom = np.array([self.w[0], 0, 0, self.w[3]])
        self.r_constr = r_constr
        self.r_target = r_target

    def reset(self):
        self.x = 1 * np.random.rand()
        self.y = 1 * np.random.rand()
        self.alive = 1
        return np.array([self.x, self.y])

    def step(self, a):
        self.x += a[0]
        self.y += a[1]
        self.x = self.clamp(self.x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(self.y, self.y_min, self.y_max - self.icon_h)
        info = []

        s_ = np.array([self.x, self.y])
        targ_feat = 1 / np.linalg.norm(s_ - self.target)
        obst_1_feat = int(np.linalg.norm(s_ - self.bad_cnstr) < self.r_constr)
        obst_2_feat = int(np.linalg.norm(s_ - self.very_bad_cnstr) < self.r_constr)
        done = False
        if np.linalg.norm(s_ - self.target) <= self.r_target:
            done = True
        living_feat = 1 - done

        feat = np.array([targ_feat, obst_1_feat, obst_2_feat, living_feat])
        r = np.dot(self.w, feat)
        return s_, r, done, info

    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)

    def render(self, trajectory_number=None):
        plt.plot(self.x, self.y, 'bo')  # Agent as a blue dot
        plt.plot(self.target[0], self.target[1], 'go')  # Target as a green dot
        target_circle = plt.Circle(self.target, self.r_target, color='green', alpha=0.3)
        plt.gca().add_artist(target_circle)
        plt.plot(self.bad_cnstr[0], self.bad_cnstr[1], 'yo')  # Bad constraint as a green dot
        bad_circle = plt.Circle(self.bad_cnstr, self.r_constr, color='orange', alpha=0.3)
        plt.gca().add_artist(bad_circle)
        plt.plot(self.very_bad_cnstr[0], self.very_bad_cnstr[1], 'ro')  # Very bad constraint as a magenta dot
        very_bad_circle = plt.Circle(self.very_bad_cnstr, self.r_constr, color='red', alpha=0.3)
        plt.gca().add_artist(very_bad_circle)
        plt.xlim(self.x_min, self.x_max)
        plt.ylim(self.y_min, self.y_max)
        if trajectory_number is not None:
            plt.title(f"Trajectory Number: {trajectory_number}")
        plt.draw()
        plt.pause(0.001)  # Adjust this value to slow down the animation
        plt.clf()

# Load the CSV data
data = pd.read_csv('traj_with_rewards.csv')
trajectories = data['Trajectory'].values
xs = data['X'].values
ys = data['Y'].values
rewards = data['Reward'].values

# Define the softmax function for computing preferences
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Create pairs of trajectories with preference labels
def generate_preference_pairs(trajectories, rewards):
    unique_trajectories = np.unique(trajectories)
    preference_pairs = []
    labels = []
    
    for i in range(len(unique_trajectories) - 1):
        traj1 = unique_trajectories[i]
        traj2 = unique_trajectories[i + 1]
        
        indices1 = np.where(trajectories == traj1)[0]
        indices2 = np.where(trajectories == traj2)[0]
        
        reward1 = rewards[indices1].sum()
        reward2 = rewards[indices2].sum()
        
        prob = softmax([reward1, reward2])
        
        if prob[0] > prob[1]:
            preference_pairs.append((traj1, traj2))
            labels.append(1)
        else:
            preference_pairs.append((traj2, traj1))
            labels.append(0)
    
    return preference_pairs, labels

# Create pairs of trajectories with demonstration labels
def generate_demonstration_pairs(trajectories, rewards, beta=1.0):
    unique_trajectories = np.unique(trajectories)
    demonstration_pairs = []
    labels = []

    for i, traj in enumerate(unique_trajectories):
        indices = np.where(trajectories == traj)[0]
        reward = rewards[indices].sum()
        
        for j, comp_traj in enumerate(unique_trajectories):
            if comp_traj != traj:
                comp_indices = np.where(trajectories == comp_traj)[0]
                comp_reward = rewards[comp_indices].sum()
                
                prob_traj = np.exp(beta * reward) / (np.exp(beta * reward) + np.exp(beta * comp_reward))
                prob_comp_traj = 1 - prob_traj
                
                demonstration_pairs.append((traj, comp_traj))
                labels.append(1 if prob_traj > prob_comp_traj else 0)

    return demonstration_pairs, labels

# Define the Preference Dataset class
class PreferenceDataset(Dataset):
    def __init__(self, trajectories, xs, ys, rewards, pairs, labels):
        self.trajectories = trajectories
        self.xs = xs
        self.ys = ys
        self.rewards = rewards
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        traj1, traj2 = self.pairs[idx]
        label = self.labels[idx]
        
        indices1 = np.where(self.trajectories == traj1)[0]
        indices2 = np.where(self.trajectories == traj2)[0]

        state1 = torch.tensor(np.vstack((self.xs[indices1], self.ys[indices1])).T, dtype=torch.float32)
        state2 = torch.tensor(np.vstack((self.xs[indices2], self.ys[indices2])).T, dtype=torch.float32)
        
        reward1 = torch.tensor(self.rewards[indices1], dtype=torch.float32)
        reward2 = torch.tensor(self.rewards[indices2], dtype=torch.float32)
        
        label = torch.tensor([label], dtype=torch.float32)

        return state1, state2, label, reward1, reward2

# Custom collate function to handle varying trajectory sizes
def custom_collate(batch):
    state1_batch = [item[0] for item in batch if isinstance(item[0], torch.Tensor)]
    state2_batch = [item[1] for item in batch if isinstance(item[1], torch.Tensor)]
    labels_batch = [item[2] for item in batch if isinstance(item[2], torch.Tensor)]
    rewards1_batch = [item[3] for item in batch if isinstance(item[3], torch.Tensor)]
    rewards2_batch = [item[4] for item in batch if isinstance(item[4], torch.Tensor)]

    if not state1_batch or not state2_batch:
        return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)

    max_len = max(max(s.size(0) for s in state1_batch), max(s.size(0) for s in state2_batch))

    padded_state1_batch = torch.stack([torch.cat([s, torch.zeros(max_len - s.size(0), s.size(1))]) for s in state1_batch])
    padded_state2_batch = torch.stack([torch.cat([s, torch.zeros(max_len - s.size(0), s.size(1))]) for s in state2_batch])

    padded_reward1_batch = torch.stack([torch.cat([r, torch.zeros(max_len - r.size(0))]) for r in rewards1_batch])
    padded_reward2_batch = torch.stack([torch.cat([r, torch.zeros(max_len - r.size(0))]) for r in rewards2_batch])

    return padded_state1_batch, padded_state2_batch, torch.stack(labels_batch), padded_reward1_batch, padded_reward2_batch

# Define the Preference Model
class PreferenceModel(nn.Module):
    def __init__(self, state_dim):
        super(PreferenceModel, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# CPL Loss function with binary cross entropy
def cpl_loss(log_probs1, log_probs2, labels, gamma=0.99, alpha=0.1):
    exp_log_prob1 = torch.exp(gamma * alpha * log_probs1)
    exp_log_prob2 = torch.exp(gamma * alpha * log_probs2)
    probs = exp_log_prob1 / (exp_log_prob1 + exp_log_prob2)
    loss = -torch.mean(labels * torch.log(probs) + (1 - labels) * torch.log(1 - probs))
    return loss

# Evaluation function to compute evaluation metrics
def evaluate(model, data_loader):
    model.eval()
    total_reward1 = 0
    total_reward2 = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for state1, state2, label, reward1, reward2 in data_loader:
            batch_size, max_len, state_dim = state1.size()
            state1_flat = state1.view(batch_size * max_len, state_dim)
            state2_flat = state2.view(batch_size * max_len, state_dim)

            # Forward pass
            pref_scores1 = model(state1_flat).view(batch_size, max_len)
            pref_scores2 = model(state2_flat).view(batch_size, max_len)

            pref_scores1_mean = pref_scores1.mean(dim=1)
            pref_scores2_mean = pref_scores2.mean(dim=1)

            predictions = (pref_scores1_mean > pref_scores2_mean).float()
            correct += (predictions == label).sum().item()
            total += label.size(0)

            total_reward1 += reward1.sum().item()
            total_reward2 += reward2.sum().item()

    accuracy = correct / total
    avg_reward1 = total_reward1 / total
    avg_reward2 = total_reward2 / total

    return accuracy, avg_reward1, avg_reward2

# Training function for CPL
def train_cpl(model, optimizer, data_loader, epochs=100, gamma=0.99, alpha=0.1, log_dir='./logs', render=False):
    writer = SummaryWriter(log_dir)
    model.train()

    env = Point(x_max=20, x_min=0, y_max=20, y_min=0, target=[20, 20], bad_cnstr=[5, 14], very_bad_cnstr=[14, 5], pen_1=-10, pen_2=-20)
    
    for epoch in range(epochs):
        total_loss = 0.0
        selected_trajectory = None

        # Shuffle pairs and labels at the beginning of each epoch
        pairs_indices = np.arange(len(pairs))
        np.random.shuffle(pairs_indices)
        shuffled_pairs = [pairs[i] for i in pairs_indices]
        shuffled_labels = [labels[i] for i in pairs_indices]

        for batch_idx, (state1, state2, label, reward1, reward2) in enumerate(data_loader):
            optimizer.zero_grad()

            # Flatten the states for the model input
            batch_size, max_len, state_dim = state1.size()
            state1_flat = state1.view(batch_size * max_len, state_dim)
            state2_flat = state2.view(batch_size * max_len, state_dim)

            # Forward pass
            pref_scores1 = model(state1_flat).view(batch_size, max_len)
            pref_scores2 = model(state2_flat).view(batch_size, max_len)

            # Compute CPL loss
            loss = cpl_loss(pref_scores1.mean(dim=1), pref_scores2.mean(dim=1), label, gamma, alpha)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Log the loss
            writer.add_scalar('Loss/train', loss.item(), epoch * len(data_loader) + batch_idx)

            # Log rewards for each step in the trajectories
            for step in range(max_len):
                if step < reward1.size(1):
                    writer.add_scalar(f'Reward/Trajectory1/Step_{step}', reward1[:, step].mean().item(), epoch * len(data_loader) + batch_idx)
                if step < reward2.size(1):
                    writer.add_scalar(f'Reward/Trajectory2/Step_{step}', reward2[:, step].mean().item(), epoch * len(data_loader) + batch_idx)

            # Update selected_trajectory based on the last item in the batch
            selected_trajectory = shuffled_pairs[batch_idx][0] if label[-1].item() == 1 else shuffled_pairs[batch_idx][1]

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}, Selected Trajectory: {selected_trajectory}")

        # Evaluate the model
        accuracy, eval_reward1, eval_reward2 = evaluate(model, data_loader)
        print(f"Evaluation - Accuracy: {accuracy}, Reward1: {eval_reward1}, Reward2: {eval_reward2}")

        # Log evaluation metrics
        writer.add_scalar('Eval/Accuracy', accuracy, epoch)
        writer.add_scalar('Eval/Reward/Trajectory1', eval_reward1, epoch)
        writer.add_scalar('Eval/Reward/Trajectory2', eval_reward2, epoch)

        # Render the environment if render is True
        if render:
            print(f"Rendering trajectory number: {selected_trajectory}")
            env.reset()
            done = False
            while not done:
                action = np.random.uniform(-1, 1, 2)
                state, reward, done, _ = env.step(action)
                env.render(trajectory_number=selected_trajectory)

    writer.close()

# Run the training
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CPL model with or without rendering")
    parser.add_argument('--render', action='store_true', help="Turn on rendering")
    parser.add_argument('--feedback', choices=['comparisons', 'demonstrations'], required=True, help="Type of feedback to use")
    args = parser.parse_args()

    state_dim = 2  # Dimension of the state space (X, Y)

    if args.feedback == 'comparisons':
        pairs, labels = generate_preference_pairs(trajectories, rewards)
    elif args.feedback == 'demonstrations':
        pairs, labels = generate_demonstration_pairs(trajectories, rewards)

    dataset = PreferenceDataset(trajectories, xs, ys, rewards, pairs, labels)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)

    # Initialize model, optimizer
    model = PreferenceModel(state_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_cpl(model, optimizer, data_loader, epochs=100, gamma=0.99, alpha=0.1, log_dir='./logs', render=args.render)
