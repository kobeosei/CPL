import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

class QLearningAgent:
    """
    Q-Learning Agent for reinforcement learning.
    """
    def __init__(self, state_space, action_space, lr=0.1, gamma=0.99, epsilon=0.1):
        """
        Initializes the Q-learning agent.

        Args:
            state_space (int): Number of unique states.
            action_space (int): Number of available actions.
            lr (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Exploration probability.
        """
        self.state_space = state_space
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((state_space, action_space))

    def choose_action(self, state):
        """
        Chooses an action based on epsilon-greedy policy.

        Args:
            state (int): Current state index.

        Returns:
            int: Chosen action index.
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)  # Explore
        return np.argmax(self.q_table[state-1])  # Exploit

    def update(self, state, action, reward, next_state, done):
        """
        Updates the Q-table using the Q-learning algorithm.

        Args:
            state (int): Current state index.
            action (int): Chosen action.
            reward (float): Received reward.
            next_state (int): Next state index.
            done (bool): Whether the episode is finished.
        """
        best_next_action = np.argmax(self.q_table[next_state])
        target = reward + (self.gamma * self.q_table[next_state, best_next_action] * (1 - done))
        self.q_table[state-1, action] += self.lr * (target - self.q_table[state-1, action])


def generate_demonstration_pairs(trajectories, rewards, beta=-1.0):
    unique_trajectories = np.unique(trajectories)
    demonstration_pairs = []
    labels = []
    
    for i in range(len(unique_trajectories)-1):
        traj = unique_trajectories[i]
        comp_traj = unique_trajectories[i + 1] 
        indices = np.where(trajectories == traj)[0]
        reward = reg_based_model(rewards[indices])
        
        comp_indices = np.where(trajectories == comp_traj)[0]
        comp_reward = rewards[comp_indices].sum()
        
        prob_traj = np.exp(beta * reward) / (np.exp(beta * reward) + np.exp(beta * comp_reward))
        prob_comp_traj = 1 - prob_traj
        
        demonstration_pairs.append((traj, comp_traj))
        
        if prob_traj > prob_comp_traj:
            labels.append(1)
        if prob_traj < prob_comp_traj:
            labels.append(0)
   
    return demonstration_pairs, labels


def generate_preference_pairs(trajectories, rewards):
    """
    Generate preference pairs based on trajectories and rewards.

    Args:
        trajectories (array): List of trajectories.
        rewards (array): List of corresponding rewards.

    Returns:
        list: Preference pairs and labels.
    """
    unique_trajectories = np.unique(trajectories)
    preference_pairs = []
    labels = []
    
    for i in range(len(unique_trajectories) - 1):
        traj1 = unique_trajectories[i]
        traj2 = unique_trajectories[i + 1]
        
        indices1 = np.where(trajectories == traj1)[0]
        indices2 = np.where(trajectories == traj2)[0]
        
        reward1 = reg_based_model(rewards[indices1])
        reward2 = reg_based_model(rewards[indices2])
        
        prob = softmax([reward1, reward2])
        
        if prob[0] > prob[1]:
            preference_pairs.append((traj1, traj2))
            labels.append(1)
        else:
            preference_pairs.append((traj2, traj1))
            labels.append(0)
    
    return preference_pairs, labels


def reg_based_model(rewards):
    """
    A regularization-based model for reward processing.

    Args:
        rewards (array): Reward array to be processed.

    Returns:
        float: The probability of action.
    """
    r_pos = 0
    r_neg = 0
    for k in rewards:
        if k > 0:
            r_pos -= np.log(k)  # Taking log for positive rewards
        elif k <= 0:
            r_neg -= np.log(-k)  # Taking log for negative rewards

    P_a = np.exp(r_neg) / (np.exp(r_neg) + np.exp(r_pos))  # Equation for P_a
    return P_a


def softmax(values):
    """
    Softmax function to convert values to probabilities.

    Args:
        values (list): List of values to be converted.

    Returns:
        list: Softmax probabilities.
    """
    exp_values = np.exp(values - np.max(values))  # Stabilized softmax
    return exp_values / np.sum(exp_values)


def generate_e_stop_pairs(trajectories, xs, ys, rewards, very_bad_cnstr, r_constr):
    """
    Generate e-stop pairs based on trajectories, positions, and constraints.

    Args:
        trajectories (array): List of trajectories.
        xs (array): X positions.
        ys (array): Y positions.
        rewards (array): Rewards.
        very_bad_cnstr (list): Position of the bad constraint.
        r_constr (float): Constraint radius.

    Returns:
        list: E-stop pairs and labels.
    """
    unique_trajectories = np.unique(trajectories)
    e_stop_pairs = []
    labels = []

    for i in range(len(unique_trajectories) - 1):
        traj1 = unique_trajectories[i]
        traj2 = unique_trajectories[i + 1]

        indices1 = np.where(trajectories == traj1)[0]
        indices2 = np.where(trajectories == traj2)[0]

        reward1 = reg_based_model(rewards[indices1])
        reward2 = reg_based_model(rewards[indices2])

        e_stop1 = any(np.linalg.norm([xs[idx] - very_bad_cnstr[0], ys[idx] - very_bad_cnstr[1]]) > r_constr for idx in indices1)
        e_stop2 = any(np.linalg.norm([xs[idx] - very_bad_cnstr[0], ys[idx] - very_bad_cnstr[1]]) > r_constr for idx in indices2)

        if e_stop1 and e_stop2:
            e_stop_pairs.append((traj1, traj2))
            labels.append(0)
        elif not e_stop1 and not e_stop2:
            e_stop_pairs.append((traj2, traj1))
            labels.append(1)
        else:
            prob = softmax([reward1, reward2])
            if prob[0] < prob[1]:
                e_stop_pairs.append((traj1, traj2))
                labels.append(0)
            else:
                e_stop_pairs.append((traj2, traj1))
                labels.append(1)

    return e_stop_pairs, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train QLearning model with or without rendering")
    parser.add_argument('--render', action='store_true', help="Turn on rendering")
    parser.add_argument('--feedback', choices=['comparisons', 'demonstrations', 'e_stop'], required=True, help="Type of feedback to use")
    args = parser.parse_args()

    # Load the CSV data
    data = pd.read_csv('traj_with_rewards.csv')
    trajectories = data['Trajectory'].values
    xs = data['X'].values
    ys = data['Y'].values
    rewards = data['Reward'].values

    # State and action space dimensions
    state_dim = len(np.unique(trajectories))
    action_dim = 2  # Example: [0, 1] for binary actions

    # Generate feedback pairs based on user input
    if args.feedback == 'comparisons':
        pairs, labels = generate_preference_pairs(trajectories, rewards)
    elif args.feedback == 'demonstrations':
        pairs, labels = generate_demonstration_pairs(trajectories, rewards)
    elif args.feedback == 'e_stop':
        pairs, labels = generate_e_stop_pairs(trajectories, xs, ys, rewards, very_bad_cnstr=[14, 5], r_constr=3)

    # Train/validation split
    train_pairs, val_pairs, train_labels, val_labels = train_test_split(pairs, labels, test_size=0.4, random_state=42)

    # Q-Learning setup
    agent = QLearningAgent(state_space=state_dim, action_space=action_dim)
    episodes = 100
    train_losses, val_accs, val_precisions = [], [], []

    # TensorBoard setup
    writer = SummaryWriter(log_dir=f"runs/{args.feedback}_run")

    for episode in range(episodes):
        total_loss = 0
        correct, total = 0, 0

        for (state, next_state), label in zip(train_pairs, train_labels):
            action = agent.choose_action(state)
            next_state = next_state - 1
            reward = 1 if label == action else -1  # Reward based on feedback
            done = False  # You can adapt this to end episodes
            agent.update(state, action, reward, next_state, done)
            total_loss += reward

        # Validation
        val_correct = sum([1 for (state, _), label in zip(val_pairs, val_labels) if agent.choose_action(state) == label])
        val_acc = val_correct / len(val_pairs)
        val_precision = val_correct / (val_correct + (len(val_pairs) - val_correct))  # Simplified precision

        train_losses.append(total_loss)
        val_accs.append(val_acc)
        val_precisions.append(val_precision)

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', total_loss, episode)
        writer.add_scalar('Accuracy/validation', val_acc, episode)
        writer.add_scalar('Precision/validation', val_precision, episode)

        print(f"Episode {episode + 1}/{episodes}, Loss: {total_loss:.2f}, Val Acc: {val_acc:.2f}, Val Prec: {val_precision:.2f}")

    # Close the TensorBoard writer
    writer.close()

