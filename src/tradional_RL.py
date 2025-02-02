import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class QLearningAgent:
    def __init__(self, states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.states = states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((len(states), n_actions))
        self.state_to_index = {state: index for index, state in enumerate(states)}

    def choose_action(self, state):
        state_idx = self.state_to_index[state]
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            return np.argmax(self.q_table[state_idx])

    def update_q_value(self, state, action, reward, next_state):
        state_idx = self.state_to_index[state]
        next_state_idx = self.state_to_index[next_state]
        
        best_next_action = np.argmax(self.q_table[next_state_idx])
        td_target = reward + self.gamma * self.q_table[next_state_idx, best_next_action]
        td_error = td_target - self.q_table[state_idx, action]
        self.q_table[state_idx, action] += self.alpha * td_error
        return td_error

    def train(self, data, episodes, validation_data=None, writer=None):
        training_losses = []
        validation_losses = []
        accuracies = []

        for episode in range(episodes):
            total_td_error = 0
            total_reward = 0
            for idx in range(len(data)-1):
                state = (data.iloc[idx]['X'], data.iloc[idx]['Y'])
                next_state = (data.iloc[idx+1]['X'], data.iloc[idx+1]['Y'])
                action = self.choose_action(state)
                reward = data.iloc[idx]['Reward']
                td_error = self.update_q_value(state, action, reward, next_state)
                total_td_error += abs(td_error)
                total_reward += reward
            avg_td_error = total_td_error / len(data)
            training_losses.append(avg_td_error)

            if validation_data is not None:
                validation_loss, accuracy = self.validate(validation_data)
                validation_losses.append(validation_loss)
                accuracies.append(accuracy)

            # Logging to TensorBoard
            if writer:
                writer.add_scalar('Training Loss', avg_td_error, episode)
                if validation_data is not None:
                    writer.add_scalar('Validation Loss', validation_loss, episode)
                    writer.add_scalar('Accuracy', accuracy, episode)

            if episode % 1 == 0:
                print(f"Episode {episode}/{episodes}, Training Loss: {avg_td_error:.4f}")
        
        return training_losses, validation_losses, accuracies

    def validate(self, data):
        total_td_error = 0
        correct_action = 0
        total_steps = len(data)-1
        for idx in range(total_steps):
            state = (data.iloc[idx]['X'], data.iloc[idx]['Y'])
            next_state = (data.iloc[idx+1]['X'], data.iloc[idx+1]['Y'])
            action = self.choose_action(state)
            reward = data.iloc[idx]['Reward']
            td_error = self.update_q_value(state, action, reward, next_state)
            total_td_error += abs(td_error)
            if action == np.argmax(self.q_table[self.state_to_index[state]]):
                correct_action += 1
        validation_loss = total_td_error / total_steps
        accuracy = correct_action / total_steps
        return validation_loss, accuracy

    def get_best_action(self, state):
        state_idx = self.state_to_index[state]
        return np.argmax(self.q_table[state_idx])

# Load data
data = pd.read_csv("traj_with_rewards.csv")

# Define unique states
unique_states = list(set(zip(data['X'], data['Y'])))

# Initialize agent
agent = QLearningAgent(states=unique_states, n_actions=4)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir="runs/ql_training")

# Train agent
training_losses, validation_losses, accuracies = agent.train(data, episodes=100, validation_data=data, writer=writer)

# Close the TensorBoard writer
writer.close()

# Plot training and validation losses
plt.figure(1)

plt.subplot(1, 2, 1)
plt.plot(training_losses, label="Training Loss")
plt.xlabel("Episodes")
plt.ylabel("Loss")
plt.title("Training Loss Progress")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(validation_losses, label="Validation Loss")
plt.xlabel("Episodes")
plt.ylabel("Loss")
plt.title("Validation Loss Progress")
plt.legend()

# Save and display plot
plt.savefig("traditional_RL_tensorboard.png")
plt.show()

