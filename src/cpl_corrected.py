import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import time
import math

# Define the Point environment
# Define the Point environment
def standardize(tensor):
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / std
class Point(object):
    def __init__(self, x_max, x_min, y_max, y_min, target, bad_cnstr, very_bad_cnstr, pen_1, pen_2, r_constr=3, r_target=3.0, feedback_type=None):
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
        self.feedback_type = feedback_type

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

        # Check if the agent reaches the very bad constraint
        if self.feedback_type == 'e_stop' and obst_2_feat:
            done = True
            r = -100  # Assign a high negative reward for reaching the very bad constraint
        else:
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




# Define the softmax function for computing preferences
def softmax(x):
    #print (x)
   # time.sleep(3)
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Create pairs of trajectories with preference labels
def generate_preference_pairs(trajectories, rewards):
    unique_trajectories = np.unique(trajectories)
    print (unique_trajectories)
    
    preference_pairs = []
    labels = []
    
    for i in range(len(unique_trajectories) - 1):
        traj1 = unique_trajectories[i]
        traj2 = unique_trajectories[i + 1]
   #     print("traj1traj1traj1traj1traj1traj1traj1",traj1)
    #    print("traj2traj2traj2traj2traj2traj2traj2",traj2)
       
    
        
        indices1 = np.where(trajectories == traj1)[0]
        indices2 = np.where(trajectories == traj2)[0]
     #   print("IND1IND1IND1IND1IND1IND1IND1IND1",indices1)
      #  print("IND2IND2IND2IND2IND2IND2IND2IND2",indices2)
        
        #reward1 = rewards[indices1].sum()
        reward1 = reg_based_model (rewards[indices1])
        #reward2 = rewards[indices2].sum()
        reward2 = reg_based_model (rewards[indices1])
        
        prob = softmax([reward1, reward2])    #
        
        if prob[0] > prob[1]:
            preference_pairs.append((traj1, traj2))
            labels.append(1)
        else:
            preference_pairs.append((traj2, traj1))
            labels.append(0)
    
    return preference_pairs, labels

# Create pairs of trajectories with demonstration labels
def generate_demonstration_pairs(trajectories, rewards, beta=-1.0):
    unique_trajectories = np.unique(trajectories)
    demonstration_pairs = []
    labels = []
    
    for i in range(len(unique_trajectories)-1):
                traj = unique_trajectories[i]
                comp_traj = unique_trajectories[i + 1] 
                indices = np.where(trajectories == traj)[0]
             #   reward = rewards[indices].sum()
             
                reward = reg_based_model (rewards[indices])
               
               
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
 
def reg_based_model(rewards):    ## updated line: log of policy and implementation of eq 2 & 3 &4.
    r_pos = 0
    r_neg = 0
    for k in rewards:
        if (k > 0):
           r_pos -= np.log(k)    #taking log of the policy according to the paper 
        if (k <= 0):
           r_neg -= np.log(-k)   #taking the log of the policy if it has negative values.   
    P_a = np.exp((r_neg))/(np.exp(r_neg)+np.exp(r_pos)) # implementation of the equation 4.
    
    return P_a

# Create pairs of trajectories with e-stop feedback
def generate_e_stop_pairs(trajectories, xs, ys, rewards, very_bad_cnstr, r_constr):
    unique_trajectories = np.unique(trajectories)
    e_stop_pairs = []
    labels = []

    for i in range(len(unique_trajectories) - 1):
        traj1 = unique_trajectories[i]
        traj2 = unique_trajectories[i + 1]

        indices1 = np.where(trajectories == traj1)[0]
        indices2 = np.where(trajectories == traj2)[0]

        #reward1 = rewards[indices1].sum()
        reward1 = reg_based_model (rewards[indices1])
        #reward2 = rewards[indices2].sum()
        reward2 = reg_based_model (rewards[indices2])

        # Check if traj1 reaches the very bad constraint
        e_stop1 = any(np.linalg.norm([xs[idx] - very_bad_cnstr[0], ys[idx] - very_bad_cnstr[1]]) > r_constr for idx in indices1)
        e_stop2 = any(np.linalg.norm([xs[idx] - very_bad_cnstr[0], ys[idx] - very_bad_cnstr[1]]) > r_constr for idx in indices2)
       
        
        if e_stop1 and e_stop2:
            e_stop_pairs.append((traj1, traj2))
            labels.append(0)  # traj2 is preferred because traj1 reaches the very bad constraint
        if not e_stop1 and not e_stop2:
            e_stop_pairs.append((traj2, traj1))
            labels.append(1)  # traj1 is preferred because traj2 reaches the very bad constraint
        if ((e_stop1 and not e_stop2) or (e_stop2 and not e_stop1)):
            prob = softmax([reward1, reward2])
            if prob[0] < prob[1]:
                e_stop_pairs.append((traj1, traj2))
                labels.append(0)
            else:
                e_stop_pairs.append((traj2, traj1))
                labels.append(1)

    return e_stop_pairs, labels


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
        #reward4 = torch.log(torch.tensor(self.rewards[indices1], dtype=torch.float32))
     
    
        reward2 = torch.tensor(self.rewards[indices2], dtype=torch.float32)
        #reward5 = torch.log(torch.tensor(self.rewards[indices2], dtype=torch.float32))
        
        
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


class PreferenceModel(nn.Module):
    def __init__(self, state_dim):
        super(PreferenceModel, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=0.2) 
        
        # Activation function
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        # Weight initialization
       # self._initialize_weights()

    def _initialize_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(0.11))
            if layer.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(layer.bias, -bound, bound)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)          ## updated line: add dropout with 0.2
        x = self.relu(self.fc2(x))
        x = self.dropout(x)          ## updated line: add dropout with 0.2
        x = (self.fc3(x))  # No activation here, as this is a regression output
        return x

# CPL Loss function
def cpl_loss(log_probs1, log_probs2, labels, gamma=0.99, alpha=0.01):    ## updated line: changed cpl loss complete function based on Paper cpl loss
    assert alpha > 0.0 and alpha <= 1.0
   
    
    log_softmax = nn.LogSoftmax(dim=0)     
    adv1 = log_probs1
    adv2 = log_probs2
   # adv1 = log_softmax(log_probs1)
   # adv2 = log_softmax(log_probs2)
    logit21 = adv2 - gamma * adv1      #applying logistic methods as mentioned the maiin paper
    logit12 = adv1 - gamma * adv2          #applying logistic methods as mentioned the maiin paper
    max21 = torch.clamp(-logit21,min=0,max=None)
    max12 = torch.clamp(-logit12,min=0,max=None)  
    nlp21 = torch.log(torch.exp(-max21) + torch.exp(-logit21 - max21)) + max21
    nlp12 = torch.log(torch.exp(-max12) + torch.exp(-logit12 - max12)) + max12
    loss = 1*nlp21 +(1-1)* nlp12
    loss = torch.triu(loss, diagonal=1)     #estimation of the loss this complete function is taken from the git of the main paper
    mask = loss != 0.0
    loss = loss.sum() / mask.sum()
    
    #loss = loss[loss!=0]
   # print(log_probs1)
    
    loss = loss.mean()
    #print(loss)
  
    with torch.no_grad():
        accuracy = ((adv2 > adv1) == torch.round(labels)).float().mean()
    return loss, accuracy
def mean_squared_error(y_true, y_pred):
    return F.mse_loss(y_pred, y_true)
# Evaluation function to compute evaluation metrics
def evaluate(model, data_loader):
  #  model.eval()
    total_reward1 = 0
    total_reward2 = 0 
    correct = 0
    correct_acc = 0
    total = 0
    i_mse = 0
    accura_me = []
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
            
            log_softmax = nn.LogSoftmax(dim=0)      ## updated line: from line 368 to 389 changed updated for accuracy some parts are taken from cpl main paper
            adv1 = log_softmax(pref_scores1)   
            adv2 = log_softmax(pref_scores2)
           
            with torch.no_grad():
                 accuracy = ((pref_scores2_mean > pref_scores1_mean) == torch.round(label)).float().mean()
                 
            pref_scores1_mean = pref_scores1.mean(dim=1) #estimating the mean scroes durng evaluting 
            pref_scores2_mean = pref_scores2.mean(dim=1) #estimating the mean scroes durng evaluting
            '''
            line is 380 to 390 are used for estimation the rewards 1 and rewards 2 during the evaluting the model
            '''            
          #  try:
            pred = ((pref_scores2_mean > pref_scores1_mean) == torch.round(label))
            pred_FP = ((pref_scores2_mean < pref_scores1_mean) == torch.round(label))
            
            std_reward1 = standardize(reward1)
            std_reward2 = standardize(reward2)
            for k in range(batch_size):
                for t in pred[k]:
                  if (t==True):     # if the conditinng mathch the reward 1 is rewarded
                    total_reward1 += std_reward1[k].mean()
                    correct_acc += 1
                    total += 1
                  #total_reward2 += reward2[k].mean()
                  if (t==False):    # if the conditinng does not mathch the reward 2 is rewarded
                  #total_reward1 += reward1[k].mean()
                    total_reward2 += std_reward2[k].mean()
                    total += 1
            correct += accuracy
    precision = correct_acc/total
    accuracy =  correct/19   #estimating of the accuracy, 19 is for choosen batch size 32.
    avg_reward1 = total_reward1
    avg_reward2 = total_reward2
    return accuracy,precision,avg_reward1,avg_reward2

# Training function for CPL
def train_cpl(model, optimizer, data_loader, data_loader_val=None,epochs=100, gamma=0.0099, alpha=0.0001, log_dir='./runs', render=False):
    temp_accuracy = 0
    t = 1
    plot_loss = []
    plot_accu = []
    plot_prec = []
    
    writer = SummaryWriter(log_dir)
    model.train()
    loss_function = torch.nn.MSELoss()

    env = Point(x_max=20, x_min=0, y_max=20, y_min=0, target=[20, 20], bad_cnstr=[5, 14], very_bad_cnstr=[14, 5], pen_1=-10, pen_2=-20, feedback_type='comparisons')
    
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
          
            #print(reward1)
           # sleep(3)
            
            # Flatten the states for the model input
            batch_size, max_len, state_dim = state1.size()
            state1_flat = state1.view(batch_size * max_len, state_dim)
            state2_flat = state2.view(batch_size * max_len, state_dim)
            
            # Forward pass
            pref_scores1 = model(state1_flat).view(batch_size, max_len)
            pref_scores2 = model(state2_flat).view(batch_size, max_len)
            # Compute CPL loss
            loss,accuracy = cpl_loss(pref_scores1, pref_scores2, label, gamma, alpha)
                  
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
          
            total_loss += loss.item()
           
            # Log the loss
            writer.add_scalar('Loss/train', total_loss, epoch * len(data_loader) + batch_idx)  

            # Log rewards for each step in the trajectories
            for step in range(max_len):
                
                if step < reward1.size(1):
                    writer.add_scalar(f'Reward/Trajectory1/Step_{step}', reward1[:, step].mean().item(), epoch * len(data_loader) + batch_idx)
                if step < reward2.size(1):
                    writer.add_scalar(f'Reward/Trajectory2/Step_{step}', reward2[:, step].mean().item(), epoch * len(data_loader) + batch_idx)

            # Update selected_trajectory based on the last item in the batch
            selected_trajectory = shuffled_pairs[batch_idx][0] if label[-1].item() == 1 else shuffled_pairs[batch_idx][1]
           
        
        avg_loss = total_loss / len(data_loader)
        print(len(data_loader))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}, Selected Trajectory: {selected_trajectory}")
        
        # Evaluate the model
        if (t == 1):
           temp_model = model
           #print(data_loader_val)
           print("inside first")
        accuracy, precision, eval_reward1, eval_reward2 = evaluate(model, data_loader_val)           
               
        print(f"Evaluation - Accuracy: {accuracy}, precision: {precision}, Reward1: {eval_reward1}, Reward2: {eval_reward2}")
        plot_loss.append(avg_loss)
        plot_accu.append(accuracy)
        plot_prec.append(precision)

        # Log evaluation metrics
        writer.add_scalar('Eval/Accuracy', accuracy, epoch)
        writer.add_scalar('Eval/Reward/Trajectory1', eval_reward1, epoch)
        writer.add_scalar('Eval/Reward/Trajectory2', eval_reward2, epoch)
        writer.add_scalar('Eval/precision', precision, epoch)

        # Render the environment if render is True
        if render:
            print(f"Rendering trajectory number: {selected_trajectory}")
            env.reset()
            done = False
            while not done:
                action = np.random.uniform(-1, 1, 2)
                state, reward, done, _ = env.step(action)
                env.render(trajectory_number=selected_trajectory)
                #print(f"State: {state}, Reward: {reward}, Done: {done}")

    writer.close()
    return plot_loss, plot_accu, plot_prec


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CPL model with or without rendering")
    parser.add_argument('--render', action='store_true', help="Turn on rendering")
    parser.add_argument('--feedback', choices=['comparisons', 'demonstrations', 'e_stop'], required=True, help="Type of feedback to use")
    args = parser.parse_args()
    # Load the CSV data
    data = pd.read_csv('traj_with_rewards.csv')
    trajectories = data['Trajectory'].values
    xs = data['X'].values
    ys = data['Y'].values
    rewards = data['Reward'].values
    print(rewards)

    state_dim = 2  # Dimension of the state space (X, Y)

    if args.feedback == 'comparisons':
        pairs, labels = generate_preference_pairs(trajectories, rewards)
    elif args.feedback == 'demonstrations':
      
        pairs, labels = generate_demonstration_pairs(trajectories, rewards)
        
        
    elif args.feedback == 'e_stop':
        pairs, labels = generate_e_stop_pairs(trajectories, xs, ys, rewards, very_bad_cnstr=[14, 5], r_constr=3)

    dataset = PreferenceDataset(trajectories, xs, ys, rewards, pairs, labels)
    train_ratio = 0.6
    validation_ratio = 0.4
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    validation_size = dataset_size - train_size
    data_train,data_val = random_split(dataset,[train_size,validation_size])
    
    data_loader_train = DataLoader(data_train, batch_size=32,shuffle=True, collate_fn=custom_collate)
    data_loader_val = DataLoader(data_val, batch_size=32, shuffle=True, collate_fn=custom_collate)
    
   

    # Initialize model, optimizer
    model = PreferenceModel(state_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.1) # 0.000000001 no change in reward functions. 0.000001 : 32 batch; 0.00001: 32 batch shows changes
 
    # Train the model
    loss, accu, prec= train_cpl(model, optimizer, data_loader_train,data_loader_val,epochs=100, gamma=0.99, alpha=0.1, log_dir='./logs', render=args.render) #0099
    np.save("loss_"+args.feedback+".npy",loss)
    np.save("acc_"+args.feedback+".npy",accu)
    np.save("prec_"+args.feedback+".npy",prec)
plt.figure(1)

plt.subplot(1,3,1)
plt.plot(loss, label= "Training Loss")
plt.xlabel("Episodes")
plt.ylabel("Loss")
plt.title("Training Loss Progress")
plt.legend()

plt.subplot(1,3,2)
plt.plot(accu, label= "Accuracy")
plt.xlabel("Episodes")
plt.ylabel("Accuracy")
plt.title("Accuracy Progress")
plt.legend()

plt.subplot(1,3,3)
plt.plot(prec, label= "Precision")
plt.xlabel("Episodes")
plt.ylabel("Precision")
plt.title("Precision Progress")
plt.legend()

plt.savefig("CPL_"+args.feedback+".png")
plt.show()



