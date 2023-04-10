"""
Cart pole agent that learns through deep neural networks, combining a DNN with 
an environment.

WILL DO:
1) First fuse the two DNNs together
2) Then clean up the code a bit to be more organized
"""

from datetime import datetime
import random

import gymnasium
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

DNN_SAVE_FOLDER = "./dnns"

UPDATE_TARGET_DNN_EVERY_N = 4096
MAX_BUFFER_SIZE = 512

def set_device():
    """
    Set the device. CUDA if available, CPU otherwise

    Args:
    None

    Returns:
    Nothing
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("Code executes on CPU.")
    else:
        print("Code executes on GPU.")

    return device



class CartpoleDNNAgent:
    DEVICE = set_device()

    class DNN(nn.Module):

        def __init__(self):
            # Initializes a new DNN.
            super(CartpoleDNNAgent.DNN, self).__init__()
            
            self.fc_relu_stack = nn.Sequential(  
                # - an input layer of size four
                nn.Linear(4, 8),
                nn.Sigmoid(),
                nn.Linear(8, 16),
                nn.Sigmoid(),
                nn.Linear(16, 2),
            )

        def forward(self, x):
            x = self.fc_relu_stack(x)
            return x
    

    def __init__(self, l_r=0.1, d_r=0.95, r_m=None):
        self.env = gymnasium.make("CartPole-v1", render_mode=r_m)
        self.discount = d_r
        
        dnn = CartpoleDNNAgent.DNN
        self.dnn_policy = dnn().to(CartpoleDNNAgent.DEVICE)
        self.dnn_target = dnn().to(CartpoleDNNAgent.DEVICE)

        self.optim = optim.SGD(self.dnn_policy.parameters(), lr=l_r)
        
        self.state = None
        self.buffer = []
        
    # "state" is a tuple of four values
    def get_optimal_action(self, state):
        return torch.argmax(self.estimate_q_values(self.dnn_policy, state)).item()
    
    # TODO later
    def save(self, path=None):
        creation_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
        if path is None: 
            path = DNN_SAVE_FOLDER + "/Cartpole_DNN_" + creation_time + ".pth"
        torch.save(self.dnn_policy.state_dict(), path)

    # TODO later
    def load(self, path=None):
        if path is None: path = "./dnns/Cartpole_DNN.pth"
        self.dnn_policy.load_state_dict(torch.load(path))
        # move to DEVICE
        self.dnn_policy.to(CartpoleDNNAgent.DEVICE)

    # Resets the environment
    def reset(self):
        self.state, info = self.env.reset()
        return self.state, info
    
    # Takes a step in the environment with the given action while sampling and keeping
    # experience in buffer. Returns:
    #  - the resulting next state
    #  - the reward as a result of the action
    #  - if the episode was corretly terminated (boolean; terminated)
    #  - if the episode was incorrectly terminated (boolean; truncated)
    #  - additional info
    def step(self, action):
        s_before_action = self.state
        n_s, r, terminated, truncated, info = self.env.step(action)
        self.state = n_s
        # punish the last action which led to the pole falling
        if terminated: r = -1
        # Stores experience into buffer
        self.buffer.append([s_before_action, action, r, n_s])
        # return info
        return n_s, r, terminated, truncated, info
        
    # Updates the algorithm at the end of episode
    def update(self):
        minibatch_size = 128

        if len(self.buffer) > MAX_BUFFER_SIZE:
            self.buffer.reverse() #ensures that newest experience at end is kept
            self.buffer = self.buffer[:MAX_BUFFER_SIZE]

        random.shuffle(self.buffer)
        
        loss_fn = nn.MSELoss()
        target_update_count = 0

        for experience in self.buffer[:minibatch_size]:
            s, a, r, n_s = experience
            # obtain the current estimate
            current_q_val = self.estimate_q_values(self.dnn_policy, s)
            # obtain the future best reward
            best_next_reward = torch.max(self.estimate_q_values(self.dnn_target, n_s))
            # obtain the updated q value based on the best reward, discount and reward r
            updated_q_val = r + self.discount * best_next_reward
            if a == 0:
                q_val_to_pass_to_loss_fun = torch.tensor([
                    updated_q_val,
                    current_q_val[0, 1]
                ]).unsqueeze(0).to(CartpoleDNNAgent.DEVICE)
            else:
                q_val_to_pass_to_loss_fun = torch.tensor([
                    current_q_val[0, 0],
                    updated_q_val
                ]).unsqueeze(0).to(CartpoleDNNAgent.DEVICE)
            # calculate the loss
            loss = loss_fn(current_q_val, q_val_to_pass_to_loss_fun)
            # propagate the result
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # update the target dnn appropriately
            if target_update_count % UPDATE_TARGET_DNN_EVERY_N == 0:
                self.update_target()
            target_update_count += 1
    
    # Updates the algorithm at the end of episode
    def update_the_new_way(self):
        minibatch_size = 32
        episodes = 3

        if len(self.buffer) > MAX_BUFFER_SIZE:
            self.buffer.reverse() #ensures that newest experience at end is kept
            self.buffer = self.buffer[:MAX_BUFFER_SIZE]

        random.shuffle(self.buffer)
        
        loss_fn = nn.MSELoss()
        target_update_count = 0

        for i in range(episodes):
            minibatch = torch.tensor(
                self.buffer[i * minibatch_size : (i + 1) * minibatch_size]
            ).to(CartpoleDNNAgent.DEVICE)
            # s, a, r, n_s
            # take all s and compute their estimated q values
            # take all n_s and estimate their max next reward
            # get the corresponding updated q val by getting r
            # from estimate of q val and action, make a mask 

            # obtain the current estimate
            current_q_val = self.estimate_q_values(self.dnn_policy, s)
            # obtain the future best reward
            best_next_reward = torch.max(self.estimate_q_values(self.dnn_target, n_s))
            # obtain the updated q value based on the best reward, discount and reward r
            updated_q_val = r + self.discount * best_next_reward
            if a == 0:
                q_val_to_pass_to_loss_fun = torch.tensor([
                    updated_q_val,
                    current_q_val[0, 1]
                ]).unsqueeze(0).to(CartpoleDNNAgent.DEVICE)
            else:
                q_val_to_pass_to_loss_fun = torch.tensor([
                    current_q_val[0, 0],
                    updated_q_val
                ]).unsqueeze(0).to(CartpoleDNNAgent.DEVICE)
            # calculate the loss
            loss = loss_fn(current_q_val, q_val_to_pass_to_loss_fun)
            # propagate the result
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # update the target dnn appropriately
            if target_update_count % UPDATE_TARGET_DNN_EVERY_N == 0:
                self.update_target()
            target_update_count += 1

    # Updates the target dnn by setting its values equal to that of the policy dnn
    def update_target(self):
        self.dnn_target.load_state_dict(self.dnn_policy.state_dict())
    
    # Gets a random action in the action space; has to be re-defined for each action.
    def get_random_action(self, state):
        return np.random.randint(0, self.env.action_space.n)
    
    # Closes the given environment
    def close(self):
        self.env.close()

    def estimate_q_values(self, dnn, state):
        state_tensor = torch.tensor(state).to(CartpoleDNNAgent.DEVICE)
        state_with_fake_minibatch_dim = state_tensor.unsqueeze(0)
        return dnn(state_with_fake_minibatch_dim)