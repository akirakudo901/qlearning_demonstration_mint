"""
Cart pole agent that learns through deep neural networks, combining a DNN with 
an environment.
"""

import numpy as np
import gymnasium

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CartpoleDNNAgent:

    class DNN(nn.Module):

        def __init__(self):
            # Initializes a new DNN.
            super(CartpoleDNNAgent.DNN, self).__init__()
            
            # - an input layer of size four
            self.fc1 = nn.Linear(4, 5)
            # - one hidden layer with 8 neurons
            # self.fc2 = nn.Linear(8, 8)
            # - another hidden layer with 8 more neurons and 2 outputs
            self.fc3 = nn.Linear(5, 1)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            # x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    def __init__(self, l_r=0.1, d_r=0.95, r_m=None):
        self.env = gymnasium.make("CartPole-v1", render_mode=r_m)
        self.learning_rate = l_r
        self.discount_rate = d_r
        self.dnn_action_zero = CartpoleDNNAgent.DNN()
        self.dnn_action_one = CartpoleDNNAgent.DNN()

        self.optim_zero = optim.SGD(self.dnn_action_zero.parameters(), lr=self.learning_rate)
        self.optim_one = optim.SGD(self.dnn_action_one.parameters(), lr=self.learning_rate)

        self.state = None
        self.episode_state_action_pairs = []
        self.episode_reward = 0
    
    # "state" is a tuple of four values
    def get_optimal_action(self, state):
        if (self.estimate_q_values_action_zero(state).item() <= self.estimate_q_values_action_one(state).item()):
            return 1
        else:
            return 0
    
    # TODO later
    def save(self):
        pass

    # TODO later
    def load(self):
        pass

    # Resets the environment
    def reset(self):
        self.state, info = self.env.reset()
        self.episode_state_action_pairs = []
        self.episode_reward = 0
        return self.state, info
    
    # Takes a step in the environment with the given action, and returns:
    #  - the resulting next state
    #  - the reward as a result of the action
    #  - if the episode was corretly terminated (boolean; terminated)
    #  - if the episode was incorrectly terminated (boolean; truncated)
    #  - additional info
    def step(self, action):
        s_before_action = self.state
        n_s, r, terminated, truncated, info = self.env.step(action)
        self.state = n_s
        # Update info required for retraining of Q table at the end
        self.episode_state_action_pairs.append([s_before_action, action, n_s])
        self.episode_reward += r
        # return info
        return n_s, r, terminated, truncated, info
    
    # Updates the algorithm accordingly
    def update(self):
        self.episode_state_action_pairs.reverse()
        for pair in self.episode_state_action_pairs:
            s, a, n_s = pair[0], pair[1], pair[2]
            estimate_q_values = self.estimate_q_values_action_zero if (a == 0) else self.estimate_q_values_action_one
            appropriate_optim = self.optim_zero if (a == 0) else self.optim_one
            
            best_next_reward = torch.max(torch.tensor( [
                self.estimate_q_values_action_zero(n_s).item(), 
                self.estimate_q_values_action_one(n_s).item()
                ] ))
            
            # print("chosen action: ", a)
            # print("1st: \n")
            # print("dnn_zero: ", list(self.dnn_action_zero.parameters()), "\n")
            # print("dnn_one : ", list(self.dnn_action_one.parameters()),  "\n")

            appropriate_optim.zero_grad()
            
            curr_q = estimate_q_values(s)
            
            corrected_reward = self.learning_rate * (self.episode_reward + self.discount_rate * best_next_reward)
            dim_correct = corrected_reward.unsqueeze(0).unsqueeze(0)
            # corrected_reward_sq = corrected_reward * corrected_reward
            # other_action_q = q_for_both_action[1 - a].item() # 1-a is 0 if a = 1, 1 if a = 0
            
            # print("2nd: \n")
            # print("dnn_zero: ", list(self.dnn_action_zero.parameters()), "\n")
            # print("dnn_one : ", list(self.dnn_action_one.parameters()),  "\n")

            loss_fn = nn.MSELoss()
            loss = loss_fn(curr_q, dim_correct)
            loss.backward()

            # print("3rd: \n")
            # print("dnn_zero: ", list(self.dnn_action_zero.parameters()), "\n")
            # print("dnn_one : ", list(self.dnn_action_one.parameters()),  "\n")
            
            # optimizer.zero_grad()   # zero the gradient buffers
            # output = net(input)
            # loss = criterion(output, target)
            # loss.backward()
            # optimizer.step()    # Does the update
    
    # Gets a random action in the action space; has to be re-defined for each action.
    def get_random_action(self, state):
        return np.random.randint(0, self.env.action_space.n)
    
    # Closes the given environment
    def close(self):
        self.env.close()

    def estimate_q_values_action_zero(self, state):
        state_tensor = torch.tensor(state)
        state_with_fake_minibatch_dim = state_tensor.unsqueeze(0)
        return self.dnn_action_zero(state_with_fake_minibatch_dim) 

    def estimate_q_values_action_one(self, state):
        state_tensor = torch.tensor(state)
        state_with_fake_minibatch_dim = state_tensor.unsqueeze(0)
        return self.dnn_action_one(state_with_fake_minibatch_dim)