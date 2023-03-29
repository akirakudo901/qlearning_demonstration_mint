"""
Cart pole agent that learns through deep neural networks, combining a DNN with 
an environment.
"""

import gymnasium
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class CartpoleDNNAgent:

    class DNN(nn.Module):

        def __init__(self):
            # Initializes a new DNN.
            super(CartpoleDNNAgent.DNN, self).__init__()
            
            # - an input layer of size four
            self.fc1 = nn.Linear(4, 8)
            # - one hidden layer with 8 neurons, and final output of 2
            self.fc2 = nn.Linear(8, 2)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    def __init__(self, l_r=0.1, d_r=0.95, r_m=None):
        self.env = gymnasium.make("CartPole-v1", render_mode=r_m)
        self.learning_rate = l_r
        self.discount_rate = d_r
        self.dnn = CartpoleDNNAgent.DNN()

        self.state = None
        self.episode_state_action_pairs = []
        self.episode_reward = 0
    
    # "state" is a tuple of four values
    def get_optimal_action(self, state):
        return torch.argmax(self.estimate_q_values(state)).item()
    
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
            estimate = self.estimate_q_values(s)
            curr_q = estimate[a]
            best_next_reward = torch.argmax(self.estimate_q_values(n_s))
            corrected_reward = curr_q + self.learning_rate * (self.episode_reward + self.discount_rate * (best_next_reward - curr_q))
            self.dnn.zero_grad()

            other_action_q = estimate[1 - a].item()

            if a == 0:
                # print([corrected_reward, other_action_q])
                estimate.backward(torch.tensor( [corrected_reward, other_action_q] ))
            else:
                # print([other_action_q, corrected_reward])
                estimate.backward(torch.tensor( [other_action_q, corrected_reward] ))


    
    # Gets a random action in the action space; has to be re-defined for each action.
    def get_random_action(self, state):
        return np.random.randint(0, self.env.action_space.n)
    
    # Closes the given environment
    def close(self):
        self.env.close()

    def estimate_q_values(self, state):
        state_tensor = torch.tensor(state)
        state_with_fake_minibatch_dim = state_tensor.unsqueeze(0)
        return self.dnn(state_with_fake_minibatch_dim)[0]