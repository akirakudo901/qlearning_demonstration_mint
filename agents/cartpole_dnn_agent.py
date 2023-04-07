# """
# Cart pole agent that learns through deep neural networks, combining a DNN with 
# an environment.
# """

# import numpy as np
# import gymnasium

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# class CartpoleDNNAgent:

#     class DNN(nn.Module):

#         def __init__(self):
#             # Initializes a new DNN.
#             super(CartpoleDNNAgent.DNN, self).__init__()
            
#             # - an input layer of size four
#             self.fc1 = nn.Linear(4, 8)
#             # - one hidden layer with 8 neurons
#             # self.fc2 = nn.Linear(8, 8)
#             # - another hidden layer with 8 more neurons and 2 outputs
#             self.fc3 = nn.Linear(8, 1)

#         def forward(self, x):
#             x = F.relu(self.fc1(x))
#             # x = F.relu(self.fc2(x))
#             x = self.fc3(x)
#             return x
    
#     def __init__(self, l_r=0.1, d_r=0.95, r_m=None):
#         self.env = gymnasium.make("CartPole-v1", render_mode=r_m)
#         self.learning_rate = l_r
#         self.discount_rate = d_r
#         self.dnn_action_zero = CartpoleDNNAgent.DNN()
#         self.dnn_action_one = CartpoleDNNAgent.DNN()

#         # self.dnn_action_zero_q = CartpoleDNNAgent.DNN()
#         # self.dnn_action_one_q = CartpoleDNNAgent.DNN()

#         self.optim_zero = optim.SGD(self.dnn_action_zero.parameters(), lr=self.learning_rate)
#         self.optim_one = optim.SGD(self.dnn_action_one.parameters(), lr=self.learning_rate)

#         # self.optim_zero_q = optim.SGD(self.dnn_action_zero_q.parameters(), lr=self.learning_rate)
#         # self.optim_one_q = optim.SGD(self.dnn_action_one_q.parameters(), lr=self.learning_rate)


#         self.state = None
        
#     # "state" is a tuple of four values
#     def get_optimal_action(self, state):
#         if (self.estimate_q_values_action_zero(state).item() <= self.estimate_q_values_action_one(state).item()):
#             return 1
#         else:
#             return 0
    
#     # TODO later
#     def save(self):
#         pass

#     # TODO later
#     def load(self):
#         pass

#     # Resets the environment
#     def reset(self):
#         self.state, info = self.env.reset()
#         return self.state, info
    
#     # Takes a step in the environment with the given action, and returns:
#     #  - the resulting next state
#     #  - the reward as a result of the action
#     #  - if the episode was corretly terminated (boolean; terminated)
#     #  - if the episode was incorrectly terminated (boolean; truncated)
#     #  - additional info
#     def step_and_update(self, action, tbremoved):
#         s_before_action = self.state
#         n_s, r, terminated, truncated, info = self.env.step(action)
#         self.state = n_s
#         # Update info required for retraining of Q table at the end
#         self.update_reward(s_before_action, action, r, n_s, tbremoved)
#         # return info
#         return n_s, r, terminated, truncated, info
    
#     # Takes a step in the environment with the given action, and returns:
#     #  - the resulting next state
#     #  - the reward as a result of the action
#     #  - if the episode was corretly terminated (boolean; terminated)
#     #  - if the episode was incorrectly terminated (boolean; truncated)
#     #  - additional info
#     def step(self, action):
#         return self.env.step(action)
        
#     # Updates the algorithm at the end of episode; in this case, does nothing
#     def update(self):
#         pass

#     # Updates the algorithm accordingly
#     def update_reward(self, s, a, r, n_s, tbremoved):
#         estimate_q_values = self.estimate_q_values_action_zero if (a == 0) else self.estimate_q_values_action_one
#         appropriate_optim = self.optim_zero if (a == 0) else self.optim_one
        
#         best_next_reward = torch.max(torch.tensor([
#             self.estimate_q_values_action_zero(n_s).item(), 
#             self.estimate_q_values_action_one(n_s).item()
#             ]))

#         appropriate_optim.zero_grad()
        
#         curr_q = estimate_q_values(s)
        
#         corrected_reward = r + self.discount_rate * best_next_reward
#         dim_correct = corrected_reward.unsqueeze(0).unsqueeze(0)

#         loss_fn = nn.MSELoss()
#         loss = loss_fn(curr_q, dim_correct)
#         loss.backward()

#         appropriate_optim.step()

#         if tbremoved: print("curr_q: ", curr_q, "corrected_reward: ", corrected_reward, "loss: ", loss)  
    
#     # Gets a random action in the action space; has to be re-defined for each action.
#     def get_random_action(self, state):
#         return np.random.randint(0, self.env.action_space.n)
    
#     # Closes the given environment
#     def close(self):
#         self.env.close()

#     def estimate_q_values_action_zero(self, state):
#         state_tensor = torch.tensor(state)
#         state_with_fake_minibatch_dim = state_tensor.unsqueeze(0)
#         return self.dnn_action_zero(state_with_fake_minibatch_dim) 

#     def estimate_q_values_action_one(self, state):
#         state_tensor = torch.tensor(state)
#         state_with_fake_minibatch_dim = state_tensor.unsqueeze(0)
#         return self.dnn_action_one(state_with_fake_minibatch_dim)

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
                # nn.ReLU(),
                # - one hidden layer with 8 neurons
                nn.Linear(8, 8),
                nn.Sigmoid(), 
                # nn.ReLU(),
                nn.Linear(8, 16),
                nn.Sigmoid(),
                # nn.ReLU(),
                # - another hidden layer with 8 more neurons and 2 outputs
                nn.Linear(16, 1),
            )

        def forward(self, x):
            x = self.fc_relu_stack(x)
            return x
    
    def __init__(self, l_r=0.1, d_r=0.95, r_m=None):
        self.env = gymnasium.make("CartPole-v1", render_mode=r_m)
        self.learning_rate = l_r
        self.discount_rate = d_r
        self.dnn_action_zero = CartpoleDNNAgent.DNN()
        self.dnn_action_one = CartpoleDNNAgent.DNN()

        self.dnn_action_zero.to(CartpoleDNNAgent.DEVICE)
        self.dnn_action_one.to(CartpoleDNNAgent.DEVICE)

        # self.dnn_action_zero_q = CartpoleDNNAgent.DNN()
        # self.dnn_action_one_q = CartpoleDNNAgent.DNN()

        self.optim_zero = optim.SGD(self.dnn_action_zero.parameters(), lr=self.learning_rate)
        self.optim_one = optim.SGD(self.dnn_action_one.parameters(), lr=self.learning_rate)

        # self.optim_zero_q = optim.SGD(self.dnn_action_zero_q.parameters(), lr=self.learning_rate)
        # self.optim_one_q = optim.SGD(self.dnn_action_one_q.parameters(), lr=self.learning_rate)


        self.state = None
        
    # "state" is a tuple of four values
    def get_optimal_action(self, state):
        if (self.estimate_q_values_action_zero(state).item() <= self.estimate_q_values_action_one(state).item()):
            return 1
        else:
            return 0
    
    # TODO later
    def save(self, path_zero="Cartpole_DNN_zero.pth", path_one="Cartpole_DNN_one.pth"):
        torch.save(self.dnn_action_zero.state_dict(), path_zero)
        torch.save(self.dnn_action_one.state_dict(), path_one)

    # TODO later
    def load(self, path_zero="Cartpole_DNN_zero.pth", path_one="Cartpole_DNN_one.pth"):
        self.dnn_action_zero.load_state_dict(torch.load(path_zero))
        self.dnn_action_one.load_state_dict(torch.load(path_one))
        self.dnn_action_zero.eval()
        self.dnn_action_one.eval()
        # move to DEVICE
        self.dnn_action_zero.to(CartpoleDNNAgent.DEVICE)
        self.dnn_action_one.to(CartpoleDNNAgent.DEVICE)

    # Resets the environment
    def reset(self):
        self.state, info = self.env.reset()
        return self.state, info
    
    # Takes a step in the environment with the given action, and returns:
    #  - the resulting next state
    #  - the reward as a result of the action
    #  - if the episode was corretly terminated (boolean; terminated)
    #  - if the episode was incorrectly terminated (boolean; truncated)
    #  - additional info
    def step_and_update(self, action, tbremoved):
        s_before_action = self.state
        n_s, r, terminated, truncated, info = self.env.step(action)
        self.state = n_s
        # Update info required for retraining of Q table at the end
        self.update_reward(s_before_action, action, r, n_s, tbremoved)
        # return info
        return n_s, r, terminated, truncated, info
    
    # Takes a step in the environment with the given action, and returns:
    #  - the resulting next state
    #  - the reward as a result of the action
    #  - if the episode was corretly terminated (boolean; terminated)
    #  - if the episode was incorrectly terminated (boolean; truncated)
    #  - additional info
    def step(self, action):
        return self.env.step(action)
        
    # Updates the algorithm at the end of episode; in this case, does nothing
    def update(self):
        pass

    # Updates the algorithm accordingly
    def update_reward(self, s, a, r, n_s, tbremoved):
        estimate_q_values = self.estimate_q_values_action_zero if (a == 0) else self.estimate_q_values_action_one
        appropriate_optim = self.optim_zero if (a == 0) else self.optim_one
        
        best_next_reward = torch.max(torch.tensor([
            self.estimate_q_values_action_zero(n_s).item(), 
            self.estimate_q_values_action_one(n_s).item()
            ]))

        appropriate_optim.zero_grad()
        
        curr_q = estimate_q_values(s)
        
        corrected_reward = torch.tensor(r).to(CartpoleDNNAgent.DEVICE)# + self.discount_rate * best_next_reward
        dim_correct = corrected_reward.unsqueeze(0).unsqueeze(0)

        loss_fn = nn.MSELoss()
        loss = loss_fn(curr_q, dim_correct)
        loss.backward()

        appropriate_optim.step()

        # if tbremoved: print("curr_q: ", curr_q, "corrected_reward: ", corrected_reward, "loss: ", loss)  
    
    # Gets a random action in the action space; has to be re-defined for each action.
    def get_random_action(self, state):
        return np.random.randint(0, self.env.action_space.n)
    
    # Closes the given environment
    def close(self):
        self.env.close()

    def estimate_q_values_action_zero(self, state):
        state_tensor = torch.tensor(state).to(CartpoleDNNAgent.DEVICE)
        state_with_fake_minibatch_dim = state_tensor.unsqueeze(0)
        return self.dnn_action_zero(state_with_fake_minibatch_dim) 

    def estimate_q_values_action_one(self, state):
        state_tensor = torch.tensor(state).to(CartpoleDNNAgent.DEVICE)
        state_with_fake_minibatch_dim = state_tensor.unsqueeze(0)
        return self.dnn_action_one(state_with_fake_minibatch_dim)