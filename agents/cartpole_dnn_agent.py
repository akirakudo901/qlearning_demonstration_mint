"""
Cart pole agent that learns through deep neural networks, combining a DNN with 
an environment.

"""

from datetime import datetime
import random

import gymnasium
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

DNN_SAVE_FOLDER = "./trained_agents/dnns"

UPDATE_TARGET_DNN_EVERY_N = 4096
MAX_BUFFER_SIZE = 4096

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
            """
            Initializes a new Deep neural network (DNN).
            """
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
        state_tensor = torch.tensor(state).to(CartpoleDNNAgent.DEVICE).unsqueeze(0)
        prediction = self.dnn_policy(state_tensor)
        return torch.argmax(prediction).item()
    
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
    
    def step(self, action):
        """
        Takes a step in the environment with the given action while sampling and keeping
        experience in buffer. Returns:
         - the resulting next state
         - the reward as a result of the action
         - if the episode was corretly terminated (boolean; terminated)
         - if the episode was incorrectly terminated (boolean; truncated)
         - additional info

        :param action: The action taken by the agent.
        """
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
        BATCH_SIZE = 32
        EPOCHS = 12
        ACTION_NUM = 2

        if len(self.buffer) > MAX_BUFFER_SIZE:
            self.buffer.reverse() #ensures that newest experience at end is kept
            self.buffer = self.buffer[:MAX_BUFFER_SIZE]
        elif len(self.buffer) < BATCH_SIZE: #if buffer too small, pass
            return
        
        loss_fn = nn.MSELoss()

        for _ in range(EPOCHS):
            random.shuffle(self.buffer)
            batches = [
                self.buffer[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
                for i in range(
                min(8, len(self.buffer) // BATCH_SIZE)
                )
            ]

            for batch in batches:
                # stack each entry into torch tensors to do further computation 
                current_states = torch.from_numpy(np.stack([exp[0] for exp in batch], dtype=np.float32))
                actions = torch.from_numpy(np.stack([exp[1] for exp in batch], dtype=np.int64))
                rewards = torch.from_numpy(np.stack([exp[2] for exp in batch], dtype=np.float32))
                next_states = torch.from_numpy(np.stack([exp[3] for exp in batch], dtype=np.float32))
                # get the corresponding updated q val
                updated_q = (rewards 
                             + self.discount
                             * torch.max(self.dnn_target(next_states).detach(), dim=1).values)
                # from action, make a mask 
                mask = torch.zeros(len(batch), ACTION_NUM)
                mask.scatter_(1, actions.unsqueeze(1), 1)
                # apply mask to obtain the relevant predictions for current states
                compared_q = torch.sum(self.dnn_policy(current_states) * mask, dim=1)
                # calculate loss)
                loss = loss_fn(compared_q, updated_q)
                # propagate the result
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

        # update the target dnn appropriately after one update
        self._update_target()
            
    # Updates the target dnn by setting its values equal to that of the policy dnn
    def _update_target(self):
        self.dnn_target.load_state_dict(self.dnn_policy.state_dict())
    
    # Gets a random action in the action space; has to be re-defined for each action.
    def get_random_action(self, state):
        return np.random.randint(0, self.env.action_space.n)
    
    # Closes the given environment
    def close(self):
        self.env.close()