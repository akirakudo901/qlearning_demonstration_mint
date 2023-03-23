"""
A class representing a qtable.
Functionalities include:
- initializing a qtable of given shape
- get_optimal_action
"""

import numpy as np
import time

QTABLE_FOLDER_PATH = "./qtable/"
DEFAULT_QTABLE_PATH = "./qtable/default.npy/"

class Qtable:

    # Initializes a Q table with: 
    # - a table of given "shape" filled with uniform random values
    # - the given learning rate "l_r"
    # - the given discount rate "d_r"
    def __init__(self, shape, taskName, l_r, d_r):
        self.shape = shape
        self.Q = np.random.uniform(low=0, high=1, size=shape)
        self.task_name = taskName
        self.learning_rate = l_r
        self.discount_rate = d_r
    
    # Returns the optimal action with highest current expected reward given the state.
    def get_optimal_action(self, state):
        return np.argmax(self.Q[state])
    
    # Obtains the highest reward possible given the state
    def get_best_reward(self, state):
        return np.max(self.Q[state])
    
    # Obtains the current rewward of a state-action pair.
    def get_reward(self, state, action):
        return self.Q[state, action]
    
    # Updates the given state action pair's reward.
    def update_state_action_reward(self, state, action, reward):
        self.Q[state, action] = reward
    
    # Updates the table given a series of state-action-next-state pair in time order, and 
    # the total reward for those series of action.
    def update(self, pairs, rEpisode):
        pairs.reverse()
        for pair in pairs:
            s, a, n_s = pair[0], pair[1], pair[2]
            updated_r = self.get_reward(s, a) + self.learning_rate * (rEpisode + self.discount_rate * (self.get_best_reward(n_s) - self.get_reward(s, a)))
            self.update_state_action_reward(s, a, updated_r)

    # saves the table into a .npy file
    def save(self, name=None):
        time_of_creation = time.localtime(time.time())[0:5]
        acc = str(time_of_creation[0])
        [acc := acc + "_" + str(x) for x in time_of_creation[1:5]]
        name_original = self.task_name + "_Q_table_" + acc
        
        if name is None:
            name = name_original

        name = QTABLE_FOLDER_PATH + "/" + name
        np.save(name, self.Q)
    
    # loads the .npy file table into a numpy array
    def load(self, path=DEFAULT_QTABLE_PATH):
        self.Q = np.load(path) 
