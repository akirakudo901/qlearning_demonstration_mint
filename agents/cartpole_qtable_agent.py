"""
A class representing the agent which learns the cart pole task.
Brings the environment and learning algorithm together in one class.

Since the problem is continuous but q learning deals with discrete environments,
takes up the task of converting continuous into discrete.
"""

import numpy as np
import gymnasium
import policy_learning_algorithms.qtable as qtable

class CartpoleQtableAgent:
    NP_ARRAY_WIN_SIZE = np.array([0.25, 0.25, 0.01, 0.1])
    QTABLE_SHAPE = [30, 30, 50, 50, 2]
    
    # Initializes a new cartpole agent which works on the cart pole problem, and with:
    #  - learning rate = l_r
    #  - discount rate = d_r
    #  - render mode to the environment = r_m
    def __init__(self, l_r=0.1, d_r=0.95, r_m=None):
        self.env = gymnasium.make("CartPole-v1", render_mode=r_m)
        self.algorithm = qtable.Qtable(CartpoleQtableAgent.QTABLE_SHAPE, 
                                       "Cartpole", 
                                       l_r=l_r, 
                                       d_r=d_r)

        self.state = None
        self.episode_state_action_pairs = []
        self.episode_reward = 0

    # Resets the environment - doesn't add too much to original
    def reset(self):
        self.episode_state_action_pairs = []
        self.state, info = self.env.reset()
        self.episode_reward = 0
        return self.state, info
    
    # Takes a step in the environment with the given action, and returns:
    #  - the resulting next state
    #  - the reward as a result of the action
    #  - if the episode was corretly terminated (boolean; terminated)
    #  - if the episode was incorrectly terminated (boolean; truncated)
    #  - additional info
    # Not much new compared to the original env's "step".
    def step(self, action):
        s_before_action_disc = CartpoleQtableAgent.get_discrete_state(self.state)
        n_s, r, terminated, truncated, info = self.env.step(action)
        self.state = n_s
        # Update info required for retraining of Q table at the end
        n_s_disc = CartpoleQtableAgent.get_discrete_state(n_s)
        self.episode_state_action_pairs.append([s_before_action_disc, action, n_s_disc])
        self.episode_reward += r
        # return info
        return n_s, r, terminated, truncated, info
    
    # Closes the given environment
    def close(self):
        self.env.close()
    
    # Gets the optimal action in this state
    def get_optimal_action(self, state):
        s_disc = CartpoleQtableAgent.get_discrete_state(state)
        return self.algorithm.get_optimal_action(s_disc)
    
    # Gets a random action in the action space
    # Has to be re-defined for each action; for CartPole, doesn't need state as input
    def get_random_action(self, state):
        return np.random.randint(0, self.env.action_space.n)
    
    # Updates the Q table accordingly to the current state action pairs and reward stored
    def update(self):
        self.algorithm.update(self.episode_state_action_pairs, self.episode_reward)

    # Saves the state of learning to a given file
    def save(self, name):
        self.algorithm.save(name)

    # Loads the state of learning from a given path
    def load(self, path):
        self.algorithm.load(path)

    # Maps each continuous state vector obtained from env to a discrete state stored in Q table
    # Divides the state array by the window sizes specified by ARRAY_WIN_SIZE, then adds the np.array
    # to turn them to values above 0 (and hopefully within range of the Q table)
    @staticmethod
    def get_discrete_state(state):
        # scale all values to be integers and above 0
        discrete_state = state/CartpoleQtableAgent.NP_ARRAY_WIN_SIZE + np.array([15,10,1,10])
        return tuple(discrete_state.astype(int))