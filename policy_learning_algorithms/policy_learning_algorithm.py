"""
Whatever policy learning algorithm we use.
"""

from abc import ABC, abstractmethod

class PolicyLearningAlgorithm(ABC):
    
    # Initializes a new agent combining environment with learning algorithm
    @abstractmethod
    def __init__(self):
        pass

    # Resets the environment
    @abstractmethod
    def reset(self):
        pass
    
    # Takes a step in the environment with the given action, and returns:
    #  - the resulting next state
    #  - the reward as a result of the action
    #  - if the episode was corretly terminated (boolean; terminated)
    #  - if the episode was incorrectly terminated (boolean; truncated)
    #  - additional info
    # Not much new compared to the original env's "step".
    @abstractmethod
    def step(self, action):
        # return info
        pass
    
    # Updates the algorithm accordingly
    @abstractmethod
    def update(self):
        self.algorithm.update(self.episode_state_action_pairs, self.episode_reward)
    
    # Chooses the optimal action given the state
    @abstractmethod
    def get_optimal_action(self, state):
        pass

    # Gets a random action in the action space; has to be re-defined for each action.
    @abstractmethod 
    def get_random_action(self, state):
        pass
    
    # Saves the policy learned so far in some form
    @abstractmethod
    def save(self):
        return
    
    # Loads the policy
    @abstractmethod
    def load(self):
        return 