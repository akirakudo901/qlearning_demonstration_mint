"""
Base abstract class for whatever agent we use which combines environment with learning algorithm.
"""

from abc import ABC, abstractmethod

class Agent(ABC):

    # Resets the environment
    @abstractmethod
    def reset(self):
        return self.env.reset()
    
    # Takes a step in the environment with the given action, and returns:
    #  - the resulting next state
    #  - the reward as a result of the action
    #  - if the episode was corretly terminated (boolean; terminated)
    #  - if the episode was incorrectly terminated (boolean; truncated)
    #  - additional info
    @abstractmethod
    def step(self, action):
        # return info
        return self.env.step(action)
    
    # Updates the algorithm accordingly
    @abstractmethod
    def update(self):
        # self.algorithm.update(self.episode_state_action_pairs, self.episode_reward)
        pass
    
    # Chooses the optimal action given the state
    @abstractmethod
    def get_optimal_action(self, state):
        return self.algorithm.get_optimal_action(state)

    # Gets a random action in the action space; has to be re-defined for each action.
    @abstractmethod 
    def get_random_action(self, state):
        return self.env.get_random_action(state)