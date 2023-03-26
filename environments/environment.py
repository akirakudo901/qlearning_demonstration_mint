"""
The abstract base class for any environments we work with.
"""

from abc import ABC, abstractmethod

class Environment(ABC):

    # Resets the environment to the initial state
    @abstractmethod
    def reset(self):
        # Returns the new state and additional info
        pass
    
    # Execute a single step in the environment through taking the given action
    # Determines how taking what action in what state leads to what new action and reward
    @abstractmethod
    def step(self, action):
        # Returns the new state, the reward, if the execution was terminated,
        #  if the execution was truncated, and additional info
        pass
    
    # Chooses a random action from the action space
    @abstractmethod
    def random_action(self, state):
        pass

    # Renders the environment in an appropriate way
    @abstractmethod
    def render(self):
        pass