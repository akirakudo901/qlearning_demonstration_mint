"""
Base abstract class for whatever policy learning algorithm we use.
"""

from abc import ABC, abstractmethod

class PolicyLearningAlgorithm(ABC):
    
    # Updates the algorithm accordingly
    @abstractmethod
    def update(self):
        self.algorithm.update(self.episode_state_action_pairs, self.episode_reward)
    
    # Chooses the optimal action given the state
    @abstractmethod
    def get_optimal_action(self, state):
        pass

    # Saves the policy learned so far in some form
    @abstractmethod
    def save(self):
        return
    
    # Loads the policy
    @abstractmethod
    def load(self):
        return 