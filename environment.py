"""
The environment we work with.
"""

class Environment:

    # Resets the environment to the initial state
    def reset(self):
        # Returns the new state and additional info
        return 
    
    # Execute a single step in the environment through taking the given action
    # Determines how taking what action in what state leads to what new action and reward
    def step(self, action):
        # Returns the new state, the reward, if the execution was terminated,
        #  if the execution was truncated, and additional info
        return
    
    # Chooses a random action from the action space
    def random_action(self, state):
        return None