"""
Handles code tracking reward over training and else, cleaning the code up.
"""

import matplotlib.pyplot as plt

class Progress:

    def __init__(self, 
                 last_N_reward : int=10,
                 initial_epsilon : float = 1):
        """
        last_N : int; number of episode reward we average over 
        initial_epsilon : float; initial value for epsilon greedy
        """
        self.last_N = last_N_reward
        self.initial_setUp(init_eps=initial_epsilon)

    def initial_setUp(self, init_eps : float=1):
        """
        To be run when first created or reset.
        Resets the internal track of rewards to empty.
        """
        self.last_N_episode_reward = []
        self.N_episode_average_reward_over_time = []
        self.epsilon_over_time = [init_eps]
    
    def at_beginning_of_episode(self):
        """
        To be run at the beggining of the episode.
        Initializes the episode reward.
        """
        self.episode_reward = 0

    def after_each_action(self, new_reward):
        """
        To be run after each action by the agent.
        Updates the episode reward.
        """
        self.episode_reward += new_reward

    def at_end_of_episode(self, new_epsilon):
        """
        To be run at the end of the episode.
        Records the reward including that of the last N episodes
        to track the progress over time of training.
        """
        self.last_N_episode_reward.append(self.episode_reward)
        
        if len(self.last_N_episode_reward) >= self.last_N:
            # very first time we calculate the sum
            if len(self.N_episode_average_reward_over_time) == 0:
                acc = 0; [acc := acc + r for r in self.last_N_episode_reward]
                avg_r = acc / self.last_N
                self.N_episode_average_reward_over_time.append(avg_r)
            # calculate the sum from previous value
            else:
                prev_sum = self.N_episode_average_reward_over_time[-1] * self.last_N
                new_sum = prev_sum - self.last_N_episode_reward.pop(0) + self.last_N_episode_reward[-1]
                avg_r = new_sum / self.last_N
                self.N_episode_average_reward_over_time.append(avg_r)
        
        # Also update the epsilon value
        self.epsilon_over_time.append(new_epsilon)
    
    def plot_result(self):
        _, ax = plt.subplots()
        ax.plot(
            range(len(self.N_episode_average_reward_over_time)), 
            self.N_episode_average_reward_over_time, 
            linewidth=2.0
            )
        plt.title("Average reward over " + str(self.last_N) + " episodes")
        plt.show()

        _, ax2 = plt.subplots()
        ax2.plot(
            range(len(self.epsilon_over_time)), self.epsilon_over_time, linewidth=2.0)
        plt.title("Epsilon over time")
        plt.show()
        
    def get_last_episode_reward(self):
        """
        Returns the last recorded reward, or 0 if past rewards are empty.
        """
        return 0 if (len(self.last_N_episode_reward) == 0) else self.last_N_episode_reward[-1]
