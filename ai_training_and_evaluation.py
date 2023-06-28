"""
Defines the functions to be used to train and evaluate a given agent in a given environment
using the epsilon-greedy exploration approach.
Tried to abstract away a lot of parts (don't know if it's gonna be useful).
"""

import math, random
 
import matplotlib.pyplot as plt
import tqdm

#+++++++++++++++++++++++++++++++++++++++
#Training

def train_epsilon_greedy(
        env_agent_merged_object,
        learning_rate : float,
        discount_rate : float,
        initial_epsilon : float,
        epsilon_decay_value : float,
        episodes : int,
        exploration_episodes : int = None,
        show_progress_every_n_episodes : int = None,
        render_training : bool = False,
        save_training_result : bool = True,
        ):

    # miscellaneous to be set
    def parameter_should_be_positive_int(param, param_name, def_val):
        if param != None:
            if (type(param) != type(0)) or param <= 0: 
                raise Exception(f"{param_name} should be a positive integer!")
            return param
        else:
            return def_val
    
    show_progress_every_n_episodes = parameter_should_be_positive_int(
        show_progress_every_n_episodes, "show_progress_every_n_episodes", episodes // 5
        )
    exploration_episodes = parameter_should_be_positive_int(
        exploration_episodes, "exploration_episodes", episodes // 6
    ) # the episode until which we explore

    # setup agent, which sets up both: 
    # - the environment (CartPole-v1 here) and 
    # - the learning algorithm (Q learning here)
    r_m = "human" if render_training else None
    agent = env_agent_merged_object(l_r=learning_rate, d_r= discount_rate, r_m=r_m)

    prior_reward = 0
    epsilon = initial_epsilon
    reward_over_time = []
    epsilon_over_time = []

    # Single episode loop:
    for episode in tqdm.tqdm(range(episodes)):
        # training:
        # 0) reset the environment (env), and setup appropriate values: 
        # - state of env
        # - rewards over episode (0)
        # - done or not? which is False
        # - array storing pairs of state, action and next state 
        # - epsilon for epsilon-greedy, which decreases by initial_epsilon / episodes every episode

        s, _ = agent.reset()
        d = False
        episode_reward = 0

        # Training loop:
        while not d:
            
            threshold = random.random()
            a = agent.get_optimal_action(s) if (threshold > epsilon) else agent.get_random_action(s)

            # -) update the environment accordingly given the action, taking: 
            # new state, new reward, done?, info
            n_s, r, terminated, truncated, _ = agent.step(a)
            episode_reward += r

            if terminated or truncated:
                d = True

            # -) update info to adjust at the end of the step
            s = n_s
        
        # Once episode is over:
        # Update learning algorithm
        
        agent.update()

        # Then adjust values accordingly
        reward_over_time.append(episode_reward)
        epsilon_over_time.append(epsilon)

        if epsilon > 0.05: #epsilon modification
            if episode_reward > prior_reward and episode > exploration_episodes:
                epsilon = initial_epsilon * math.pow(epsilon_decay_value, episode - exploration_episodes)
                prior_reward = episode_reward
        
        if episode % show_progress_every_n_episodes == 0:
            evaluate(env_agent_merged_object, agent)

    # End of training things
    agent.close() # close the training env
    if save_training_result:
        agent.save()
    
    _, ax = plt.subplots()
    ax.plot(reward_over_time, linewidth=2.0)
    plt.show()

    _, ax2 = plt.subplots()
    ax2.plot(epsilon_over_time, linewidth=2.0)
    plt.title("epsilon over time")
    plt.show()

    return agent

#+++++++++++++++++++++++++++++++++++++++
#Evaluating

# Runs a full cycle of the environment given the agent or a path. 
# If a path is given, the agent object will load the path.
def evaluate(
        env_agent_merged_object,
        trained_agent,
        path : str = None
        ):
    
    # load the path if it is given and not None 
    if path != None:
        try: 
            trained_agent.load(path)
            print("\n path has been loaded! \n")
        except:
            raise Exception("Something went wrong when loading the path into the agent...")

    # Evaluation loop:
    env_eval = env_agent_merged_object(r_m="human")
    s, _ = env_eval.reset() #reset the environment
    terminated = truncated = False

    while not (terminated or truncated):

        # get optimal action by agent
        a = trained_agent.get_optimal_action(s)

        # update env accordingly
        s, _, terminated, truncated, _ = env_eval.step(a)

    env_eval.close()