
# First install dependencies as required through:
# pip install gymnasium
import math, random
 
import matplotlib.pyplot as plt
import tqdm

# import agents.cartpole_qtable_agent as cartpole_qtable_agent
import agents.cartpole_dnn_agent as cartpole_dnn_agent

# the environment object to be used for training and evaluation
# env_object = cartpole_qtable_agent.CartpoleQtableAgent

env_object = cartpole_dnn_agent.CartpoleDNNAgent

# parameters related to training
EPISODES = 10000
SHOW_PROGRESS_EVERY_N_EPISODES = EPISODES / 5
EXPLORATION_EPISODES = EPISODES / 6
# related to q-learning specifically?
LEARNING_RATE = 0.1
DISCOUNT_RATE = 0.95
INITIAL_EPSILON = 1 # probability for exploration
EPSILON_DECAY_VALUE = 0.99995

RENDER_TRAINING = False
SAVE_TRAINING_RESULT = True
TRAIN_AGENT = True


#+++++++++++++++++++++++++++++++++++++++
#Training

def train():
    # setup agent, which sets up both: 
    # - the environment (CartPole-v1 here) and 
    # - the learning algorithm (Q learning here)
    r_m = "human" if RENDER_TRAINING else None
    env = env_object(l_r=LEARNING_RATE, d_r= DISCOUNT_RATE,r_m=r_m)

    prior_reward = 0
    epsilon = INITIAL_EPSILON
    reward_over_time = []
    epsilon_over_time = []

    # Single episode loop:
    for episode in tqdm.tqdm(range(EPISODES)):
        # training:
        # 0) reset the environment (env), and setup appropriate values: 
        # - state of env
        # - rewards over episode (0)
        # - done or not? which is False
        # - array storing pairs of state, action and next state 
        # - epsilon for epsilon-greedy, which decreases by INITIAL_EPSILON / EPISODES every episode

        s, _ = env.reset()
        d = False
        episode_reward = 0

        # Training loop:
        while not d:

            # SPECIFICS: WHAT IS EPSILON-GREEDY?
            # -) choose an action: if epsilon is greater than random value, choose optimal solution so far
            #    otherwise choose an action at random in the action space
            
            threshold = random.random()
            a = env.get_optimal_action(s) if (threshold > epsilon) else env.get_random_action(s)

            # SPECIFICS: HOW IS THE OPTIMAL SOLUTION DETERMINED IN Q LEARNING? 

            # -) update the environment accordingly given the action, taking: 
            # new state, new reward, done?, info
            n_s, r, terminated, truncated, _ = env.step_and_update(a)
            episode_reward += r

            if terminated or truncated:
                d = True

            # -) update info to adjust at the end of the step
            s = n_s
        
        # Once episode is over:
        # Update learning algorithm
        # SPECIFICS: HOW IS THE QTABLE UPDATED?
        
        env.update()

        # Then adjust values accordingly
        reward_over_time.append(episode_reward)
        epsilon_over_time.append(epsilon)

        if epsilon > 0.05: #epsilon modification
            if episode_reward > prior_reward and episode > EXPLORATION_EPISODES:
                epsilon = math.pow(EPSILON_DECAY_VALUE, episode - EXPLORATION_EPISODES)
                prior_reward = episode_reward
        
        if episode % SHOW_PROGRESS_EVERY_N_EPISODES == 0:
            # TOREMOVE
            # print("\naction_zero_parameters: \n", list(env.dnn_action_zero.parameters()))
            # print("\naction_one_parameters: \n", list(env.dnn_action_one.parameters()))
            evaluate(env)

    # End of training things
    env.close() # close the training env
    if SAVE_TRAINING_RESULT:
        env.save()
    
    _, ax = plt.subplots()
    ax.plot(reward_over_time, linewidth=2.0)
    plt.show()

    _, ax2 = plt.subplots()
    ax2.plot(epsilon_over_time, linewidth=2.0)
    plt.title("epsilon over time")
    plt.show()

    return env


#+++++++++++++++++++++++++++++++++++++++
#Evaluating

# Runs a full cycle of the environment given the agent or a path. 
# If the agent is given, the path is not considered.
def evaluate(agent=None, path=None):
    if agent is None:
        agent = env_object(l_r=LEARNING_RATE, d_r=DISCOUNT_RATE)
        agent.load(path)
    else:
        pass
        # print("\n agent is not None; path will not be considered. \n")

    # Evaluation loop:
    env_eval = env_object(r_m="human")
    s, _ = env_eval.reset() #reset the environment
    terminated = truncated = False

    while not (terminated or truncated):

        # get optimal action by agent
        a = agent.get_optimal_action(s)

        # update env accordingly
        s, _, terminated, truncated, _ = env_eval.step_and_update(a)

    env_eval.close()

if __name__ == "__main__":
    #First train, details in train function
    if TRAIN_AGENT:
        env = train()
    # Evaluation? See it in action, probably + store the result in some way & allow reading.
    evaluate(env)
    # evaluate(path="qtable\Cartpole_Q_table_2023_3_23_0_38.npy")