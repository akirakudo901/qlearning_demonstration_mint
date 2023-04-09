"""
Code demonstrating the basic training loop for reinforcement learning through q-learning
and epsilon-greedy algorithm.

**WHAT IS REINFORCEMENT LEARNING?**

*The four features of reinforcement learning (RL)*

RL comes down to four features:
-) environment - the world that we want to learn from. 
                 A single situation in an environment is a "state"
-) agent = the entity that learns through observing the environment and taking actions
-) observation - what the agent perceives about the world
-) action - what the agent does
______________________________________________

*Agent-environment interaction (step)*

Nice 16 sec illustration of the flow on YouTube!
https://www.youtube.com/watch?v=-uXVu0l8guo&t=12s

We represent a single "step" of agent-environment interaction as follows:
1) the agent observes the environment
2) it chooses action based on observation 
3) the action modifies the environment (& environment might also change naturally)

In the next step, the agent observes the new environment state, takes action, 
which modifies the environment, and so on.
A single step can be thought of as a single time step in the view of the agent.

______________________________________________

*Reward function and policy learning*

The agent learns from interacting with the world:
-) a "reward function" inherent in the environment and hidden from the agent, 
   evaluates each state & action's rewarding-ness to the agent in numerical values
-) the agent learns through associating actions taken in certain states with resultant rewards
   Different algorithms (Q-learning, deep neural networks, etc.) are used for the agent's learning
-) the agent starts off knowing nothing about the world, and explores the world taking random actions
-) its goal is to learn a guiding rule for action maximizing reward at each state (called "policy")

Letting the agent learn a highly performant policy is our goal.
Usual RL training involves the agent interacting a lot with the environment.
______________________________________________

*What changes from problem to problem for RL*

We change:
-) THE ENVIRONMENT : the world we learn from depends on the problem
+ (also includes observations and actions)
  <- probably the software & signal processing team's field! 

-) THE REWARD FUNCTION : what quantifies the goal of the agent
  <- probably the whole team's field! (maybe more on the deep learning team?)

-) THE AGENT'S LEARNING ALGORITHM : the algorithm by which the agent learns
  <- probably the deep learning team's field!

The general flow of problem solving stays the same for most RL problems!

_____________________________________
SPECIFICS:
There are lines of code noted with SPECIFICS, which relates to the nature of Q-learning.
If interested further, read the online article below!
https://blog.floydhub.com/an-introduction-to-q-learning-reinforcement-learning/ 

Recently:
- Tried to add a penalty when the simulation terminates with pole falling; didn't quite work

"""


# First install dependencies as required through:
# pip install gymnasium
from datetime import datetime
import math, random
 
import matplotlib.pyplot as plt
import tqdm

# import agents.cartpole_qtable_agent as cartpole_qtable_agent
import agents.cartpole_dnn_agent as cartpole_dnn_agent

# the environment object to be used for training and evaluation
# env_object = cartpole_qtable_agent.CartpoleQtableAgent

env_object = cartpole_dnn_agent.CartpoleDNNAgent

# parameters related to training
EPISODES = 15000 #with l_r=0.005, d_r=0.95, init_eps=0.5 and decay_val=0.9995, eps = 0.05 by 15000
SHOW_PROGRESS_EVERY_N_EPISODES = EPISODES / 5
EXPLORATION_EPISODES = EPISODES / 6
UPDATER_SECOND_DNNS_EVERY_N = 1000
# related to q-learning specifically?
LEARNING_RATE = 0.005
DISCOUNT_RATE = 0.95
INITIAL_EPSILON = 0.5 # probability for exploration
EPSILON_DECAY_VALUE = 0.9995 #0.99995 seemed too low for speedy learning

RENDER_TRAINING = False
SAVE_TRAINING_RESULT = True
TRAIN_AGENT = True
EVALUATE_DURING_TRAINING = True

def save_with_creation_time_name(env):
    creation_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
    pz = "./dnns/Cartpole_DNN_" + creation_time + "_zero_over460"
    po = "./dnns/Cartpole_DNN_" + creation_time + "_one_over460"
    env.save(path_zero=pz, path_one=po)


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
    last_ten_episode_reward = []
    ten_episode_average_reward_over_time = []
    epsilon_over_time = []
    
    second_dnn_update_count = 0

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

            # update the second dnns appropriately
            second_dnn_update_count += 1
            if second_dnn_update_count == UPDATER_SECOND_DNNS_EVERY_N:
                second_dnn_update_count = 0
                env.update_second_dnns()

            if terminated or truncated:
                d = True

            # -) update info to adjust at the end of the step
            s = n_s
        
        # Once episode is over:
        # Update learning algorithm
        # SPECIFICS: HOW IS THE QTABLE UPDATED?
        
        env.update()

        # Then adjust values accordingly
        last_ten_episode_reward.append(episode_reward)
        if len(last_ten_episode_reward) == 10:
            acc = 0
            for last_rewards in last_ten_episode_reward:
                acc += last_rewards
            avg_r = acc / 10
            ten_episode_average_reward_over_time.append(avg_r)
            last_ten_episode_reward.pop(0)

        epsilon_over_time.append(epsilon)

        if epsilon > 0.05: #epsilon modification
            if episode_reward > prior_reward and episode > EXPLORATION_EPISODES:
                epsilon = INITIAL_EPSILON * math.pow(EPSILON_DECAY_VALUE, episode - EXPLORATION_EPISODES)
        
        prior_reward = episode_reward
        
        if (episode % SHOW_PROGRESS_EVERY_N_EPISODES == 0):
            # TOREMOVE
            # print("\naction_zero_parameters: \n", list(env.dnn_action_zero.parameters()))
            # print("\naction_one_parameters: \n", list(env.dnn_action_one.parameters()))
            if EVALUATE_DURING_TRAINING:
                evaluate(env)
            
            if SAVE_TRAINING_RESULT:
                save_with_creation_time_name(env)

        #Also let's me show when the result was pretty good (above 150)
        if episode_reward >= 460 and SAVE_TRAINING_RESULT:
            save_with_creation_time_name(env)

    # End of training things
    env.close() # close the training env
    if SAVE_TRAINING_RESULT:
        env.save()
    
    _, ax = plt.subplots()
    ax.plot(range(len(ten_episode_average_reward_over_time)), ten_episode_average_reward_over_time, linewidth=2.0)
    plt.title("reward over time")
    plt.show()

    _, ax2 = plt.subplots()
    ax2.plot(range(len(epsilon_over_time)), epsilon_over_time, linewidth=2.0)
    plt.title("epsilon over time")
    plt.show()

    return env


#+++++++++++++++++++++++++++++++++++++++
#Evaluating

# Runs a full cycle of the environment given the agent or a path. 
# If the agent is given, the path is not considered.
def evaluate(agent=None, path_zero=None, path_one=None):
    if agent is None:
        agent = env_object(l_r=LEARNING_RATE, d_r=DISCOUNT_RATE)
        # agent.load(path)
        agent.load(path_zero=path_zero, path_one=path_one)
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
        s, _, terminated, truncated, _ = env_eval.step(a)

    env_eval.close()

if __name__ == "__main__":
    #First train, details in train function
    if TRAIN_AGENT and True:
        env = train()
    # Evaluation? See it in action, probably + store the result in some way & allow reading.
    evaluate(env)
    # evaluate(path_zero="dnns/2023_04_09_15_38_path_zero_over450", path_one="dnns/2023_04_09_15_38_path_one_over450")