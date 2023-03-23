"""
A code demonstrating the basic training loop for reinforcement learning (RL) through q-learning
and epsilon-greedy algorithm.

**WHAT IS REINFORCEMENT LEARNING?**

*The four features of RL*

RL comes down to four features:
-) environment - the world that we want to learn from. 
                 A single situation in an environment is a "state"
-) agent = the entity that learns through observing the environment and taking actions
-) observation - what the agent perceives about the world
-) action - what the agent does
______________________________________________

*Agent-environment interaction (step)*

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

"""

# First install dependencies as required through:
# pip install gymnasium
import gymnasium
import random, math

import cartpole_agent

QTABLE_FOLDER_PATH = "./qtable/"
DEFAULT_QTABLE_PATH = "./qtable/default.npy/"

EPISODES = 60000
PRINT_EVERY_N_EPISODES = EPISODES / 25
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
    env = cartpole_agent.CartpoleAgent(l_r=LEARNING_RATE, d_r= DISCOUNT_RATE,r_m=r_m)

    prior_reward = 0
    epsilon = INITIAL_EPSILON

    # Single episode loop:
    for episode in range(EPISODES):
        if (episode + 1) % PRINT_EVERY_N_EPISODES == 0: print("At loop ", episode + 1, "!")
        # training:
        # 0) reset the environment (env), and setup appropriate values: 
        # - state of env
        # - rewards over episode (0)
        # - done or not? which is False
        # - array storing pairs of state, action and next state 
        # - epsilon for epsilon-greedy, which decreases by INITIAL_EPSILON / EPISODES every episode

        s, _ = env.reset()
        d = False

        # Training loop:
        while not d:

            # TODO EXPLAIN WHAT EPSILON-GREEDY IS?
            # -) choose an action: if epsilon is greater than random value, choose optimal solution so far
            #    otherwise choose an action at random in the action space

            threshold = random.random()
            a = env.get_optimal_action(s) if (threshold > epsilon) else env.get_random_action(s)

            # TODO DETAILS AS TO HOW THE OPTIMAL SOLUTION IS DETERMINED IN Q LEARNING

            # -) update the environment accordingly given the action, taking: 
            # new state, new reward, terminated?, truncated?, info
            n_s, _, terminated, truncated, _ = env.step(a)

            if terminated or truncated:
                d = True

            # -) update info to adjust at the end of the step
            s = n_s
        
        # Once episode is over:
        # Update learning algorithm
        env.update()

        # Then adjust values accordingly
        if epsilon > 0.05: #epsilon modification
            if env.episode_reward > prior_reward and episode > 10000:
                epsilon = math.pow(EPSILON_DECAY_VALUE, episode - 10000)

    # End of training things
    env.close() # close the training env
    if SAVE_TRAINING_RESULT:
        env.algorithm.save()

    return env


#+++++++++++++++++++++++++++++++++++++++
#Evaluating

def evaluate(agent=None, path=None):
    if agent is None:
        agent = cartpole_agent.CartpoleAgent(l_r=LEARNING_RATE, d_r=DISCOUNT_RATE)
        agent.algorithm.load(path)

    # Evaluation loop:
    env_eval = gymnasium.make("CartPole-v1", render_mode="human")
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
    if TRAIN_AGENT and False:
        env = train()
    # Evaluation? See it in action, probably + store the result in some way & allow reading.
    # evaluate(env)
    evaluate(path="qtable\Cartpole_Q_table_2023_3_23_0_38.npy")