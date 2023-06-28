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

"""

# First install dependencies as required through:
# pip install gymnasium
import gymnasium
import numpy as np
import random, time, math, tqdm

QTABLE_FOLDER_PATH = "./qtables/"
DEFAULT_QTABLE_PATH = "./qtables/default.npy/"

EPISODES = 60000
ALPHA = 0.1 # learning rate
GAMMA = 0.95 # discount rate
INITIAL_EPSILON = 1 # probability for exploration
EPSILON_DECAY_VALUE = 0.99995

RENDER_TRAINING = False
SAVE_TRAINING_RESULT = True
TRAIN_AGENT = True

# Sets up the whole Q table
class Qtable:
    NP_ARRAY_WIN_SIZE = np.array([0.25, 0.25, 0.01, 0.1])

    def __init__(self):
        discretized_observation = [30, 30, 50, 60] # determines how small we chop the observation space
        env = gymnasium.make("CartPole-v1")
        self.Q = np.random.uniform(low=0, high=1, size=(discretized_observation + [env.action_space.n]))
        # print(Q.shape)
    
    # "state" is discrete
    def get_optimal_action(self, state):
        return np.argmax(self.Q[state])
    
    # "state" is discrete
    def update_state_action_reward(self, state, action, reward):
        self.Q[state + (action, )] = reward

    # "state" is discrete
    def get_reward(self, state, action):
        return self.Q[state + (action, )]
    
    # "state" is discrete
    def get_best_reward(self, state):
        return np.max(self.Q[state])
    
    # "state" is discrete
    def update_given_state_action_next_state_pairs_in_time_order(self, s_a_n_s, rEpisode):
        s_a_n_s.reverse()
        for pairs in s_a_n_s:
            s, a, n_s = pairs[0], pairs[1], pairs[2]
            r = self.get_reward(s, a) + ALPHA * (rEpisode + GAMMA * (self.get_best_reward(n_s) - self.get_reward(s, a)))
            self.update_state_action_reward(s, a, r)

    # saves the table into a .npy file
    def save_table(self, name=None):
        time_of_creation = time.localtime(time.time())[0:5]
        acc = str(time_of_creation[0])
        [acc := acc + "_" + str(x) for x in time_of_creation[1:5]]
        name_original = "Cartpole_Q_table_" + acc
        
        if name is None:
            name = name_original

        name = QTABLE_FOLDER_PATH + "/" + name
        np.save(name, self.Q)
    
    # loads the .npy file table into a numpy array
    def load_table(self, path=None):
        if path is None:
            path = DEFAULT_QTABLE_PATH

        self.Q = np.load(path) 
    
    # Maps each continuous state vector obtained from env to a discrete state stored in Q table
    # Divides the state array by the window sizes specified by ARRAY_WIN_SIZE, then adds the np.array
    # to turn them to values above 0 (and hopefully within range of the Q table)
    @staticmethod
    def get_discrete_state(state):
        discrete_state = state/Qtable.NP_ARRAY_WIN_SIZE + np.array([15,10,1,10])
        return tuple(discrete_state.astype(int))


#+++++++++++++++++++++++++++++++++++++++
#Training

def train():
    # setup environment
    r_m = "human" if RENDER_TRAINING else None 
    env = gymnasium.make("CartPole-v1", render_mode=r_m) # setting the environment up

    # setup algorithm that learns what action to take.
    # In our case, we use, specifically the qtable for qlearning.
    Q = Qtable()

    prior_reward = 0
    epsilon = INITIAL_EPSILON

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
        s_disc = Qtable.get_discrete_state(s)
        rEpisode = 0
        d = False

        # pairs of state, action and next state in episode used at the episode's end to 
        # update values of each actions given each state (as related to entire reward in episode)
        state_action_nextstate_pairs = [] 

        # Training loop:
        while not d:

            # SPECIFICS: WHAT IS EPSILON-GREEDY?
            # -) choose an action: if epsilon is greater than random value, choose optimal solution so far
            #    otherwise choose an action at random in the action space
            optimal_action = Q.get_optimal_action(s_disc) 
            random_action = random.sample([0, 1], 1)[0]

            threshold = random.random()
            a = optimal_action if (threshold > epsilon) else random_action

            # SPECIFICS: HOW IS THE OPTIMAL SOLUTION DETERMINED IN Q LEARNING? 

            # -) update the environment accordingly given the action, taking: 
            # new state, new reward, done?, info
            n_s, r, terminated, truncated, _ = env.step(a)

            if terminated or truncated:
                d = True
        
            n_s_disc = Qtable.get_discrete_state(n_s)
            state_action_nextstate_pairs.append([s_disc, a, n_s_disc])

            # -) update info to adjust at the end of the step
            s_disc = n_s_disc
            rEpisode += r
        
        # Then adjust values accordingly once a single episode loop is over.
        if epsilon > 0.05: #epsilon modification
            if rEpisode > prior_reward and episode > 10000:
                epsilon = math.pow(EPSILON_DECAY_VALUE, episode - 10000)
        
        prior_reward = rEpisode
        
        # Update qtable appropriately
        # SPECIFICS: HOW IS THE QTABLE UPDATED?
        Q.update_given_state_action_next_state_pairs_in_time_order(state_action_nextstate_pairs, rEpisode)

    # End of training things
    env.close() # close the training env
    if SAVE_TRAINING_RESULT:
        # new_table_name = input("Input the name of the new table.\n")
        Q.save_table()

    return Q


#+++++++++++++++++++++++++++++++++++++++
#Evaluating

def evaluate(qtable=None, path=None):
    if qtable is None:
        Q = Qtable()
        Q.load_table(path)
    else:
        Q = qtable

    # Evaluation loop:
    env_eval = gymnasium.make("CartPole-v1", render_mode="human")
    s, _ = env_eval.reset() #reset the environment

    s_disc = Qtable.get_discrete_state(s)
    terminated = truncated = False

    while not (terminated or truncated):

        # get optimal action by agent
        a = Q.get_optimal_action(s_disc)

        # update env accordingly
        s, _, terminated, truncated, _ = env_eval.step(a)
        s_disc = Qtable.get_discrete_state(s)

    env_eval.close()

if __name__ == "__main__":
    #First train, details in train function
    qtable = None
    if TRAIN_AGENT and False:
        qtable = train()
    # Evaluation. See it in action, and some metrics! (TODO ADD METRICS! TO COME).
    evaluate(qtable=qtable)
    # evaluate(path="qtables\Cartpole_Q_table_2023_4_3_12_40.npy")
    

