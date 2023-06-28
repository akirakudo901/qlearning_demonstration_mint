# **Introduction to reinforcement learning (RL) with code**

A quick introduction to general flow of RL!

## To learn the basics of reinforcement learning

* *qlearning_demonstration_MINT_notebook.ipynb* gives an overview of RL and an example code as Jupyter Notebook.

* *qtable_demonstration_with_cartpole.py* holds the same content as above, but in Python code.

* *environment_ttt.py* in the folder environments is an example of RL environment I implemented a couple years ago.
Might give you some insights into designing one.

* I brought **example_qlearning.py** from [this online article](https://medium.com/swlh/using-q-learning-for-openais-cartpole-v1-4a216ef237df
), works very well! If you want to see other people's code.

If further interested, tons of resources are online! Also feel free to ask me questions!

## How this repo is structured
My goal for this repo was to separate the complex components of training into:
* the **environment**, where different agents train
* the **policy learning algorithms**, which learn the task (e.g. Q-table, DDQN)
* the **agent**, which combines a policy learning algorithm and environment into a specific way to realize learning
* the **training / evaluation loops**, which happen roughly similarly for many agents / environments (e.g. epsilon-greedy learning)

If one wants to train a new algorithm on an environment, they can: 
* define their own algorithm mimicking the definitions in the ***policy_learning_algorithms*** folder 
* define their own environment micmiking definitions in the ***environments*** folder

  **We have mostly followed the **OpenAI gymnasium framework** so far.*
* combine the environment and algorithm into an agent, mimicking the definitions in the ***agents*** folder
* use the code in main.py to train the algorithms.
