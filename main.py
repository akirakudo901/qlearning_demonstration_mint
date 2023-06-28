"""
Trains a DDQN agent on the cartpole environment, using the train and evaluate functions
from ai_training_and_evaluation.py.
"""

from agents.cartpole_dnn_agent import CartpoleDNNAgent
from ai_training_and_evaluation import train_epsilon_greedy, evaluate

parameters_for_training = {
    "DDQN" : {
        "env_agent_merged_object" : CartpoleDNNAgent,
        "episodes" : 1000,
        "exploration_episodes" : 500 // 6,
        "learning_rate" : 5e-3,
        "discount_rate" : 0.95,
        "initial_epsilon" : 1.0,
        "epsilon_decay_value" : 0.999
    },

}

def train_and_evaluate_with_this_agent(agent_name : str):
    if agent_name not in parameters_for_training.keys():
        raise Exception("That agent name is not in the dictionary!")
    
    params = parameters_for_training["DDQN"]

    trained_agent = train_epsilon_greedy(
        env_agent_merged_object=params["env_agent_merged_object"], 
        learning_rate=params["learning_rate"],
        discount_rate=params["discount_rate"],
        initial_epsilon=params["initial_epsilon"],
        epsilon_decay_value=params["epsilon_decay_value"],
        episodes=params["episodes"],
        exploration_episodes=params["exploration_episodes"],
        show_progress_every_n_episodes=None,
        render_training=False,
        save_training_result=True
        )

    evaluate(
        env_agent_merged_object=params["env_agent_merged_object"],
        trained_agent=trained_agent
    )


if __name__ == "__main__":
    train_and_evaluate_with_this_agent("DDQN")