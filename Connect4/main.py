from Game import sim_games, rendered_games
from TrainAgent import train_DQN_agent
from PlotResults import plot_results
from RandomAgent import RandomAgent
from HumanAgent import HumanAgent
from AgentType import AgentType
from DQNAgent import DQNAgent
from KerasModels import *

board_size      = (6,7)
connect_to_win  = 4

# Keras model to be used
model, model_name       = compile_model(board_size)

#E-Greedy Agent
#epsilon_vals            = [1.0, 1.0, 0.5, 0.2, 0.1, 0.05]
#epsilon_decay           = [50000, 100000, 500000, 1000000, 1500000]
#egreedy_dqn_agent       = DQNAgent(board_size, "models/egreedy_2M", model, model_name, AgentType.EGreedy, epsilon_vals, epsilon_decay)


# Softmax Agent                                    |
temperature_vals        = [0.200, 0.050, 0.025, 0.020, 0.016, 0.013]
temperature_decay       = [50000, 100000, 500000, 1000000, 1500000]
softmax_dqn_agent       = DQNAgent(board_size, "models/dqn_model_6_7", model, model_name, AgentType.Softmax, temperature_vals, temperature_decay)


# Random Agent
rand_agent              = RandomAgent()

# Human Agent
human_agent             = HumanAgent()

train_episodes          = 1000
test_episodes           = 1
test_train_epochs       = 2000
display_stats_frequency = 1000              # Display stats after this amount of games


train_DQN_agent(softmax_dqn_agent, train_episodes, test_episodes, test_train_epochs, board_size, connect_to_win, display_stats_frequency)
plot_results(softmax_dqn_agent)

# Play against RL agent
#rendered_games(egreedy_dqn_agent, human_agent, board_size, connect_to_win)

# Test Match between agents
#sim_games(dqn_agent, prev_dqn_agent, board_size, connect_to_win, episodes=1000, doTraining=False, is_exploring=True, display_results = True)
#sim_games(dqn_agent, prev_dqn_agent, board_size, connect_to_win, episodes=2, doTraining=False, is_exploring=False, display_results = True)
#sim_games(egreedy_dqn_agent, rand_agent, board_size, connect_to_win, episodes=1000, doTraining=False, is_exploring=False, display_results = True)
