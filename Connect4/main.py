from Game import sim_games, rendered_games
from TrainAgent import train_DQN_agent
from PlotResults import plot_results
from RandomAgent import RandomAgent
from HumanAgent import HumanAgent
from DQNAgent import DQNAgent
from KerasModels import *

board_size      = (6,7)
connect_to_win  = 4

model2, model2_name     = compile_model2(board_size)
dqn_agent               = DQNAgent(board_size, "models/dqn_model2_6_7", model2, model2_name)
rand_agent              = RandomAgent()
human_agent             = HumanAgent()

train_episodes          = 1000
test_episodes           = 1
test_train_epochs       = 3
display_stats_frequency = 1000              # Display stats after this amount of games

# Train RL agent and plot it
train_DQN_agent(dqn_agent, train_episodes, test_episodes, test_train_epochs, board_size, connect_to_win, display_stats_frequency)
plot_results(dqn_agent)

# Train against random agent
#for i in range(5):
#    sim_games(rand_agent, dqn_agent, board_size, connect_to_win, episodes=train_episodes, display_results = True)

# Play against RL agent
# rendered_games(dqn_agent, human_agent, board_size, connect_to_win)

# Test Match between agents
#sim_games(dqn_agent, prev_dqn_agent, board_size, connect_to_win, episodes=1000, doTraining=False, epsilon_greedy=True, display_results = True)
#sim_games(dqn_agent, prev_dqn_agent, board_size, connect_to_win, episodes=2, doTraining=False, epsilon_greedy=False, display_results = True)
#sim_games(dqn_agent, rand_agent, board_size, connect_to_win, episodes=1000, doTraining=False, epsilon_greedy=False, display_results = True)
