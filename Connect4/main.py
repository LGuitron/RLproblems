import numpy as np
from RandomAgent import RandomAgent
from HumanAgent import HumanAgent
from DQNAgent import DQNAgent
from Game import sim_games, rendered_games

board_size      = (6,7)
connect_to_win  = 4

dqn_agent   = DQNAgent(board_size, load_path = "models/dqn_model1_6_7")
rand_agent  = RandomAgent()
human_agent = HumanAgent()

train_episodes  = 1000
test_episodes   = 200
train_stats     = np.zeros(5)
test_stats      = np.zeros(5)
avg_moves_test  = 0
avg_moves_train = 0



while test_stats[4] < 0.95*test_episodes:
    avg_moves_train, train_stats = sim_games(rand_agent, dqn_agent, board_size, connect_to_win, episodes=train_episodes)
    avg_moves_test, test_stats = sim_games(rand_agent, dqn_agent, board_size, connect_to_win, episodes=test_episodes , doTraining=False)


dqn_agent.save()
rendered_games(dqn_agent, human_agent, board_size, connect_to_win)
