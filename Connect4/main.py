import numpy as np
from KerasModels import *
from DQNAgent import DQNAgent
from HumanAgent import HumanAgent
from RandomAgent import RandomAgent
from Game import sim_games, rendered_games


board_size      = (6,7)
connect_to_win  = 4

model1, model1_name = compile_model1(board_size)
model2, model2_name = compile_model2(board_size)

dqn_agent_1   = DQNAgent(board_size, "models/dqn_model1_6_7", model1, model1_name)
dqn_agent_2 = DQNAgent(board_size, "models/dqn_model2_6_7", model2, model2_name)
rand_agent  = RandomAgent()
human_agent = HumanAgent()

train_episodes  = 1000
test_episodes   = 200
train_stats     = np.zeros(5)
test_stats      = np.zeros(5)
avg_moves_test  = 0
avg_moves_train = 0



#while test_stats[4] < 0.9*test_episodes:
for i in range(5):
    avg_moves_train, train_stats = sim_games(dqn_agent_1, dqn_agent_2, board_size, connect_to_win, episodes=train_episodes)
    #avg_moves_test, test_stats = sim_games(dqn_agent, dqn_agent_2, board_size, connect_to_win, episodes=test_episodes , doTraining=False)


dqn_agent_1.save()
dqn_agent_2.save()
#rendered_games(dqn_agent, human_agent, board_size, connect_to_win)
