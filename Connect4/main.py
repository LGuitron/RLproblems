import numpy as np
from RandomAgent import RandomAgent
from HumanAgent import HumanAgent
from DQNAgent import DQNAgent
from Game import sim_games, rendered_games

board_size      = (6,7)
connect_to_win  = 4

dqn_agent_1 = DQNAgent(board_size, isPlayer1 = True, load_path = "models/model1_6_7_p1")
dqn_agent_2 = DQNAgent(board_size, isPlayer1 = False, load_path = "models/model1_6_7_p2")

rand_agent  = RandomAgent()
human_agent = HumanAgent()

train_episodes  = 1000
test_episodes   = 200
train_stats     = np.zeros(3)
test_stats      = np.zeros(3)
avg_moves_test  = 0
avg_moves_train = 0


'''
while avg_moves_train < 18:
    avg_moves_train, train_stats = sim_games(dqn_agent_1, dqn_agent_2, board_size, connect_to_win, episodes=train_episodes)
    #avg_moves_test, test_stats = sim_games(dqn_agent_1, dqn_agent_2, board_size, connect_to_win, episodes=test_episodes , doTraining=False)

dqn_agent_1.save()
dqn_agent_2.save()
'''

rendered_games(dqn_agent_1, human_agent, board_size, connect_to_win)
