import numpy as np
from RandomAgent import RandomAgent
from HumanAgent import HumanAgent
from QAgent import QAgent
from Game import sim_games, rendered_games

board_size      = (4,4)
connect_to_win  = 3

q1      = QAgent(board_size, isPlayer1 = True, load_name="C3_B4_4")
q2      = QAgent(board_size, isPlayer1 = False, load_name = "C3_B4_4")

rand_agent  = RandomAgent()
human_agent = HumanAgent()


train_episodes  = 1000
test_episodes   = 200
train_stats     = np.zeros(3)
test_stats      = np.zeros(3)
avg_moves_test  = 0
avg_moves_train = 0

while avg_moves_train < 12:
    #print("-----")
    #print("Train")
    #print("-----")
    avg_moves_train, train_stats = sim_games(q1, q2, board_size, connect_to_win, episodes=train_episodes)
    #print("-----")
    #print("Test")
    #print("-----")
    #avg_moves_test, test_stats = sim_games(q1, q2, board_size, connect_to_win, episodes=test_episodes , doTraining=False)
    
q1.save(connect_to_win)
q2.save(connect_to_win)
q1.print_train_history()
q2.print_train_history()
rendered_games(q1, human_agent, board_size, connect_to_win)
