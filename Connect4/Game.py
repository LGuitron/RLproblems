import numpy as np
from Environment import Environment
from RandomAgent import RandomAgent
from HumanAgent import HumanAgent
from QAgent import QAgent

games               = 10000
board_size          = (4,4)
connections_to_win  = 3
p1                  = QAgent(board_size, isPlayer1 = True)
p2                  = RandomAgent()

# Game result stats
stats         = np.zeros(3)
stats_display_freq = 100


for i in range(games):
    connect4         = Environment(p1,p2, board_size, connections_to_win)
    done             = False
    while not done:
        done = connect4.request_action()
    stats[connect4.winner] += 1
    
    if (i+1)%stats_display_freq == 0:
        print("P1: " , stats[0], " P2: " , stats[1], " Ties: " , stats[2])
        stats = np.zeros(3)
