import time
import numpy as np
from Environment import Environment

'''
Games for AI (board not displayed)
'''
def sim_games(p1, p2, board_size, connections_to_win, episodes=1000, doTraining = True , stats_display_freq = 100):
    
    # Set epsilon_greedy to players who have it
    if hasattr(p1, 'is_training'):
        p1.is_training = doTraining
    
    if hasattr(p2, 'is_training'):
        p2.is_training = doTraining

    # [P1 wins, P2 wins, Ties, Agent1 Wins, Agent2 Wins]
    stats      = np.zeros(5)            
    prev_stats = np.zeros(5)
    moves_made = 0
    
    current_time = time.time()
    for i in range(episodes):
        
        # Make players alternate the turn they play each game
        if i%2 == 0:
            connect4         = Environment(p1,p2, board_size, connections_to_win)
        else:
            connect4         = Environment(p2,p1, board_size, connections_to_win)
        
        done             = False
        while not done:
            done = connect4.request_action()
        stats[connect4.winner] += 1
        moves_made             += connect4.moves_made
        
        # Add stats to agent's win
        if connect4.winner == 0:
            stats[3 + i%2] += 1
        elif connect4.winner == 1:
            stats[4 - i%2] += 1
            
        
        
        if (i+1)%stats_display_freq == 0:
            elapsed_time = time.time()-current_time            
            print("P1: " , int(stats[0] - prev_stats[0]), " P2: " , int(stats[1] - prev_stats[1]), " Ties: " , int(stats[2] - prev_stats[2]), " A1: ", int(stats[3] - prev_stats[3]), " A2: ", int(stats[4] - prev_stats[4]) , " Time: ", '{0:.3f}'.format(elapsed_time) , "   Avg. Moves: " , '{0:.3f}'.format(moves_made/(i+1)))
            
            prev_stats   = np.copy(stats)
            current_time = time.time()
        
    return moves_made/episodes, stats

'''
Games to be rendered move by move
'''
def rendered_games(p1, p2, board_size, connections_to_win):
    
    # Remove randomness from the players
    if hasattr(p1, 'is_training'):
        p1.is_training = False
    
    if hasattr(p2, 'is_training'):
        p2.is_training = False
    
    print("----------")
    print("Connect ", connections_to_win)
    print("----------")
    
    while True:
        print("New Game: \n")
        connect4 = Environment(p1,p2, board_size, connections_to_win)
        done     = False
        while not done:
            connect4.print_board()
            done = connect4.request_action()

        connect4.print_board()
        print("-----------")
        print("Game Over: ")
        if connect4.winner == 0:
            print("P1 wins")
        elif connect4.winner ==1:
            print("P2 wins")
        else:
            print("Tie")
        print("-----------\n")
        
        exit_value = input("Enter 1 to play another game: ")
        if exit_value != "1":
            break
