from RL import RL
from State import State
from Policy import Policy
from pathlib import Path
import numpy

print("___________")
print("TIC TAC TOE")
print("___________\n")
print("Learning...")

trainEpisodes = 100
AI = RL(0.05)

Qfile = Path("Qvals.npy")
if Qfile.is_file():
    print("Loaded Q File")
    AI.policy.Q = numpy.load("Qvals.npy")
    AI.QLearning(1.0,0.9,0.1,trainEpisodes)

else:
    print("Starting New Training")
    AI.QLearning(1.0,0.9,0.1,trainEpisodes)
numpy.save("Qvals.npy", AI.policy.Q)

'''
Game
'''

while(True):
    val = input("\nEnter 1 to go first, enter otherwise to go second: ")

    state = State()
    stateID = state.getID()
    R=0
    movesLeft = 9
    currentPlayer = 1
    avMoves = [True]*9
    playerTurn = False

    if(val=="1"):
        playerTurn = True
        
    while((R!=-1 and R!=1) and movesLeft>0):
        state.print()
        
        #Player Moves
        if(playerTurn):
            #Check that the player introduced an avaiable move
            
            while(True):
                val = input("\nYour Move (1 - 9): ")
                A = eval(val)-1
                if(avMoves[A]):
                    break
                else:
                    state.print()
                    print("\nInvalid move, please select an available move only")
        #AI Moves
        else:
            A = AI.policy.greedy(stateID, avMoves)
        state.act(A, currentPlayer)
        stateID = state.getID()
        R = state.reward(currentPlayer)
        currentPlayer *= -1
        movesLeft-=1
        avMoves[A] = False
        playerTurn = not playerTurn

    state.print()
    if(R==-1):
        print("P2 wins")
    elif(R==1):
        print("P1 wins")
    else:
        print("Tie")
    
    val = input("\nEnter 1 to play another game: ")
    if(val!="1"):
        break
