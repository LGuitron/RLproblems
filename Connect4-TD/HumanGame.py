from RL import RL
from State import State
from pathlib import Path
import numpy
import torch
from torch.autograd import Variable

print("___________")
print("Connect 4")
print("___________\n")
print("Learning...")

batch_size = 16
learning_rate = 0.03
epsilon = 0.05
discount = 0.95
_lambda = 0.80
trainEpisodes = 1
experience_stored = 5000
step_delta = 50

AI = RL(batch_size , learning_rate, epsilon,discount, _lambda, experience_stored, step_delta)
AI.QLearning(trainEpisodes)

'''
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

while(True):
    val = input("\nEnter 1 to go first, enter otherwise to go second: ")

    state = State()
    stateVector = state.getVector()
    playerTurn = False
    R = 0
    inputTensor = torch.FloatTensor(1, 85).zero_() 

    if(val=="1"):
        playerTurn = True
        
    while((R!=-2.2 and R!=2.2) and state.movesLeft>0):
        state.print("+","-","0")
        
        #Player Moves
        if(playerTurn):
            #Check that the player introduced an avaiable move
            
            while(True):
                val = input("\nYour Move (1 - 7): ")
                A = eval(val)-1
                if(state.avMoves[A]):
                    break
                else:
                    state.print("+","-","0")
                    print("\nInvalid move, please select an available move only")
        #AI Moves
        else:
            print("\n AI Moved")
            inputTensor[0] = torch.FloatTensor(stateVector)
            A , _ = AI.approximator.bestAction(Variable(inputTensor), state.avMoves) 
        state.act(A)
        stateVector = state.getVector()
        R = state.reward()
        playerTurn = not playerTurn

    state.print("+","-","0")
    
    if(R!=-2.2 and R!=2.2):
        print("Tie")
    elif(R and playerTurn):
        print("AI wins")
    else:
        print("You win")
    
    val = input("\nEnter 1 to play another game: ")
    if(val!="1"):
        break
