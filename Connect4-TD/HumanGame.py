from RL import RL
from State import State
from pathlib import Path
import numpy
import torch
from torch.autograd import Variable
import sys

print("___________")
print("Connect 4")
print("___________\n")

mode = "CPU"
runOnGPU = len(sys.argv)>1      #If a console parameter is received run in GPU. else run on CPU
if(runOnGPU):
    mode = "GPU"


learning_rate = 0.00001
epsilon = 0.05
discount = 0.95
_lambda = 0.80
trainEpisodes = 20

#Number of episodes to run before displaying learning stats
display_frequency = 10

AI = RL(learning_rate, epsilon,discount, _lambda, display_frequency)

CPUfile = Path("netCPU.pt")
GPUfile = Path("netGPU.pt")

if (runOnGPU and GPUfile.is_file()) or (not runOnGPU and CPUfile.is_file()):
    print("Loaded Network", mode)
    print("Learning...")
    if(runOnGPU):
        print("Implement for GPU")
        #AI.approximator = torch.load("netGPU.pt")
        #AI.QLearningGPU(trainEpisodes)
        #torch.save(AI.approximator, "netGPU.pt")
    else:
        AI.approximator = torch.load("netCPU.pt")
        AI.QLearning(trainEpisodes)
        torch.save(AI.approximator, "netCPU.pt")

else:
    print("Starting New Training" , mode)
    print("Learning...")
    if(runOnGPU):
        print("Implement for GPU")
        #AI.QLearningGPU(trainEpisodes)
        #torch.save(AI.approximator, "netGPU.pt")
    else:
        AI.QLearning(trainEpisodes)
        torch.save(AI.approximator, "netCPU.pt")

while(True):
    val = input("\nEnter 1 to go first, enter otherwise to go second: ")

    state = State()
    stateVector = state.getTensor()
    playerTurn = False
    R = 0
    inputTensor = torch.FloatTensor(1, 2, 6, 7).zero_()    #Initialize tensor for input states

    if(val=="1"):
        playerTurn = True

    while((R!=-1 and R!=1) and state.movesLeft>0):
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
            if(runOnGPU):
                A, _ = AI.approximator.bestAction(Variable(inputTensor).cuda(), state.avMoves)
            else:
                A, _ = AI.approximator.bestAction(Variable(inputTensor), state.avMoves)
        state.act(A)
        stateVector = state.getTensor()
        R = state.reward()
        playerTurn = not playerTurn

    state.print("+","-","0")

    if(R!=-1 and R!=1):
        print("Tie")
    elif(R and playerTurn):
        print("AI wins")
    else:
        print("You win")

    val = input("\nEnter 1 to play another game: ")
    if(val!="1"):
        break
