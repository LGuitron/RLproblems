from State import State
from Transition import Transition
from Approximator import Approximator
import torch
from torch.autograd import Variable
import numpy
'''

    Q-Learning

'''
class RL:

    def __init__(self, batch_size, learning_rate , epsilon, discount , _lambda, experience_stored, step_delta, display_frequency, runOnGPU):
        self.approximator = Approximator(batch_size,learning_rate, epsilon, _lambda, experience_stored, step_delta, runOnGPU)
        self.discount = discount
        self.display_frequency = display_frequency

    '''
    CPU
    '''

    def QLearningCPU(self, episodes):
        matchRecord = [0]*3
        totalLoss = 0                                                   #Store the total loss from weight updates
        updateCount= 0                                                  #Store amount of updates made to calculate average loss
        for i in range(episodes):
            state = State()                                             #Init empty board
            while(True):
                loss, updated = self.approximator.updateWeightsCPU(self.discount)                            #Make weight updates at the begining of each turn
                totalLoss += loss
                updateCount += updated
                '''
                #P1 Turn
                '''
                tensor1_bef = state.getTensor()                                                        #State before P1 moved
                A1 = self.approximator.epsilonGreedy(Variable(tensor1_bef), state.avMoves)             #Choose action epsilon greedly
                state.act(A1)
                tensor2_new = state.getTensor()                                                        #Resulting state from P2 perspective
                R1 = state.reward()                                                                    #Immediate reward for P1

                #If P1 won with its previous move update weights for its last action, as well as P2's last action
                if(R1==2.2):
                    self.approximator.addExperience(Transition(tensor1_bef, A1, R1, tensor1_new, False))           #Store transition information from P1 interactions
                    self.approximator.addExperience(Transition(tensor2_bef, A2, -1*R1, tensor2_new, False))        #Store transition information from P2 interactions
                    matchRecord[0]+=1
                    break

                #If P2 has made at least one move update Q values from P2 actions
                if(state.movesLeft<41):
                    self.approximator.addExperience(Transition(tensor2_bef, A2, R2, tensor2_new, state.avMoves))  #Store transition information from P2 interactions
                '''
                #P2 Turn
                '''
                tensor2_bef = tensor2_new                                                               #Feature vector before P2 moved
                A2 = self.approximator.epsilonGreedy(Variable(tensor2_bef), state.avMoves)              #Choose action epsilon greedly
                state.act(A2)
                tensor1_new = state.getTensor()                                                         #Resulting state from P1 perspective
                R2 = state.reward()                                                                     #Check P2 Win

                #If P2 won with its previous move update both P1 and P2 Q values
                if(R2==2.2):
                    self.approximator.addExperience(Transition(tensor1_bef, A1, -1*R2, tensor1_new, False))     #Store transition information from P1 interactions
                    self.approximator.addExperience(Transition(tensor2_bef, A2, R2, tensor2_new, False))        #Store transition information from P2 interactions
                    matchRecord[2]+=1
                    break

                #Update weights for P1 actions
                if(state.movesLeft>0):                                                                          #Game continues
                    self.approximator.addExperience(Transition(tensor1_bef, A1, R1, tensor1_new, state.avMoves)) #Store transition information from P1 interactions
                else:                                                                                   #Game tied
                    self.approximator.addExperience(Transition(tensor1_bef, A1, R1, tensor1_new, False)) #Store transition information from P1 interactions
                    matchRecord[1]+=1
                    break

            #display match stats
            if((i+1)%self.display_frequency==0):
                matchRecord
                print("P1: ", 100*matchRecord[0]/self.display_frequency, "% T: " , 100*matchRecord[1]/self.display_frequency, "% P2: " , 100*matchRecord[2]/self.display_frequency, "%        LOSS: " , totalLoss/updateCount)
                matchRecord = [0]*3
                totalLoss=0
                updateCount=0

    '''
    GPU
    '''

    def QLearningGPU(self, episodes):
        matchRecord = [0]*3
        totalLoss = 0                                                   #Store the total loss from weight updates
        updateCount= 0                                                  #Store amount of updates made to calculate average loss
        for i in range(episodes):
            state = State()                                             #Init empty board
            while(True):
                loss, updated = self.approximator.updateWeightsGPU(self.discount)                            #Make weight updates at the begining of each turn
                totalLoss += loss
                updateCount += updated
                '''
                #P1 Turn
                '''
                tensor1_bef = state.getTensor()                                                        #State before P1 moved
                A1 = self.approximator.epsilonGreedy(Variable(tensor1_bef).cuda(), state.avMoves)             #Choose action epsilon greedly
                state.act(A1)
                tensor2_new = state.getTensor()                                                        #Resulting state from P2 perspective
                R1 = state.reward()                                                                    #Immediate reward for P1

                #If P1 won with its previous move update weights for its last action, as well as P2's last action
                if(R1==2.2):
                    self.approximator.addExperience(Transition(tensor1_bef, A1, R1, tensor1_new, False))           #Store transition information from P1 interactions
                    self.approximator.addExperience(Transition(tensor2_bef, A2, -1*R1, tensor2_new, False))        #Store transition information from P2 interactions
                    matchRecord[0]+=1
                    break

                #If P2 has made at least one move update Q values from P2 actions
                if(state.movesLeft<41):
                    self.approximator.addExperience(Transition(tensor2_bef, A2, R2, tensor2_new, state.avMoves))  #Store transition information from P2 interactions
                '''
                #P2 Turn
                '''
                tensor2_bef = tensor2_new                                                               #Feature vector before P2 moved
                A2 = self.approximator.epsilonGreedy(Variable(tensor2_bef).cuda(), state.avMoves)              #Choose action epsilon greedly
                state.act(A2)
                tensor1_new = state.getTensor()                                                         #Resulting state from P1 perspective
                R2 = state.reward()                                                                     #Check P2 Win

                #If P2 won with its previous move update both P1 and P2 Q values
                if(R2==2.2):
                    self.approximator.addExperience(Transition(tensor1_bef, A1, -1*R2, tensor1_new, False))     #Store transition information from P1 interactions
                    self.approximator.addExperience(Transition(tensor2_bef, A2, R2, tensor2_new, False))        #Store transition information from P2 interactions
                    matchRecord[2]+=1
                    break

                #Update weights for P1 actions
                if(state.movesLeft>0):                                                                          #Game continues
                    self.approximator.addExperience(Transition(tensor1_bef, A1, R1, tensor1_new, state.avMoves)) #Store transition information from P1 interactions
                else:                                                                                   #Game tied
                    self.approximator.addExperience(Transition(tensor1_bef, A1, R1, tensor1_new, False)) #Store transition information from P1 interactions
                    matchRecord[1]+=1
                    break

            #display match stats
            if((i+1)%self.display_frequency==0):
                matchRecord
                print("P1: ", 100*matchRecord[0]/self.display_frequency, "% T: " , 100*matchRecord[1]/self.display_frequency, "% P2: " , 100*matchRecord[2]/self.display_frequency, "%        LOSS: " , totalLoss/updateCount)
                matchRecord = [0]*3
                totalLoss=0
                updateCount=0
