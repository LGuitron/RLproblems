from State import State
from Transition import Transition
from Approximator import Approximator
import torch
from torch.autograd import Variable
import numpy
import time
'''
    Q-Learning
'''
class RL:

    def __init__(self, batch_size, learning_rate , epsilon, discount , _lambda, experience_stored, step_delta, display_frequency, runOnGPU):
        self.approximator = Approximator(batch_size,learning_rate, epsilon, _lambda, experience_stored, step_delta, runOnGPU)
        self.discount = discount
        self.display_frequency = display_frequency

        #Variables used for displaying learning stats
        self.matchRecord = [0]*3
        self.totalLoss = 0                                                   #Store the total loss from weight updates
        self.totalGameLength = 0                                             #Store game duration for later stats
        self.updateCount = 0                                                 #Store amount of updates made to calculate average loss
        self.initialTime =  time.time()                                      #Record time required for specified number of episodes

    '''
    CPU Learning
    '''

    def QLearningCPU(self, episodes):
        self.initialTime = time.time()
        for i in range(episodes):
            state = State()                                             #Init empty board
            while(True):
                loss, updated = self.approximator.updateWeightsCPU(self.discount)                            #Make weight updates at the begining of each turn
                self.totalLoss += loss
                self.updateCount += updated
                '''
                #P1 Turn
                '''
                tensor1_bef = state.getTensor()                                                        #State before P1 moved
                A1 = self.approximator.epsilonGreedy(Variable(tensor1_bef), state.avMoves)             #Choose action epsilon greedly
                state.act(A1)
                tensor2_new = state.getTensor()                                                        #Resulting state from P2 perspective
                R1 = state.reward()                                                                    #Immediate reward for P1

                if(state.movesLeft<41):                                                                #Store transactions starting in the second turn
                    if(self.P1turnTransactions(tensor1_bef, tensor1_new, tensor2_bef, tensor2_new, A1, A2, R1, R2, state)):
                        break

                '''
                #P2 Turn
                '''
                tensor2_bef = tensor2_new                                                               #Feature vector before P2 moved
                A2 = self.approximator.epsilonGreedy(Variable(tensor2_bef), state.avMoves)              #Choose action epsilon greedly
                state.act(A2)
                tensor1_new = state.getTensor()                                                         #Resulting state from P1 perspective
                R2 = state.reward()                                                                     #Check P2 Win

                if(self.P2turnTransactions(tensor1_bef, tensor1_new, tensor2_bef, tensor2_new, A1, A2, R1, R2, state)):
                    break
            self.displayStats(i)

    '''
    GPU Learning
    '''

    def QLearningGPU(self, episodes):
        self.initialTime = time.time()
        for i in range(episodes):
            state = State()                                             #Init empty board
            while(True):
                loss, updated = self.approximator.updateWeightsGPU(self.discount)                            #Make weight updates at the begining of each turn
                self.totalLoss += loss
                self.updateCount += updated
                '''
                #P1 Turn
                '''
                tensor1_bef = state.getTensor()                                                        #State before P1 moved
                A1 = self.approximator.epsilonGreedy(Variable(tensor1_bef).cuda(), state.avMoves)             #Choose action epsilon greedly
                state.act(A1)
                tensor2_new = state.getTensor()                                                        #Resulting state from P2 perspective
                R1 = state.reward()                                                                    #Immediate reward for P1

                if(state.movesLeft<41):                                                                #Store transactions starting in the second turn
                    if(self.P1turnTransactions(tensor1_bef, tensor1_new, tensor2_bef, tensor2_new, A1, A2, R1, R2, state)):
                        break
                '''
                #P2 Turn
                '''
                tensor2_bef = tensor2_new                                                               #Feature vector before P2 moved
                A2 = self.approximator.epsilonGreedy(Variable(tensor2_bef).cuda(), state.avMoves)              #Choose action epsilon greedly
                state.act(A2)
                tensor1_new = state.getTensor()                                                         #Resulting state from P1 perspective
                R2 = state.reward()                                                                     #Check P2 Win

                if(self.P2turnTransactions(tensor1_bef, tensor1_new, tensor2_bef, tensor2_new, A1, A2, R1, R2, state)):
                    break
            self.displayStats(i)

    '''
    Other functions (apply for both GPU and CPU)
    '''
    #Function for displaying execution stats
    def displayStats(self,i):
        if((i+1)%self.display_frequency==0):
            if(self.updateCount>0):
                print("P1:", 100*self.matchRecord[0]/self.display_frequency, "%   T:" , 100*self.matchRecord[1]/self.display_frequency, "%   P2:" , 100*self.matchRecord[2]/self.display_frequency, "%   ep_time: ", round((time.time()-self.initialTime)/self.display_frequency, 3)  ,"  ep_length:" , round(self.totalGameLength/self.display_frequency, 1), "  LOSS: " , self.totalLoss/self.updateCount)
            else:
                print("P1:", 100*self.matchRecord[0]/self.display_frequency, "%   T:" , 100*self.matchRecord[1]/self.display_frequency, "%   P2:" , 100*self.matchRecord[2]/self.display_frequency, "%   ep_time: ", round((time.time()-self.initialTime)/self.display_frequency, 3),"  ep_length:" , round(self.totalGameLength/self.display_frequency, 1))
            self.matchRecord = [0]*3
            self.totalLoss=0
            self.updateCount=0
            self.totalGameLength = 0
            self.initialTime = time.time()


    #Store transactions after player 1 turn
    def P1turnTransactions(self, tensor1_bef, tensor1_new, tensor2_bef, tensor2_new, A1, A2, R1, R2, state):
        #If P1 won with its previous move update weights for its last action, as well as P2's last action
        if(R1==1):
            self.approximator.addExperience(Transition(tensor1_bef, A1, R1, tensor1_new, False))           #Store transition information from P1 interactions
            self.approximator.addExperience(Transition(tensor2_bef, A2, -1*R1, tensor2_new, False))        #Store transition information from P2 interactions
            self.matchRecord[0]+=1
            self.totalGameLength += 42 - state.movesLeft
            return True     #Game ended

        #If P2 has made at least one move update Q values from P2 actions
        else:
            self.approximator.addExperience(Transition(tensor2_bef, A2, R2, tensor2_new, state.avMoves))  #Store transition information from P2 interactions
        return False        #Game continues

    def P2turnTransactions(self, tensor1_bef, tensor1_new, tensor2_bef, tensor2_new, A1, A2, R1, R2, state):
        #If P2 won with its previous move update both P1 and P2 Q values
        if(R2==1):
            self.approximator.addExperience(Transition(tensor1_bef, A1, -1*R2, tensor1_new, False))     #Store transition information from P1 interactions
            self.approximator.addExperience(Transition(tensor2_bef, A2, R2, tensor2_new, False))        #Store transition information from P2 interactions
            self.matchRecord[2]+=1
            self.totalGameLength += 42 - state.movesLeft
            return True     #Game ended

        #Update weights for P1 actions
        if(state.movesLeft>0):                                                                              #Game continues
            self.approximator.addExperience(Transition(tensor1_bef, A1, R1, tensor1_new, state.avMoves))    #Store transition information from P1 interactions
        else:                                                                                               #Game tied
            self.approximator.addExperience(Transition(tensor1_bef, A1, R1, tensor1_new, False))            #Store transition information from P1 interactions
            self.approximator.addExperience(Transition(tensor2_bef, A2, R2, tensor2_new, False))            #Store transition information from P2 interactions
            self.matchRecord[1]+=1
            self.totalGameLength += 42
            return True     #Game ended
        return False        #Game continues
