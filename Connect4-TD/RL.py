from State import State
from Transition import Transition
from NeuralNetwork import NeuralNetwork
import torch
from torch.autograd import Variable
import numpy
'''

    Q-Learning 

'''
class RL:

    def __init__(self, batch_size, learning_rate , epsilon, discount , _lambda, experience_stored, step_delta):
        #42 neurons for P1 perspective, 42 for P2 perspective, 1 for player turn = 85 neurons
        self.approximator = NeuralNetwork(batch_size,85,85,7, learning_rate, epsilon, _lambda, experience_stored, step_delta) 
        self.discount = discount

    def QLearning(self, episodes):
        matchRecord = [0]*3   
        totalLoss = 0                                                   #Store the total loss from weight updates
        updateCount= 0                                                  #Store amount of updates made to calculate average loss
        for i in range(episodes):
            state = State()                                             #Init empty board                         
            inputTensor = torch.FloatTensor(1, 85).zero_()              #Initialize input tensor for action selection
            inputTensor[0] = torch.FloatTensor(state.getVector())       #Get current state vector representation
            result = 0                                                  #Get match result (0 - Tie, 1 - P1 Win, 2 - P2 Win)
            
            while(True):
                loss, updated = self.approximator.updateWeights(self.discount)                            #Make weight updates at the begining of each turn
                totalLoss += loss
                updateCount += updated
                '''
                P1 Turn
                '''
                vector1_bef = state.getVector()                                                        #State before P1 moved
                inputTensor[0] = torch.FloatTensor(vector1_bef)
                A1, y_pred1 = self.approximator.epsilonGreedy(Variable(inputTensor), state.avMoves)    #Choose action epsilon greedly
                state.act(A1)
                vector2_new = state.getVector()                                                        #Resulting state from P2 perspective
                R1 = state.reward()                                                                    #Immediate reward for P1
                
                #If P1 won with its previous move update weights for its last action, as well as P2's last action
                if(R1==2.2):
                    
                    self.approximator.addExperience(Transition(vector1_bef, A1, R1, vector1_new, False))           #Store transition information from P1 interactions
                    self.approximator.addExperience(Transition(vector2_bef, A2, -1*R1, vector2_new, False))        #Store transition information from P2 interactions
                    #delta = R1 - y_pred1.data[0][A1]
                    #self.approximator.updateWeights(E1,P1A, delta, self.discount)
                    #delta = -1*R1 - y_pred2.data[0][A2]
                    #self.approximator.updateWeights(E2,P2A, delta, self.discount)
                    result = 1
                    break
                    
                #If P2 has made at least one move update Q values from P2 actions
                if(state.movesLeft<41):
                    
                    self.approximator.addExperience(Transition(vector2_bef, A2, R2, vector2_new, state.avMoves))  #Store transition information from P2 interactions
                    
                    #inputTensor[0] = torch.FloatTensor(vector2_new)
                    #optA, optY = self.approximator.bestAction(Variable(inputTensor), state.avMoves)     #Get optimal Q value for following greedy policy
                    #optA, optY = self.approximator.bestActionTarget(Variable(inputTensor), state.avMoves)     #Get optimal Q value for following greedy policy
                    #delta = R2 +  self.discount*optY.data[0][optA] - y_pred2.data[0][A2]
                    #totalLoss += delta
                    #self.approximator.updateWeights(E2,P2A, delta, self.discount)
                    #updateCount+=1
                '''
                P2 Turn
                '''
                vector2_bef = vector2_new                                                               #Feature vector before P2 moved
                inputTensor[0] = torch.FloatTensor(vector2_bef)
                A2, y_pred2 = self.approximator.epsilonGreedy(Variable(inputTensor), state.avMoves)     #Choose action epsilon greedly
                state.act(A2)
                vector1_new = state.getVector()                                                         #Resulting state from P1 perspective
                R2 = state.reward()                                                                     #Check P2 Win
                
                #If P2 won with its previous move update both P1 and P2 Q values
                if(R2==2.2):
                    
                    self.approximator.addExperience(Transition(vector1_bef, A1, -1*R2, vector1_new, False))     #Store transition information from P1 interactions
                    self.approximator.addExperience(Transition(vector2_bef, A2, R2, vector2_new, False))        #Store transition information from P2 interactions
                    
                    #delta = -1*R2 - y_pred1.data[0][A1]
                    #totalLoss += delta
                    #self.approximator.updateWeights(E1,P1A, delta, self.discount)
                    #delta = R2 - y_pred2.data[0][A2]
                    #totalLoss += delta
                    #self.approximator.updateWeights(E2,P2A, delta, self.discount)
                    #updateCount+=2
                    result = 2
                    break
                
                #Update weights for P1 actions                                                                   
                if(state.movesLeft>0):                                                                          #Game continues
                    self.approximator.addExperience(Transition(vector1_bef, A1, R1, vector1_new, state.avMoves)) #Store transition information from P1 interactions
                    #inputTensor[0] = torch.FloatTensor(vector1_new)
                    #optA, optY = self.approximator.bestAction(Variable(inputTensor), state.avMoves)     #Get optimal Q value for following greedy policy
                    #optA, optY = self.approximator.bestActionTarget(Variable(inputTensor), state.avMoves)     #Get optimal Q value for following greedy policy
                    #delta = R1 +  self.discount*optY.data[0][optA] - y_pred1.data[0][A1]
                    #totalLoss += delta
                    #self.approximator.updateWeights(E1,P1A, delta, self.discount)
                    #updateCount+=1
                else:                                                                                   #Game tied
                    self.approximator.addExperience(Transition(vector1_bef, A1, R1, vector1_new, False)) #Store transition information from P1 interactions
                    #delta = R1 - y_pred1.data[0][A1]
                    #totalLoss += delta
                    #self.approximator.updateWeights(E1,P1A, delta, self.discount)
                    #updateCount+=1
                    break
            
            #self.approximator.updateWeights(self.discount)
            #Information about last 100 matches played
            if(result==0):                #Tie
                matchRecord[1]+=1
            elif(result==1):              #P1 Win
                matchRecord[0]+=1
            else:                         #P2 Win
                matchRecord[2]+=1
            
            #self.approximator.test()
            if((i+1)%100==0):
                #self.approximator.updateTargets()
                print("P1: ", matchRecord[0], "% T: " , matchRecord[1], "% P2: " , matchRecord[2], "%        LOSS: " , totalLoss/updateCount)
                matchRecord = [0]*3
                totalLoss=0
                updateCount=0
