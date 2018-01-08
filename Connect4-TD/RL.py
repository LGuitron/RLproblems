from State import State
from Approximator import Approximator
import torch
from torch.autograd import Variable
import numpy
import time
'''

    Q-Learning 

'''
class RL:

    def __init__(self, learning_rate , epsilon, discount , _lambda, display_frequency):
        self.approximator = Approximator(learning_rate, epsilon, _lambda) 
        self.discount = discount
        self.display_frequency = display_frequency
        
        #Variables used for displaying learning stats
        self.matchRecord = [0]*3
        self.totalLoss = 0                                                   #Store the total loss from weight updates
        self.totalGameLength = 0                                             #Store game duration for later stats
        self.updateCount = 0                                                 #Store amount of updates made to calculate average loss
        self.initialTime =  time.time()                                      #Record time required for specified number of episodes

    def QLearning(self, episodes):
        self.initialTime = time.time()
        for i in range(episodes):
            state = State()                                             #Init empty board                         
            
            #Elegibility traces, and actions performed by both players
            E1 = []
            E2 = []
            P1A = []
            P2A = []
            
            while(True):
                '''
                P1 Turn
                '''
                tensor1_bef = state.getTensor()                                                        #State before P1 moved
                A1, y_pred1 = self.approximator.epsilonGreedy(Variable(tensor1_bef), state.avMoves)    #Choose action epsilon greedy
                state.act(A1)
                tensor2_new = state.getTensor()                                                        #Resulting state from P2 perspective
                R1 = state.reward()                                                                    #Immediate reward for P1
                
                #Store initial state and action performed for TD update
                E1.append(tensor1_bef)
                P1A.append(A1)
                
                #If P1 won with its previous move update weights for its last action, as well as P2's last action
                if(R1==1):
                    delta = R1 - y_pred1.data[0][A1]
                    self.totalLoss += delta*delta
                    self.approximator.updateWeights(E1,P1A, delta, self.discount)
                    
                    delta = -1*R1 - y_pred2.data[0][A2]
                    self.totalLoss += delta*delta
                    self.approximator.updateWeights(E2,P2A, delta, self.discount)
                    result = 1
                    
                    self.updateCount+=2
                    self.matchRecord[0]+=1
                    self.totalGameLength += 42 - state.movesLeft
                    break
                    
                #If P2 has made at least one move update Q values from P2 actions
                if(state.movesLeft<41):
                    optA, optY = self.approximator.bestAction(Variable(tensor2_new), state.avMoves)     #Get optimal Q value for following greedy policy
                    delta = R2 +  self.discount*optY.data[0][optA] - y_pred2.data[0][A2]
                    self.totalLoss += delta*delta
                    self.approximator.updateWeights(E2,P2A, delta, self.discount)
                    
                    #Register total loss for printing stats later
                    self.totalLoss += delta*delta
                    self.updateCount+=1
                '''
                P2 Turn
                '''
                tensor2_bef = tensor2_new                                                               
                A2, y_pred2 = self.approximator.epsilonGreedy(Variable(tensor2_bef), state.avMoves)     #Choose action epsilon greedly
                state.act(A2)
                tensor1_new = state.getTensor()                                                         #Resulting state from P1 perspective
                R2 = state.reward()                                                                     #Check P2 Win
                
                #Store initial state and action performed for TD update
                E2.append(tensor2_bef)
                P2A.append(A2)
                
                
                #If P2 won with its previous move update both P1 and P2 Q values
                if(R2==1):
                    delta = -1*R2 - y_pred1.data[0][A1]
                    self.totalLoss += delta*delta
                    self.approximator.updateWeights(E1,P1A, delta, self.discount)
                    
                    
                    delta = R2 - y_pred2.data[0][A2]
                    self.totalLoss += delta*delta
                    self.approximator.updateWeights(E2,P2A, delta, self.discount)
                    
                    self.updateCount+=2
                    self.matchRecord[2]+=1
                    self.totalGameLength += 42 - state.movesLeft
                    break
                
                #Update weights for P1 actions                                                                   
                if(state.movesLeft>0):                                                                          #Game continues
                    optA, optY = self.approximator.bestAction(Variable(tensor1_new), state.avMoves)     #Get optimal Q value for following greedy policy
                    delta = R1 +  self.discount*optY.data[0][optA] - y_pred1.data[0][A1]
                    self.totalLoss += delta*delta
                    self.approximator.updateWeights(E1,P1A, delta, self.discount)
                    self.updateCount+=1
                else:                                                                                   #Game tied
                    delta = R1 - y_pred1.data[0][A1]
                    self.totalLoss += delta*delta
                    self.approximator.updateWeights(E1,P1A, delta, self.discount)
                    self.updateCount+=1
                    self.matchRecord[1]+=1
                    self.totalGameLength += 42
                    break
            
            self.displayStats(i)
                
        #Function for displaying execution stats
    def displayStats(self,i):
        if((i+1)%self.display_frequency==0):
            print("P1:", 100*self.matchRecord[0]/self.display_frequency, "%   T:" , 100*self.matchRecord[1]/self.display_frequency, "%   P2:" , 100*self.matchRecord[2]/self.display_frequency, "%   ep_time: ", round((time.time()-self.initialTime)/self.display_frequency, 3)  ,"  ep_length:" , round(self.totalGameLength/self.display_frequency, 1), "  LOSS:" , self.totalLoss/self.updateCount, " epsilon:" , round(self.approximator.epsilon,4))
            self.matchRecord = [0]*3
            self.totalLoss=0
            self.updateCount=0
            self.totalGameLength = 0
            self.initialTime = time.time()
