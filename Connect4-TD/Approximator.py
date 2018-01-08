import torch
from torch.autograd import Variable
from copy import deepcopy
from QNet import QNet
import numpy
import random


class Approximator:

    def __init__(self, learning_rate, epsilon, _lambda):
        self.epsilon = epsilon
        self.network = QNet()
        
        #if(runOnGPU):
        #    self.network.cuda()
        self.loss_fn = torch.nn.MSELoss(size_average=False)
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=learning_rate)
        self._lambda = _lambda
    
    #Select action with epsilon greedy policy
    def epsilonGreedy(self, x , avMoves):      
        #Act Greedly
        if(random.random()>self.epsilon):
            return self.bestAction(x,avMoves)
        
        #Act Randomly
        else:
            return self.randomAction(x,avMoves) 
    
    #Get the best action from nn forward pass to be performed by the agent
    def bestAction(self, x , avMoves):
        y_pred = self.network(x)                                   #Action value for all actions in current state
        
        #Sorted action Q values in descending order
        sortedY, indices = torch.sort(y_pred,1, True)
        i = 0
        while(not avMoves[indices[0][i].data[0]]):                    #Pick best move available
            i+=1
        return indices[0][i].data[0] , y_pred                         #Return best action & output values
    
    #Select random action from all available moves
    def randomAction(self, x ,avMoves):
        y_pred = self.network(x)                                        #Action value for all actions in current state
        
        possibleActions = []
        for i in range(len(avMoves)):
            if(avMoves[i]):
                possibleActions.append(i)
        index = random.randrange(0,len(possibleActions))
        return possibleActions[index] , y_pred                        #Return random action & output values

    #Correct neural network weights for current state and for previous states
    #Variables passed are 
        #Elegibility Traces
        #Action taken in each state
        #Delta seen from last state
        #Discount factor for future rewards
    def updateWeights(self, E, A , delta, discount):
        for i in range(len(E)-1, -1, -1):
            y_pred = self.network(Variable(E[i], requires_grad=False))
            
            #Correct output value is modified by delta with respect to the original
            y = y_pred.clone()                      
            y.data[0][A[i]] += delta*(self._lambda * discount)**(len(E)-1-i) 
            self.optimizer.zero_grad()
            loss = self.loss_fn(y_pred, Variable(y.data[0], requires_grad=False))
            loss.backward()
