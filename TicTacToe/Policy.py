import numpy
import random

class Policy:
    
    def __init__(self, epsilon):
        self.Q = numpy.zeros((3**9,9))              #Initialize action values Q matrix for all possible states & actions
        self.epsilon = epsilon
    
    def setEpsilon(self, epsilon):
        self.epsilon=epsilon
    
    #Select action with epsilon greedy policy
    def epsilonGreedy(self, stateId, avMoves):          
       
        #Act Greedly
        if(random.random()>self.epsilon):
            return self.greedy(stateId, avMoves)
        
        #Act Randomly
        else:
            return self.randomAction(avMoves)
    
    #Random Action
    def randomAction(self, avMoves):
        possibleActions = []
        for i in range(len(avMoves)):
            if(avMoves[i]):
                possibleActions.append(i)
        index = random.randrange(0,len(possibleActions))
        return possibleActions[index]
    
    
    #Select action with greedy policy
    def greedy(self, stateId, avMoves):          
        
        stateQ = self.Q[stateId]
        bestQ = -2               #Best Q value found so far for this state
        bestActions = []         #Array of all best actions found

        for i in range(len(stateQ)):
            if(avMoves[i]):
                
                #Reset best actions array if a better action was found
                if(stateQ[i]>bestQ):
                    bestActions = []
                    bestActions.append(i)
                    bestQ = stateQ[i]
            
                #Add action to list of best actions
                elif(stateQ[i]==bestQ):
                    bestActions.append(i)
        
        #Select one action randomly from best solutions found
        index = random.randrange(0,len(bestActions))
        return bestActions[index]
