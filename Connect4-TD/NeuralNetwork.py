import torch
from torch.autograd import Variable
from copy import deepcopy
import numpy
import random


class NeuralNetwork:
    
    #N - batch size
    #D_in - input layer size
    #H - hidden layer size
    #D_out - output layer size
    #learning_rate - Step size
    #epsilon - Exploration parameter (probability of random action)
    #_lambda - Lambda value for Q-target
    #experience_stored - Number of states cached for experience replay
    #step_delta - Step difference between the target NN, and the NN currently being updated

    def __init__(self, batch_size, D_in, H, D_out, learning_rate, epsilon, _lambda , experience_stored, step_delta):
        self.epsilon = epsilon
        self.model = torch.nn.Sequential(               #Both Linear layers include bias neuron
            torch.nn.Linear(D_in, H),                   
            torch.nn.ReLU(),                            
            torch.nn.Linear(H, D_out),      
        )                                               
        
        #Fixed model used for target values for the 
        self.targetModel = deepcopy(self.model)
        self.loss_fn = torch.nn.MSELoss(size_average=False)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self._lambda = _lambda
        self.batch_size = batch_size
        self.step_delta = step_delta
        
        #Array of latest states visited for performing experience replay
        self.experience_stored = experience_stored
        self.experience = []                                                     
        self.index = 0                  #Index used when replacing transitions seen in memory
        self.currentStepDelta = 0       #Step difference between fixed model and current model
    
    #Add visited state to the memory
    def addExperience(self, transition):
        #Keep adding experience information 
        if(len(self.experience)<self.experience_stored):                                        
            self.experience.append(transition)
        
        #Replace information of old transitions
        else:
            self.experience[self.index] = transition
            self.index = (self.index+1)%self.experience_stored 
    
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
        y_pred = self.model(x)                                   #Action value for all actions in current state
        
        #Sorted action Q values in descending order
        sortedY, indices = torch.sort(y_pred,1, True)
        i = 0
        while(not avMoves[indices[0][i].data[0]]):                    #Pick best move available
            i+=1
        return indices[0][i].data[0] , y_pred                         #Return best action & output values
    
    #Select random action from all available moves
    def randomAction(self, x ,avMoves):
        y_pred = self.model(x)                                        #Action value for all actions in current state
        
        possibleActions = []
        for i in range(len(avMoves)):
            if(avMoves[i]):
                possibleActions.append(i)
        index = random.randrange(0,len(possibleActions))
        return possibleActions[index] , y_pred                        #Return random action & output values
    
    #Get optimal Actions to be performed on S' states according to the frozen model
    def optimalFutureActions(self,x,avMoves):
        actions = []
        y_pred = self.targetModel(x)
        
        #Sorted action Q values in descending order
        sortedY, indices = torch.sort(y_pred,1, True)
        for i in range(len(y_pred)):
            if(not avMoves[i]):                                      #If there are no more actions available return -1
                actions.append(-1)
            else:
                j = 0
                while(j<7 and (not avMoves[i][indices[i][j].data[0]])):       #Best move available according to frozen targets
                    j+=1
                if(j<7):
                    actions.append(indices[i][j].data[0])
                else:
                    actions.append(-1)
        return y_pred, actions
    
    #Update target model with the current model which defines the agent's behaviour
    def updateTargets(self):
        if(self.currentStepDelta==self.step_delta):
            self.targetModel = deepcopy(self.model)

    def test(self):
        count = 0
        print("Targets")
        for param in self.targetModel.parameters():
            if(count==0):
                print(param.data[0][0])
            count+=1
        
        count = 0
        print("Actions")
        for param in self.model.parameters():
            if(count==0):
                print(param.data[0][0])
            count+=1
    
    #Update weights using experience replay and fixed-Q values
    def updateWeights(self, discount):
        
        #Update target parameters after a defined number of steps
        self.updateTargets()
        
        #Only update weights when the required batch size has been reached
        if(len(self.experience)>=self.batch_size):
            
            transitions = []
            rewards = []
            avMoves = []
            actions = []
            inputTensor = torch.FloatTensor(self.batch_size, 85).zero_()         #Initialize tensor for input states
            inputTensor2 = torch.FloatTensor(self.batch_size, 85).zero_()       #Initialize tensor for states reached (used fo calculating optimal future reward with frozen parameters) 
            
            #Sample random transitions from experience
            for i in range (self.batch_size):
                transitions.append(random.choice(self.experience))
                rewards.append(transitions[i].reward)
                avMoves.append(transitions[i].avMoves2)
                actions.append(transitions[i].action)
                inputTensor[i] = torch.FloatTensor(transitions[i].state1)
                inputTensor2[i] = torch.FloatTensor(transitions[i].state2)
            
            #targetQ = self.targetModel(Variable(inputTensor2))
            prevQ = self.model(Variable(inputTensor))
            targetQ = prevQ.clone()
            Qvals, actions2 = self.optimalFutureActions(Variable(inputTensor2), avMoves)
            
            #Modify target variable only for values corresponding to actions that were actually taken
            for i in range(len(targetQ)):
                targetQ[i][actions[i]].data[0] = rewards[i]+discount*Qvals[i][actions2[i]].data[0]
                
            self.optimizer.zero_grad()
            loss = self.loss_fn(prevQ, Variable(targetQ.data, requires_grad=False))
            loss.backward()
            self.optimizer.step()
            
            self.currentStepDelta += 1
            return loss.data[0], 1             #Return loss from this update, as well as a 
                                               #0 or a 1 to indicate if an update was made (useful for calculating avg loss after many updates)
        return 0,0
'''




OLD


    #Correct neural network weights for current state and for previous states
    #Variables passed are 
        #Elegibility Traces
        #Action taken in each state
        #Delta seen from last state
        #Discount factor for future rewards
    def updateWeights(self, E, A , delta, discount):
        
        for i in range(len(E)-1, -1, -1):
            y_pred = self.model(Variable(E[i], requires_grad=False))
            ##y_pred = self.targetModel(Variable(E[i], requires_grad=False))                  #Use target model to calculate gradients for the action model
            
            #Correct output value is modified by delta with respect to the original
            y = y_pred.clone()                      
            y.data[0][A[i]] += delta            
            self.optimizer.zero_grad()
            loss = self.loss_fn(y_pred, Variable(y.data[0], requires_grad=False))
            loss.backward()
            
            #TODO find better way discount gradient for old states
            for param in self.model.parameters():
                for j in range (len(param.grad)):
                    for k in range(len(param.grad[j].data)):
                        if(param.grad[j].data[k]!=0):
                            param.grad[j].data[k] = param.grad[j].data[k]*(self._lambda * discount)**(len(E)-1-i)
            self.optimizer.step()                       #Modify weights according to Elegibility Traces
            print("Loss: " , delta)
     
    
        #Get the best action from nn forward pass
    #Target values obtained from frozen parameters
    def bestActionTarget(self, x , avMoves):
        y_pred = self.targetModel(x)                                  #Action value for all actions in current state
        
        #Sorted action Q values in descending order
        sortedY, indices = torch.sort(y_pred,1, True)
        i = 0
        while(not avMoves[indices[0][i].data[0]]):                    #Pick best move available
            i+=1
        return indices[0][i].data[0] , y_pred                         #Return best action & output values
    '''
