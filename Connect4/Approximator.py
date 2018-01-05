import torch
from torch.autograd import Variable
from copy import deepcopy
from QNet import QNet
import numpy
import random


class Approximator():

    #N - batch size
    #learning_rate - Step size
    #epsilon - Exploration parameter (probability of random action)
    #_lambda - Lambda value for Q-target
    #experience_stored - Number of states cached for experience replay
    #step_delta - Step difference between the target NN, and the NN currently being updated
    def __init__(self, batch_size, learning_rate, epsilon, _lambda , experience_stored, step_delta, runOnGPU):

        self.network = QNet()
        if(runOnGPU):
            self.network.cuda()
        self.epsilon = epsilon
        self.targetNetwork = deepcopy(self.network)         #Fixed model used for target values for error calculation
        self.loss_fn = torch.nn.MSELoss(size_average=False)
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=learning_rate)

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
            return self.randomAction(avMoves)

    #Get the best action from nn forward pass to be performed by the agent
    def bestAction(self, x,avMoves):
        y_pred = self.network(x)                                     #Action value for all actions in current state

        #Sorted action Q values in descending order
        sortedY, indices = torch.sort(y_pred,1, True)
        i = 0
        while(not avMoves[indices[0][i].data[0]]):                    #Pick best move available
            i+=1
        return indices[0][i].data[0]                                  #Return best action

    #Select random action from all available moves
    def randomAction(self,avMoves):
        possibleActions = []
        for i in range(len(avMoves)):
            if(avMoves[i]):
                possibleActions.append(i)
        index = random.randrange(0,len(possibleActions))
        return possibleActions[index]                               #Return random action

    #Get optimal Actions to be performed on S' states according to the frozen model
    def optimalFutureActions(self,x,avMoves):
        actions = []
        y_pred = self.targetNetwork(x)

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
            self.targetNetwork = deepcopy(self.network)
            self.currentStepDelta=0

    #Get batch_size sample from previous experience
    def sampleExperience(self):
        transitions = []
        rewards = []
        avMoves = []
        actions = []
        inputTensor = torch.FloatTensor(self.batch_size, 2, 6, 7).zero_()    #Initialize tensor for input states
        inputTensor2 = torch.FloatTensor(self.batch_size, 2, 6,7).zero_()    #Initialize tensor for states reached (used fo calculating optimal future reward with frozen parameters)

        #Sample random transitions from experience
        for i in range (self.batch_size):
            transitions.append(random.choice(self.experience))
            rewards.append(transitions[i].reward)
            avMoves.append(transitions[i].avMoves2)
            actions.append(transitions[i].action)
            inputTensor[i] = torch.FloatTensor(transitions[i].state1)
            inputTensor2[i] = torch.FloatTensor(transitions[i].state2)

        return inputTensor, inputTensor2, rewards, avMoves, actions

    #Update weights using experience replay and fixed-Q values
    def updateWeightsCPU(self, discount):

        #Update target parameters after a defined number of steps
        self.updateTargets()

        #Only update weights when the required batch size has been reached
        if(len(self.experience)>=self.batch_size):
            inputTensor, inputTensor2, rewards, avMoves, actions = self.sampleExperience()

            prevQ = self.network(Variable(inputTensor))
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

     #Update weights using experience replay and fixed-Q values
    def updateWeightsGPU(self, discount):

        #Update target parameters after a defined number of steps
        self.updateTargets()

        #Only update weights when the required batch size has been reached
        if(len(self.experience)>=self.batch_size):

            inputTensor, inputTensor2, rewards, avMoves, actions = self.sampleExperience()
            prevQ = self.network(Variable(inputTensor).cuda())
            targetQ = prevQ.clone()
            Qvals, actions2 = self.optimalFutureActions(Variable(inputTensor2).cuda(), avMoves)

            #Modify target variable only for values corresponding to actions that were actually taken
            for i in range(len(targetQ)):
                targetQ[i][actions[i]].data[0] = rewards[i]+discount*Qvals[i][actions2[i]].data[0]

            self.optimizer.zero_grad()
            loss = self.loss_fn(prevQ, Variable(targetQ.data, requires_grad=False).cuda())
            loss.backward()
            self.optimizer.step()

            self.currentStepDelta += 1
            return loss.data[0], 1             #Return loss from this update, as well as a
                                               #0 or a 1 to indicate if an update was made (useful for calculating avg loss after many updates)
        return 0,0
