import numpy
from math import floor

class State:
    
    ##Initilize empty board##
    def __init__(self):
        self.board =  numpy.zeros((3,3))     #0 = Empty
                                             #1 = P1 move
                                             #-1 = P2 move
        
    #Get unique id (base 3 number) converted to base 10
    def getID(self):
        _id = 0
        for i in range(3):
            for j in range(3):
                _id += (self.board[i][j]+1)*3**(3*i+j)
        return int(_id)
    
    ##R received, and S' reached after taking action A in current S
    def act(self, A, player):
        self.board[floor(A/3)][A%3] = player  #Move performed by current agent
    
    #R received for this state
    #1 : Win
    #-1 : Lose
    #-0.1 : StepTaken
    
    def reward(self, player):
        
        #Horizontal Win/Lose
        for i in range(3):
            if(self.board[i][0] != 0 and self.board[i][0] == self.board[i][1] and self.board[i][0] == self.board[i][2]):
                return self.board[i][0]*player
        
        #Vertical Win/Lose
        for i in range(3):
            if(self.board[0][i] != 0 and self.board[0][i] == self.board[1][i] and self.board[0][i] == self.board[2][i]):
                return self.board[0][i]*player
        
        #Diagonal Win/Lose
        if(self.board[1][1]!=0):
            if(self.board[0][0] == self.board[1][1] and self.board[1][1] == self.board[2][2]):
                return self.board[1][1]*player
            
            if(self.board[0][2] == self.board[1][1] and self.board[1][1] == self.board[2][0]):
                return self.board[1][1]*player
        
        #Return -0.1 reward per step taken
        return -0.1
    
    
    #Print on console
    def print(self):
        print()
        for i in range(3):
            print(self.board[i])
