import numpy
import torch
from math import floor

class State:
    
    ##Initilize empty board##
    def __init__(self):
        self.board = [[0 for x in range(7)] for y in range(6)]                      #0 = Empty, #1 = P1 move, #-1 = P2 move
        self.playerTurn = 1                                                         #1 = P1 turn, #0 = P2 turn
        self.avMoves = [True]*7                                                     #Array for available moves in current state
        self.movesLeft = 42                                                         #Number of remaining moves available from this state
        self.P1tensor = inputTensor = torch.FloatTensor(1, 2, 6, 7).zero_()         #Input tensor for P1
        self.P2tensor = inputTensor = torch.FloatTensor(1, 2, 6, 7).zero_()         #Input tensor for P2
        
    #Get input tensor for current state
    def getTensor(self):
        if(self.playerTurn):         #P1 turn
            return self.P1tensor
        else:                        #P2 turn
            return self.P2tensor
    
    ##R received after taking action A (column played) in current S 
    def act(self, A):
        for i in range (len(self.board)):
            if(i==len(self.board)-1):           #Column is going to be full after this move
                self.avMoves[A] = False
            
            if(self.board[i][A]==0):            #Available space found
                if(self.playerTurn):
                    self.board[i][A] = 1        #Set chip value for stopping location (P1)
                    self.P1tensor[0][0][i][A] = 1
                    self.P2tensor[0][1][i][A] = 1
                else:
                    self.board[i][A] = -1       #Set chip value for stopping location (P2)
                    self.P1tensor[0][1][i][A] = 1
                    self.P2tensor[0][0][i][A] = 1
                break
        self.playerTurn = (self.playerTurn+1)%2     #Change player turn
        self.movesLeft-=1                           #Decrease number of moves
    
    #R received for this state
    #2.2 : Win
    #-2.2 : Lose
    #-0.1 : StepTaken
    def reward(self):
        
        #Horizontal Win/Loss
        for i in range(len(self.board)):                                    #Check all rows
            lastChip = -2                                                   #Chip previously viewed
            consecutiveChips = 0                                            #Consecutive chips seen so far (4 to win)
            for j in range(len(self.board[i])):
                if(self.board[i][j] != 0):                                  #Not empty space
                    if(lastChip==self.board[i][j]):                         #Equal chips increase counter
                       consecutiveChips+=1
                    else:                                                   #Different chips set consecutive count to 1
                        lastChip=self.board[i][j]
                        consecutiveChips=1
                else:                                                       #Empty spaces reset counter                                                                         
                    lastChip = -2
                    consecutiveChips = 0
                if(consecutiveChips==4):                                    #Reward of +2.2 (for agent who played last move)
                    return 2.2
                
        #Vertical Win/Loss
        for i in range(len(self.board[0])):                                 #Check all columns
            lastChip = -2                                                   #Chip previously viewed
            consecutiveChips = 0                                            #Consecutive chips seen so far (4 to win)
            for j in range(len(self.board)):
                if(self.board[j][i] != 0):                                  #Not empty space
                    if(lastChip==self.board[j][i]):                         #Equal chips increase counter
                       consecutiveChips+=1
                    else:                                                   #Different chips set consecutive count to 1
                        lastChip=self.board[j][i]
                        consecutiveChips=1
                else:                                                       #Empty spaces reset counter                                                                         
                    lastChip = -2
                    consecutiveChips = 0
                if(consecutiveChips==4):                                    #Reward of +2.2 (for agent who played last move)
                    return 2.2      
                
        #Left to Right Diagonals Win/Loss
        for i in range(len(self.board)-3):                                    #Check all possiblestarting rows (0-2)
            for j in range(len(self.board[i])-3):                             #Check all possible starting chips from this row (columns 0-3)
                    if(self.board[i][j]==0):                                  #Starting location is empty, so there is no diagonal win here
                        break
                    else:                                                     #Store the value of the chip viewed at starting position
                        lastChip = self.board[i][j]
                        consecutiveChips = 1
                        for k in range(1,4,1):                                #Check the rest of the digonal
                            if(self.board[i+k][j+k]==lastChip):
                                consecutiveChips+=1
                            else:
                                break
                        if(consecutiveChips==4):
                            return 2.2
        
        #Right to Left Diagonals Win/Loss
        for i in range(len(self.board)-3):                                    #Check all possiblestarting rows (0-2)
            for j in range(3,len(self.board[i]),1):                           #Check all possible starting chips from this row (columns 3-6)
                    if(self.board[i][j]==0):                                  #Starting location is empty, so there is no diagonal win here
                        break
                    else:                                                     #Store the value of the chip viewed at starting position
                        lastChip = self.board[i][j]
                        consecutiveChips = 1
                        for k in range(1,4,1):                                #Check the rest of the digonal
                            if(self.board[i+k][j-k]==lastChip):
                                consecutiveChips+=1
                            else:
                                break
                        if(consecutiveChips==4):
                            return 2.2
        return 0                             #No reward if nobody wins
        #return -0.1                         #-0.1 reward per step

    #Print game on console
    def print(self, P1char, P2char, emptyChar):
        for i in range(len(self.board)-1,-1,-1):
            currentLine = "["
            for j in range(len(self.board[i])):
                if(self.board[i][j]==1):                #P1
                    currentLine+=" " + P1char + " "
                elif(self.board[i][j]==-1):             #P2
                    currentLine+=" " + P2char + " "
                else:                                   #Empty
                    currentLine+=" " + emptyChar + " "
            currentLine+="]"
            print(currentLine)
