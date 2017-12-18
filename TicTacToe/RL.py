from State import State
from Policy import Policy
import numpy

'''

    Q-Learning 

'''
class RL:

    def __init__(self, epsilon):
        self.policy = Policy(epsilon)

    def QLearning(self, discount, _lambda, alpha, episodes):
        matchRecord = [0]*3
        for i in range(episodes):
        
            state = State()                  #Init empty board
            avMoves = [True]*9               #Available moves for current player
            movesRem = 9                     #Remaining possible moves
            E1 = numpy.zeros((3**9,9))       #Initialize elegibility traces E matrix for all possible states & actions for P1
            E2 = numpy.zeros((3**9,9))       #Initialize elegibility traces E matrix for all possible states & actions for P2
            playerWon = 0                    #Variable for registering game results
            
            while(True):        
                '''
                
                P1 Move
                
                '''
                id1_bef = state.getID()                              #State ID before P1 moved
                A1 = self.policy.epsilonGreedy(id1_bef, avMoves)     #Choose P1 action
                state.act(A1, 1)
                id2_new = state.getID()                              #Resulting state from P2 perspective
                R1 = state.reward(1)                                 #Reward for P1
                avMoves[A1] = False
                movesRem-=1
                
                #If P1 won with its previous move update its Q values as well as P2 Q values
                if(R1==1):
                    delta = R1 - self.policy.Q[id1_bef][A1]
                    E1[id1_bef][A1] += 1
                    self.policy.Q+=alpha*delta*E1
                    E1 = discount*_lambda*E1
                
                    delta = -1*R1 - self.policy.Q[id2_bef][A2]
                    E2[id2_bef][A2] += 1
                    self.policy.Q+=alpha*delta*E2
                    E2 = discount*_lambda*E2
                    
                    playerWon = 1
                    break
                
                #If P2 has made at least one move update Q values from P2 actions
                if(movesRem<8):
                    tie = False
                    
                    #Calculate P2 delta considering the game may have been a tie
                    if(movesRem>0):
                        delta = R2 + discount*self.policy.Q[id2_new][self.policy.greedy(id2_new,avMoves)] - self.policy.Q[id2_bef][A2]  #Error between Optimal Future Return (new State for P2)
                                                                                                                        #and Return from Performed Action
                    else:
                        delta = R2 - self.policy.Q[id2_bef][A2]
                        tie=True
                                                                                                    
                    E2[id2_bef][A2] += 1                                                                       
                    self.policy.Q += alpha*delta*E2                                                                   
                    E2= discount*_lambda*E2
                    if(tie):
                        break
                '''
                
                P2 Move
                
                '''
                id2_bef = id2_new                                       #State ID before P2 moved
                A2 = self.policy.epsilonGreedy(id2_bef, avMoves)        #Choose P2 action
                state.act(A2, -1)
                id1_new = state.getID()                                 #Resulting state from P1 perspective
                R2 = state.reward(-1)                                   #Check P2 Win
                avMoves[A2] = False
                movesRem-=1
                
                #If P2 won with its previous move update both P1 and P2 Q values
                if(R2==1):
                    delta = -1*R2 - self.policy.Q[id1_bef][A1]
                    E1[id1_bef][A1] += 1
                    self.policy.Q+=alpha*delta*E1
                    E1 = discount*_lambda*E1
                
                    delta = R2 - self.policy.Q[id2_bef][A2]
                    E2[id2_bef][A2] += 1
                    self.policy.Q+=alpha*delta*E2
                    E2 = discount*_lambda*E2
                    
                    playerWon = -1
                    break
                    
                #Update Q values from P1 actions (when game continues)
                delta = R1 + discount*self.policy.Q[id1_new][self.policy.greedy(id1_new,avMoves)] - self.policy.Q[id1_bef][A1]   #Error between Optimal Future Return (new State for P1)
                                                                                                                    #and Return from Performed Action
                E1[id1_bef][A1] += 1                                                                        
                self.policy.Q += alpha*delta*E1                                       
                E1 = discount*_lambda*E1
                
            
            matchRecord[playerWon+1]+=1
            if((i+1)%100==0):
                print("P1: ", matchRecord[2], "% T: " , matchRecord[1], "% P2: " , matchRecord[0], "%")
                matchRecord = [0]*3
            
                
        
