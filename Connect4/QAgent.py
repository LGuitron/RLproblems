import numpy as np
import random
from copy import deepcopy
from Transition import Transition
from keras.models import Sequential, Model
from keras.layers import Dense, Concatenate, Input, Conv2D, Flatten
from keras.models import load_model

'''

QAgent that receives current board as input

'''
class QAgent:
    
    def __init__(self, board_size, isPlayer1):

        # Q Learning parameters
        self.batch_size        = 16
        self.discount          = 0.9
        self.exp_size          = 5000
        self.experience        = np.empty(self.exp_size, dtype=Transition)
        self.exp_index         = 0
        self.epsilon           = 0.1

        self.start_training    = False                  # Training starts when experience memory reaches batch_size
        self.experience_full   = False                  # Determine if experience memory is full
        
        # Player turn and border settings
        self.isPlayer1 = isPlayer1
        self.board_size = board_size

        # Board input
        self.board_input   = Input(shape=(self.board_size[0],self.board_size[1],2), name='board')
        self.board_conv1   = Conv2D(16, (2,2), strides=(1, 1), activation='relu')(self.board_input)
        #self.board_conv2   = Conv2D(32, (2,2), strides=(1, 1), activation='relu')(self.board_conv1)
        self.board_flat    = Flatten()(self.board_conv1)
        self.board_dense   = Dense(10, activation='relu')(self.board_flat)

        # Out layer for as many columns as the board has
        self.out = Dense(self.board_size[1])(self.board_dense)

        self.model = Model(inputs = [self.board_input], outputs = self.out)
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.save('model.h5')
        
        # Store previous input (state1, action)
        self.prev_input   = [None, None] 
        
        self.tie_count    = 0 
        
        
    def play(self, state, actions, reward):
        
        one_hot_board = self.transform_board(state)

        # Make random move
        if random.random() < self.epsilon:

            sel_action                     = random.choice(actions)
            sel_action_input               = np.zeros((1,self.board_size[1]))
            sel_action_input[0,sel_action] = 1
            
            # Add transition to experience memory
            if(self.prev_input[0] is not None):
                self.add_experience(Transition(self.prev_input[0], self.prev_input[1], reward, one_hot_board, actions))
            
            self.prev_input = [one_hot_board, sel_action_input]
            return sel_action
        
        # Make best move according to current policy
        else:
            q_vals = self.model.predict(x = [one_hot_board], batch_size=1)[0]
            max_index                      = np.argmax(q_vals[actions])
            sel_action                     = actions[max_index]
            
            # Add transition to experience memory
            if(self.prev_input[0] is not None):
                self.add_experience(Transition(self.prev_input[0], self.prev_input[1], reward, one_hot_board, actions))

            self.prev_input = [one_hot_board, sel_action]
            return sel_action
    

    # Represent board as one hot vector
    # First image  = binary features of player's pieces
    # Second image = binary features of opponent's pieces
    def transform_board(self, state):
        one_hot_board = np.zeros((1,self.board_size[0], self.board_size[1], 2))
        
        if self.isPlayer1:
            one_hot_board[:,:,:,0] =  1 == state
            one_hot_board[:,:,:,1] = -1 == state
        
        else:
            one_hot_board[:,:,:,0] = -1 == state
            one_hot_board[:,:,:,1] =  1 == state
        
        return one_hot_board
    
    
    # Receive reward for winning or losing the game
    def receive_last_reward(self, final_state, reward):
        one_hot_board = self.transform_board(final_state)
        self.add_experience(Transition(self.prev_input[0], self.prev_input[1], reward, one_hot_board, np.zeros(0)))
        self.prev_input = [None, None]
        self.fit_model()
    
    
    # Add a transition into experience memory
    def add_experience(self, transition):
        self.experience[self.exp_index] = deepcopy(transition)
        self.exp_index = (self.exp_index + 1) % self.exp_size
        
        if self.exp_index >= self.batch_size and not self.start_training:
            self.start_training = True
            print("Starting Training...")
        
        if self.exp_index == 0:
            self.experience_full = True


    # Function for fitting model
    def fit_model(self):
        if self.start_training:
            
            # Max index that can be obtained for getting samples
            max_index    = self.exp_index
            if self.experience_full:
                max_index = self.exp_size

            indices = random.sample(range(0, max_index), self.batch_size)
            
            state1       = np.zeros((self.batch_size,self.board_size[0], self.board_size[1], 2))
            action       = np.zeros(self.batch_size)
            reward       = np.zeros(self.batch_size)
            state2       = np.zeros((self.batch_size,self.board_size[0], self.board_size[1], 2))
            action2      = []                                                                       # Actions available from state2
            future_rew  = np.zeros(self.batch_size)
            
            for i, memory_index in enumerate(indices):
                state1[i] = self.experience[memory_index].state1
                action[i] = self.experience[memory_index].action
                reward[i] = self.experience[memory_index].reward
                state2[i] = self.experience[memory_index].state2
                action2.append(self.experience[memory_index].action2)

            # Forward pass to generate Q values with state1
            targetQ = self.model.predict(x = [state1], batch_size = self.batch_size)
            
            # Get optimal future Q values from state2
            futureQ        = self.model.predict(x = [state2], batch_size = self.batch_size)
            optimalFutureQ = np.zeros(self.batch_size) 
            
            for i in range(self.batch_size):
                
                targetQ[i, int(action[i])] = reward[i]
                if len(action2[i]) == 0:
                    continue

                future_reward  = np.max(futureQ[i][action2[i]])
                targetQ[i, int(action[i])] += self.discount*future_reward
            
            self.model.fit(x = [state1], y = [targetQ], batch_size = self.batch_size, verbose=None)
