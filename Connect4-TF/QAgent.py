import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Concatenate, Input, Conv2D, Flatten
from keras.models import load_model

class QAgent:

    #def __init__(self, batch_size, learning_rate , initial_epsilon, epsilon_decay ,discount, experience_stored, step_delta):
    
    def __init__(self, board_size, isPlayer1):

        # Q Learning parameters
        self.discount = 0.9
        
        # Player turn and border settings
        self.isPlayer1 = isPlayer1
        self.board_size = board_size
        
        # Board input
        self.board_input   = Input(shape=(self.board_size[0],self.board_size[1],2), name='board')
        self.x1            = Conv2D(64, (2,2), strides=(1, 1))(self.board_input)
        self.flat_x1       = Flatten()(self.x1)
        self.dense_x1      = Dense(10, activation='relu')(self.flat_x1)
        
        # Action input
        self.action_input = Input(shape=(self.board_size[1],), name = 'action')
        self.conc = Concatenate()([self.dense_x1, self.action_input])

        # Out layer is single Q value
        self.out = Dense(1)(self.conc)

        self.model = Model(inputs = [self.board_input, self.action_input], outputs = self.out)
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.save('model.h5')
        
        # Store previous input for training purposes
        self.prev_input   = [None, None] 
        
    def play(self, state, actions, reward):

        one_hot_board = self.transform_board(state)
        q_vals = np.zeros(len(actions))
        
        for i, action in enumerate(actions):
            action_input = np.zeros((1,self.board_size[1]))
            action_input[0,action] = 1
            q_vals[i] = self.model.predict(x = [one_hot_board,  action_input], batch_size=1)[0]
        
        
        
        
        # Select action with highest Q value
        max_reward                     = np.max(q_vals)
        max_index                      = np.argmax(q_vals)
        sel_action                     = actions[max_index]
        sel_action_input               = np.zeros((1,self.board_size[1]))
        sel_action_input[0,sel_action] = 1
        
        
        # Update previous action whenever there is one
        if(self.prev_input[0] is not None):
            q_target = reward + self.discount*max_reward
            self.model.fit(x = [self.prev_input[0], self.prev_input[1]], y = [q_target], batch_size = 1, verbose=None)
        
        self.prev_input = [one_hot_board, sel_action_input]
        
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
    def receive_last_reward(self, reward):
        if(self.model is not None):
            q_target = reward
            self.model.fit(x = [self.prev_input[0], self.prev_input[1]], y = [q_target], verbose=None)
            self.prev_input = [None, None]
