import numpy as np
import random
from Transition import Transition
from keras.models import Sequential, Model
from keras.layers import Dense, Concatenate, Input, Conv2D, Flatten
from keras.models import load_model

class QAgent:

    #def __init__(self, batch_size, learning_rate , initial_epsilon, epsilon_decay ,discount, experience_stored, step_delta):
    
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
        
        # TODO IMPROVE MODEL
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
        
        # Store previous input (state1, action)
        self.prev_input   = [None, None] 
        
        
        
    def play(self, state, actions, reward):
        
        one_hot_board = self.transform_board(state)
        
        # Make random move
        if self.epsilon < random.random():
            
            #print("RAND")
            sel_action                     = random.choice(actions)
            sel_action_input               = np.zeros((1,self.board_size[1]))
            sel_action_input[0,sel_action] = 1
            
            # Add transition to experience memory
            if(self.prev_input[0] is not None):
                self.add_experience(Transition(self.prev_input[0], self.prev_input[1], reward, one_hot_board, actions))
            
            self.prev_input = [one_hot_board, sel_action_input]
            return sel_action

        # Use model to pick move
        else:
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
            
            # Add transition to experience memory
            if(self.prev_input[0] is not None):
                self.add_experience(Transition(self.prev_input[0], self.prev_input[1], reward, one_hot_board, actions))
            
            
            # Update previous action whenever there is one
            #if(self.prev_input[0] is not None):
            #    q_target = reward + self.discount*max_reward
            #    self.model.fit(x = [self.prev_input[0], self.prev_input[1]], y = [q_target], batch_size = 1, verbose=None)
            
            # Sample from experience and update model
            
            
            
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
    def receive_last_reward(self, final_state, reward):
        #if(self.model is not None):
        one_hot_board = self.transform_board(final_state)
        self.add_experience(Transition(self.prev_input[0], self.prev_input[1], reward, one_hot_board, np.zeros(0)))
        #q_target = reward
        #self.model.fit(x = [self.prev_input[0], self.prev_input[1]], y = [q_target], verbose=None)
        self.prev_input = [None, None]
        
        # Fit once after finishing a game
        #self.fit_model()
    
    
    # Add a transition into experience memory
    def add_experience(self, transition):
        self.experience[self.exp_index] = transition
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
            action       = np.zeros((self.batch_size,self.board_size[1]))
            reward       = np.zeros(self.batch_size)
            state2       = np.zeros((self.batch_size,self.board_size[0], self.board_size[1], 2))
            action_space = []
            future_rew  = np.zeros(self.batch_size)
            
            for i, memory_index in enumerate(indices):
                state1[i] = self.experience[memory_index].state1
                action[i] = self.experience[memory_index].action
                reward[i] = self.experience[memory_index].reward
                state2[i] = self.experience[memory_index].state2
                action_space.append(self.experience[memory_index].action_space)
               
            # Generate predictions for optimal future rewards with all actions
            future_q_vals = np.zeros((self.batch_size, self.board_size[1]))
            for i in range(self.board_size[1]):
                future_action     = np.zeros((self.batch_size, self.board_size[1]))
                future_action[:,i]  = 1
                future_q_vals[:,i] = self.model.predict(x = [state2,  future_action], batch_size = self.batch_size)[:,0]
            
            # Iterate through all actions to pick in each state2
            for i in range(self.batch_size):
                if len(action_space[i]) != 0:
                    best_q = future_q_vals[i, action_space[i][0]]
                else:
                    continue
                
                for j in range(1, len(action_space[i])):
                    current_q = future_q_vals[i, action_space[i][j]]
                    if current_q > best_q:
                        best_q = current_q

                future_rew[i] = best_q
            
            
            # Fit model
            q_target = reward + self.discount*future_rew
            self.model.fit(x = [state1, action], y = [q_target], batch_size = self.batch_size, verbose=None)
