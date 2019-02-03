import numpy as np
import random
from copy import deepcopy
from Transition import Transition
from PickleUtilities import save_data, load_data
from pathlib import Path
from keras.models import Sequential, Model
from keras.layers import Dense, Concatenate, Input, Conv2D, Flatten
from keras.models import load_model

'''

QAgent that receives current board as input

'''
class QAgent:
    
    def __init__(self, board_size, isPlayer1, load_name = ""):

        self.board_size = board_size
        
        # Q Learning parameters
        self.batch_size        = 16
        self.discount          = 0.9
        self.epsilon           = 0.1
        self.exp_size          = 5000
        self.exp_index         = 0
        self.experience        = []
        
        # Determine if player uses epsilon_greedy startegy and makes updates on its model
        self.is_training       = True                   
        
        # Player turn P1 move_value = 1, P2 move_value = 2
        self.player     = 1
        self.move_value = 1
        if not isPlayer1:
            self.player     =  2
            self.move_value = -1
        
        # Store total updates and episodes run (while on training mode)
        self.episode_count = 0
        self.update_count  = 0
        
        # Load information from files
        expfile = Path("models/" + load_name + "_p" + str(self.player) + ".h5")
        if expfile.is_file():
            self.model          = load_model("models/" + load_name + "_p" + str(self.player) + ".h5")
            loaded_values       = load_data("experience/" + load_name + "_p" + str(self.player) + ".pkl")
            self.episode_count  = loaded_values[0]
            self.update_count   = loaded_values[1]
            self.exp_index      = loaded_values[2]
            self.experience     = loaded_values[3]
            self.print_train_history()
            
        # Start new model
        else:
            self.board_input   = Input(shape=(self.board_size[0],self.board_size[1],2), name='board')
            self.board_conv1   = Conv2D(8, (3,3), strides=(1, 1), activation='relu')(self.board_input)
            self.board_conv2   = Conv2D(16, (2,2), strides=(1, 1), activation='relu')(self.board_conv1)
            self.board_flat    = Flatten()(self.board_conv2)
            self.board_dense   = Dense(10, activation='relu')(self.board_flat)
            self.out = Dense(self.board_size[1])(self.board_dense)                # Out layer for as many columns as the board has

            self.model = Model(inputs = [self.board_input], outputs = self.out)
            self.model.compile(loss='mean_squared_error', optimizer='adam')
            
        # Store previous input (state1, action)
        self.prev_state  = None
        self.prev_action = None
        
    def play(self, state, actions, reward):
        
        state_one_hot = self.transform_board(state)

        # Make random move
        if self.is_training and random.random() < self.epsilon:
            sel_action = random.choice(actions)
        
        # Make best move according to current model
        else:
            q_vals      = self.model.predict(x = [state_one_hot], batch_size=1)[0]
            max_index   = np.argmax(q_vals[actions])
            sel_action  = actions[max_index]
            
        # Add transition to experience memory
        if(self.prev_state is not None):
            self.add_experience(Transition(self.prev_state, self.prev_action, reward, state_one_hot, actions))

        self.prev_state  = state_one_hot
        self.prev_action = sel_action
        return sel_action
    

    # Represent board as one hot vector
    # First image  = binary features of player's pieces
    # Second image = binary features of opponent's pieces
    def transform_board(self, state):
        one_hot_board = np.zeros((1,self.board_size[0], self.board_size[1], 2))
        one_hot_board[:,:,:,0] = self.move_value    == state
        one_hot_board[:,:,:,1] = -1*self.move_value == state
        return one_hot_board
    
    
    # Receive reward for winning or losing the game
    def receive_last_reward(self, final_state, reward):
        state_one_hot = self.transform_board(final_state)
        self.add_experience(Transition(self.prev_state, self.prev_action, reward, state_one_hot, np.zeros(0)))
        self.prev_state     = None
        self.prev_action    = None
        
        if self.is_training:
            self.episode_count += 1
            self.fit_model()
    
    
    # Add a transition into experience memory
    def add_experience(self, transition):
        
        if len(self.experience) < self.exp_size:
            self.experience.append(deepcopy(transition))
        else:
            self.experience[self.exp_index] = deepcopy(transition)
            self.exp_index = (self.exp_index + 1) % self.exp_size

    # Function for fitting model
    def fit_model(self):
        if len(self.experience) >= self.batch_size:

            transitions = random.sample(self.experience, k=self.batch_size)
            state1       = np.zeros((self.batch_size,self.board_size[0], self.board_size[1], 2))
            action       = np.zeros(self.batch_size)
            reward       = np.zeros(self.batch_size)
            state2       = np.zeros((self.batch_size,self.board_size[0], self.board_size[1], 2))
            action2      = []                                                                       # Actions available from state2
            
            #for i, memory_index in enumerate(indices):
            for i , transition in enumerate(transitions):
                state1[i] = transition.state1
                action[i] = transition.action
                reward[i] = transition.reward
                state2[i] = transition.state2
                action2.append(transition.action2)

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
            self.update_count += 1
            
    # Function for storing agent model and memory in text files
    def save(self, connections_to_win):
        self.model.save("models/C" + str(connections_to_win) + "_B" + str(self.board_size[0]) + "_" + str(self.board_size[1]) + "_p" + str(self.player) + ".h5")
        save_data(self, "experience/C" + str(connections_to_win) + "_B" + str(self.board_size[0]) + "_" + str(self.board_size[1]) + "_p" + str(self.player) + ".pkl")
        
    # Function to display this agent's training history
    def print_train_history(self):
        print("P" + str(self.player), " Training History  -   Ep: " , self.episode_count, "   Up: ", self.update_count)
