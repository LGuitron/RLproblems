import random
import numpy as np
from pathlib import Path
from Transition import Transition
from KerasModels import compile_model1
from ExperiencedModel import ExperiencedModel
'''

Deep Q Network Agent that receives current board as input

'''
class DQNAgent:
    
    def __init__(self, board_size, isPlayer1, load_path = None):

        self.board_size = board_size
        
        # Q Learning parameters
        self.batch_size         = 16
        self.discount           = 0.9
        self.epsilon            = 0.1

        # Create new model if path is not specified or if the load_path entered does not exist
        if load_path is None or not (Path(load_path + ".h5").is_file() and Path(load_path + ".pkl").is_file()):
            comp_model, model_name = compile_model1(self.board_size)
            self.experiencedModel  = ExperiencedModel(comp_model, model_name, exp_size = 5000)
        
        # Load experiencedModel from load_path
        else:
            self.experiencedModel = ExperiencedModel.load_data(load_path)
        
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
            q_vals      = self.experiencedModel.model.predict(x = [state_one_hot], batch_size=1)[0]
            max_index   = np.argmax(q_vals[actions])
            sel_action  = actions[max_index]
            
        # Add transition to experience memory
        if(self.prev_state is not None):
            self.experiencedModel.add_experience(Transition(self.prev_state, self.prev_action, reward, state_one_hot, actions))

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
        self.experiencedModel.add_experience(Transition(self.prev_state, self.prev_action, reward, state_one_hot, np.zeros(0)))
        self.prev_state     = None
        self.prev_action    = None
        
        if self.is_training:
            self.episode_count += 1
            self.fit_model()

    # Function for fitting model
    def fit_model(self):
        if len(self.experiencedModel.experience) >= self.batch_size:

            transitions = random.sample(self.experiencedModel.experience, k=self.batch_size)
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
            targetQ = self.experiencedModel.model.predict(x = [state1], batch_size = self.batch_size)
            
            # Get optimal future Q values from state2
            futureQ        = self.experiencedModel.model.predict(x = [state2], batch_size = self.batch_size)
            optimalFutureQ = np.zeros(self.batch_size) 
            
            for i in range(self.batch_size):
                
                targetQ[i, int(action[i])] = reward[i]
                if len(action2[i]) == 0:
                    continue

                future_reward  = np.max(futureQ[i][action2[i]])
                targetQ[i, int(action[i])] += self.discount*future_reward
            
            self.experiencedModel.model.fit(x = [state1], y = [targetQ], batch_size = self.batch_size, verbose=None)
            self.update_count += 1
    
    def save(self):
        self.experiencedModel.save_data("models/"+ self.experiencedModel.model_name + "_" + str(self.board_size[0]) + "_" + str(self.board_size[1]))
    
    # Function to display this agent's training history
    def print_train_history(self):
        print("P" + str(self.player), " Training History  -   Ep: " , self.episode_count, "   Up: ", self.update_count)
