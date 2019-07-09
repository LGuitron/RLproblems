import random
import numpy as np
from pathlib import Path
from AgentType import AgentType
from Transition import Transition
from KerasModels import *
from TransformBoard import transform_board
from ExperiencedModel import ExperiencedModel

'''

Deep Q Network Agent that receives current board as input

'''
class DQNAgent:
    
    def __init__(self, board_size, load_path = None, compiled_model = None, model_name="", agent_type=AgentType.EGreedy, exploration_params=[1.0, 1.0, 0.1], exploration_decay=[100000, 1000000]):

        self.board_size = board_size

        # Q Learning parameters
        self.batch_size         = 256
        self.discount           = 0.9
        
        # Update parameters
        self.update_frequency   = 256      # Steps that the agent has to perform before updating its weights
        self.step_count         = 0        # Steps made since last update

        self.is_exploring      = True      # Agent uses exploration strategy when True (Softmax or E-Greedy)
        self.is_training       = True      # Agent updates its model when this is True
        
        # Create new model if path is not specified or if the load_path entered does not exist
        if load_path is None or not (Path(load_path + ".h5").is_file() and Path(load_path + ".pkl").is_file()):
            
            if compiled_model is not None:
                self.experiencedModel  = ExperiencedModel(compiled_model, model_name, agent_type, exploration_params, exploration_decay, exploration_params[0], exp_size = 1000000)
            else:
                print("ERROR, unspecified model for DQN Agent.")
                exit()
        
        # Load experiencedModel from load_path
        else:
            self.experiencedModel = ExperiencedModel.load_data(load_path)

        # Store previous input (state1, action) for P1 and P2 in the current game
        self.prev_state  = [None, None]
        self.prev_action = [None, None] 

    def play(self, state, turn, actions, reward):
        
        state_one_hot = transform_board(self.board_size, state, turn)

        # Follow exploration strategy
        if self.is_exploring:
            
            # E-Greedy Exploration
            if self.experiencedModel.agent_type == AgentType.EGreedy and random.random() < self.experiencedModel.exploration:
                sel_action = random.choice(actions)
        
            # Make best move according to current model
            else:
                q_vals      = self.experiencedModel.model.predict(x = [state_one_hot], batch_size=1)[0]
                max_index   = np.argmax(q_vals[actions])
                sel_action  = actions[max_index]
            
            
        # Make best move according to current model
        else:
            q_vals      = self.experiencedModel.model.predict(x = [state_one_hot], batch_size=1)[0]
            max_index   = np.argmax(q_vals[actions])
            sel_action  = actions[max_index]
            
        # Add transition to experience memory
        if(self.prev_state[turn] is not None):            
            self.experiencedModel.add_experience(Transition(self.prev_state[turn], self.prev_action[turn], reward, state_one_hot, actions))
            self.fit_model()
            
        self.prev_state[turn]  = state_one_hot
        self.prev_action[turn] = sel_action        
        return sel_action
    
    # Receive reward for winning or losing the game
    def receive_last_reward(self, final_state, turn, reward):
        state_one_hot = transform_board(self.board_size, final_state, turn)        
        
        self.experiencedModel.add_experience(Transition(self.prev_state[turn], self.prev_action[turn], reward, state_one_hot, np.zeros(0)))
        self.fit_model()
        
        self.prev_state[turn]  = None
        self.prev_action[turn] = None            
        
    # Function for fitting model
    def fit_model(self):
        # 3 conditions to fit:
        # 1. Update frequency step reached
        # 2. Training mode active
        # 3. Agent experience has at least the size of the batch
        if self.step_count % self.update_frequency == 0 and self.is_training and len(self.experiencedModel.experience) >= self.batch_size:

            # Get samples from experience memory
            state1, action, reward, state2, action2 = self.experiencedModel.get_samples(self.batch_size, self.board_size)

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

            history = self.experiencedModel.model.fit(x = [state1], y = [targetQ], batch_size = self.batch_size, verbose=None)
            
            # Add loss information to list for later plotting
            self.experiencedModel.last_loss = history.history['loss'][0]
        
        # Increase step count
        self.step_count += 1

    def save(self):
        self.experiencedModel.save_data("models/dqn_"+ self.experiencedModel.model_name + "_" + str(self.board_size[0]) + "_" + str(self.board_size[1]))
