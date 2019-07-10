import pickle
import random
import numpy as np
import numpy as np
from copy import deepcopy
from AgentType import AgentType
from keras.models import load_model


'''

Class for storing a Keras Model together with its Experience Memory

'''
class ExperiencedModel:

    def __init__(self, model, model_name, agent_type, exploration_params, exploration_decay, exploration, exp_size, experience = [], exp_index = 0, games_trained=0, episode_list=[], last_loss=0, loss_history=[], game_length_history=[], last_game_results=np.zeros((100, 3)), last_game_index=0):

        self.model              = model
        self.model_name         = model_name
        self.agent_type         = agent_type
        self.exploration_params = exploration_params     # Epsilon or Tempterature depending on agent type
        self.exploration_decay  = exploration_decay      # Episode at which the next decay will occur (decays are linear)
        self.exploration        = exploration            # Current value of the exploration parameter

        self.exp_size      = exp_size
        self.exp_index     = exp_index
        self.experience    = experience
        self.games_trained = games_trained

        # Store episode list for later plotting
        self.episode_list   = episode_list

        # Store lose progression for plotting
        self.last_loss      = last_loss
        self.loss_history   = loss_history

        # Store average game length for later plotting
        self.game_length_history = game_length_history

        # Store game results from last 100 games
        self.last_game_results  = last_game_results
        self.last_game_index    = last_game_index       # Ranges from 0 to 99

    # Add a transition into experience memory
    def add_experience(self, transition):

        if len(self.experience) < self.exp_size:
            self.experience.append(deepcopy(transition))
        else:
            self.experience[self.exp_index] = deepcopy(transition)
            self.exp_index = (self.exp_index + 1) % self.exp_size


    # Get samples from experience list
    def get_samples(self, n, board_size):

        transitions  = random.sample(self.experience, k=n)
        state1       = np.zeros((n,board_size[0], board_size[1], 2))
        action       = np.zeros(n)
        reward       = np.zeros(n)
        state2       = np.zeros((n,board_size[0], board_size[1], 2))
        action2      = []                                                                       # Actions available from state2

        for i , transition in enumerate(transitions):
            state1[i] = transition.state1
            action[i] = transition.action
            reward[i] = transition.reward
            state2[i] = transition.state2
            action2.append(transition.action2)
        return state1, action, reward, state2, action2

    # Save instance values to text files
    def save_data(self, save_path):
        with open(save_path + ".pkl", "wb") as f:
            pickle.dump([self.agent_type, self.exploration_params, self.exploration_decay, self.exploration, self.exp_size, self.exp_index, self.experience, self.model_name, self.games_trained, self.episode_list, self.last_loss, self.loss_history, self.game_length_history, self.last_game_results, self.last_game_index], f)
        self.model.save(save_path + ".h5")


    # Load instance values from text files
    def load_data(load_path):
        try:
            with open(load_path + ".pkl", "rb") as f:
                agent_type, exploration_params, exploration_decay, exploration, exp_size, exp_index, experience, model_name, games_trained, episode_list, last_loss, loss_history, game_length_history, last_game_results, last_game_index = pickle.load(f)

        except:
            agent_type, exploration_params, exploration_decay, exploration, exp_size, exp_index, experience, model_name, games_trained, episode_list, last_loss, loss_history, game_length_history, last_game_results, last_game_index = AgentType.EGreedy, [1.0, 1.0, 0.1], [100000, 1000000], 0 ,0, 0, [], "", 0, [], 0, [], [], np.zeros((100, 3)), 0

        model = load_model(load_path + ".h5")
        experiencedModel = ExperiencedModel(model, model_name, agent_type, exploration_params, exploration_decay, exploration, exp_size, experience, exp_index, games_trained, episode_list, last_loss, loss_history, game_length_history, last_game_results, last_game_index)
        return experiencedModel

    # Helper function to decay exploration parameter
    def decay_exploration(self):
        decay_index = 0
        for i in range(len(self.exploration_decay)):
            if self.games_trained < self.exploration_decay[i]:
                break
            decay_index += 1

        # Decay only if not in the minimum exploration value
        if decay_index + 1 < len(self.exploration_params):
            numerator = self.exploration_params[decay_index] - self.exploration_params[decay_index+1]
            if decay_index == 0:
                denominator = self.exploration_decay[0]
            else:
                denominator = self.exploration_decay[decay_index] - self.exploration_decay[decay_index - 1]
            decay_value = numerator/(2*denominator)
            self.exploration -= decay_value
