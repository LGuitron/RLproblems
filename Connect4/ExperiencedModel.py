import pickle
import random
import numpy as np
from copy import deepcopy
from keras.models import load_model
'''

Class for storing a Keras Model together with its Experience Memory

'''
class ExperiencedModel:

    def __init__(self, model, model_name, exp_size, experience = [], exp_index = 0, rating=0, games_trained=0):
    
        self.rating        = rating
        self.model         = model
        self.model_name    = model_name
        self.exp_size      = exp_size
        self.exp_index     = exp_index
        self.experience    = experience
        self.games_trained = games_trained
    
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
            pickle.dump([self.exp_size, self.exp_index, self.experience, self.model_name, self.rating, self.games_trained], f)
        self.model.save(save_path + ".h5")
        

    # Load instance values from text files
    def load_data(load_path):
        try:
            with open(load_path + ".pkl", "rb") as f:
                exp_size, exp_index, experience, model_name, rating, games_trained = pickle.load(f)
        except:
            exp_size, exp_index, experience, model_name, rating, games_trained = 0, 0, [], "", 0, 0

        model = load_model(load_path + ".h5")
        experiencedModel = ExperiencedModel(model, model_name, exp_size, experience, exp_index, rating, games_trained)
        return experiencedModel
        
