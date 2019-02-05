import pickle
from copy import deepcopy
from keras.models import load_model
'''

Class for storing a Keras Model together with its Experience Memory

'''
class ExperiencedModel:

    def __init__(self, model, model_name, exp_size, experience = [], exp_index = 0):
    
        self.model        = model
        self.model_name   = model_name
        self.exp_size     = exp_size
        self.exp_index    = exp_index
        self.experience   = experience

    
    # Add a transition into experience memory
    def add_experience(self, transition):
        
        if len(self.experience) < self.exp_size:
            self.experience.append(deepcopy(transition))
        else:
            self.experience[self.exp_index] = deepcopy(transition)
            self.exp_index = (self.exp_index + 1) % self.exp_size
    
    
    # Save instance values to text files
    def save_data(self, save_path):
        with open(save_path + ".pkl", "wb") as f:
            pickle.dump([self.exp_size, self.exp_index, self.experience, self.model_name], f)
        self.model.save(save_path + ".h5")
        

    # Load instance values from text files
    def load_data(load_path):
        try:
            with open(load_path + ".pkl", "rb") as f:
                exp_size, exp_index, experience, model_name = pickle.load(f)
        except:
            exp_size, exp_index, experience, model_name = 0, 0, [], ""

        model = load_model(load_path + ".h5")
        experiencedModel = ExperiencedModel(model, model_name, exp_size, experience, exp_index)

        return experiencedModel
        
