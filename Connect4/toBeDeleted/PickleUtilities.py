from pathlib import Path
import pickle

'''

Functions for storing and recovering experience memory for QAgents

'''

def load_data(filename):
    try:
        with open(filename, "rb") as f:
            episode_count, update_count, exp_index, experience = pickle.load(f)
    except:
        episode_count, update_count, exp_index, experience = 0, 0, 0, []
    return [episode_count, update_count, exp_index, experience]

def save_data(Qagent, filename):
    with open(filename, "wb") as f:
        pickle.dump([Qagent.episode_count, Qagent.update_count, Qagent.exp_index, Qagent.experience], f)
