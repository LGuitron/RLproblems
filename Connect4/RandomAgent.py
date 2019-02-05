import random

class RandomAgent:

    def play(self, state, turn, actions, reward):
        return random.choice(actions)
    
    def receive_last_reward(self, final_state, turn, reward):
        pass
