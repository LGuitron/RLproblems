import random

class RandomAgent:

    def play(self, state, actions, reward):
        return random.choice(actions)
    
    def receive_last_reward(self, reward):
        pass
        #print("Reward Random: " , reward)
