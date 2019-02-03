#Class used for storing transitions for later performing mini batch gradient descent
class Transition:
    def __init__(self, state1, action, reward, state2, action2):
        self.state1       = state1
        self.action       = action
        self.reward       = reward
        self.state2       = state2
        self.action2      = action2
