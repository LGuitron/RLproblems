#Class used for storing transitions for later performing mini batch gradient descent
class Transition:
    def __init__(self, state1, action, reward, state2, avMoves2):
        self.state1 = state1
        self.action = action
        self.reward = reward
        self.state2 = state2
        self.avMoves2 = avMoves2        #Boolean vector for available moves in state2
