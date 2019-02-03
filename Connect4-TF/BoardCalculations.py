import numpy as np

'''
Function to get resulting boards from playing available actions from the current board
'''
def get_future_boards(state, actions, move_value):

    future_states = np.zeros((len(actions), state.shape[0], state.shape[1]))
    
    if(len(actions) == 0):
        return None
    
    for i, column in enumerate(actions):
        row              = np.argmax(np.argwhere(state[:, int(column)] == 0))
        future_states[i] = state
        future_states[i,row, int(column)] = move_value
    
    return future_states


'''
Function to calculate actions available in a given state
Returns:
np.array with indeces of columns available
'''
#def calculate_action_space(state):
#    action_space = np.argwhere(state[0] == np.zeros(state.shape[1]))[:,0]
