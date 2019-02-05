import numpy as np

# Method for transforming a given board into one ot vector representation
# First image  = binary features of player's pieces
# Second image = binary features of opponent's pieces
def transform_board(board_size, state, turn):
    
    # Set move value depending on the current player turn
    move_value = 1
    if turn == -1:
        move_value = -1
        
    one_hot_board = np.zeros((1, board_size[0], board_size[1], 2))
    one_hot_board[:,:,:,0] = move_value    == state
    one_hot_board[:,:,:,1] = -1*move_value == state
    return one_hot_board
