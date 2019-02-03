import numpy as np
class Environment:

    '''
    Input:
    tuple board_size   - board's dimensions
    connections_to_win - number of chips to connect in any direction to win
    '''
    def __init__(self , p1, p2, board_size = (6,7), connections_to_win=4):
        
        # Board parameters
        self.board_size         = board_size
        self.connections_to_win = connections_to_win
        self.players            = (p1, p2)
        
        # Environment parameters
        self.win_reward         = 20
        self.tie_reward         = 0
        self.wait_reward        = -1
        
        # True for P1 turn
        self.turn               = 0                                                       # 0 for P1, 1 for P2                  
        self.board              = np.zeros((self.board_size[0], self.board_size[1]))      #0 = Empty, #1 = P1 move, #-1 = P2 move 
        self.action_space       = None                                                    # Indices of columns where moves can be made
        self.update_action_space()
        
        # 0 for P1 win
        # 1 for P2 win
        # 2 for tie
        self.winner             = -1
        self.moves_made         = 0

    # Update action space and return its value
    def update_action_space(self):
        self.action_space = np.argwhere(self.board[0] == np.zeros(self.board_size[1]))[:,0]
        
    '''
    Request action from the current player
    
    Output:
    Same as calc_reward()
    '''
    def request_action(self):
        
        # Check if opponent won the game on its last turn, in that case return done=True
        #if 
        
        # If there are no more actions available the game ended in a tie
        if len(self.action_space) == 0:
            self.players[self.turn].receive_last_reward(self.board, self.tie_reward)
            self.players[1-self.turn].receive_last_reward(self.board, self.tie_reward)
            self.winner = 2
            return True
        
        column = self.players[self.turn].play(self.board, self.action_space, self.wait_reward)
        return self.update_board(column)
        

    '''
    Update the board after a move has been made
    
    Output:
    Same as calc_reward()
    '''
    def update_board(self, column):
        row = np.argmax(np.argwhere(self.board[:, column] == 0))
        
        move_value = 1
        if self.turn == 1:
            move_value = -1
            
        self.board[row, column] = move_value
        self.moves_made        += 1
        return self.calc_reward(row, column, move_value)
        
    '''
    Returns:
    bool game finished
    '''
    def calc_reward(self, row, column, move_value):        
        
        # Directions from current chip to:
        # Up
        # Up right
        # Right
        # Down right
        directions = [(1,0), (1,1), (0,1), (-1,1)]

        for direction in directions:
            connected_count = 1             # Number of chips connected to the current one in the current direction

            # Check connected count in this direction
            y = row + direction[0]
            x = column + direction[1]
            while 0 <= y and y < self.board_size[0] and 0 <= x and x < self.board_size[1]:
                if self.board[y,x] == move_value:
                    connected_count += 1
                    y += direction[0]
                    x += direction[1]
                else:
                    break
            
            # Check connected count in opposite direction
            y = row - direction[0]
            x = column - direction[1]
            while 0 <= y and y < self.board_size[0] and 0 <= x and x < self.board_size[1]:
                if self.board[y,x] == move_value:
                    connected_count += 1
                    y -= direction[0]
                    x -= direction[1]
                else:
                    break
            
            # Game ended (Give final reward to both players)
            if connected_count >= self.connections_to_win:
                self.players[self.turn].receive_last_reward(self.board, self.win_reward)
                self.players[1-self.turn].receive_last_reward(self.board, -1*self.win_reward)
                self.winner = self.turn
                return True
        
        # If game did not end go to next turn
        self.next_turn()
        return False
    
        
    # Change the turn if game is not over yet
    def next_turn(self):
        self.update_action_space()
        self.turn = 1 - self.turn
        
    # Funtion to print a good looking board
    def print_board(self):
        
        board_string = ""
        for i in range(self.board.shape[0]):        # Iterate Rows
            board_string += "["
            for j in range(self.board.shape[1]):    # Iterate Columns
                if self.board[i][j] == 1:
                    char = ' X '
                elif self.board[i][j] == -1:
                    char = ' O '
                else:
                    char = ' - '
                board_string += char
            board_string += "]\n"
        
        print(board_string)
        
        
        
