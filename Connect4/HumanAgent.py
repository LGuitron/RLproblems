class HumanAgent:

    def play(self, state, actions, reward):
        
        column = int(input("Enter column index to play {}: ".format(actions)))
        while column not in actions:
            column = int(input("Invalid move, select a value from the action space {}: ".format(actions)))
        return column

    def receive_last_reward(self, final_state, reward):
        pass
