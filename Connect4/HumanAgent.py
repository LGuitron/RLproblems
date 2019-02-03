class HumanAgent:

    def play(self, state, actions, reward):
        
        column = int(input("Please enter an action {}: ".format(actions)))
        while column not in actions:
            column = int(input("Invalid move, select a value from the action space {}: ".format(actions)))
        return column
