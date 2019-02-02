class HumanAgent:

    def play(self, state, actions, reward):
        
        column = int(input("Please enter your sales from day {}: ".format(actions)))
        while column not in actions:
            column = int(input("Invalid move, select a value from the action space {}: ".format(actions)))
        return column
