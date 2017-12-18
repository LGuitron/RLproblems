# **Q-Learning Tic Tac Toe**

Q-Learning implementation for Tic Tac Toe via self play.
In this implementation a single lookup table is stored for 3^9 states mapped with 9 possible actions. 
Rewards given to the agents are +1 per win, -1 per loss, and -0.1 per step taken.
The behavioral policy is an epsilon-greedy policy, with a default epsilon of 0.05 (exploration probability by making random actions), 
and a default alpha (learning rate) of 0.1. Rewards given to the agent are 

Q-values for the trained policy are saved in "Qvals.npy" file, the program will attempt to load this file at the beginning in order to
continue training from this point, if no file is found with that name a new training will begin.

Training results can be tested by a human player by playing against the trained AI after specified number of episodes.
