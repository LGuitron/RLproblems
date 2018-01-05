# **Deep Q-Learning Connect4**

Deep Q-Learning implementation for Connect4 via self play using Pytorch.

This implementation uses a Convolutional Neural Network to estimate the action value function,
as well as experience replay and fixed Q-targets.

The input to the neural network is the raw board representation of the current state from both players' perspective.
A player's perspective image consists on a 6x7 binary feature plain (1 - chip present for current player, 0 - empty space, or chip present for opponent)

The neural network architecture is the following:
Convolution of 64 filters with 3x3 kernel size
Batch Normalization
ReLU activation

10 residual blocks consisting on:
Convolution of 64 filters with 3x3 kernel size
Batch Normalization
ReLU activation
Convolution of 64 filters with 3x3 kernel size
Batch Normalization
Skip connection
ReLU activation

Finally a fully connected layer is used between all 2688 output units from the last residual block, and 7 output neurons which represent possible actions.

Training can be performed with CPU or with GPU, to train with GPU run HumanGame.py with 1 or more console parameters (their value does not matter),
otherwise training will begin with CPU.
After a training session the network's parameters and the experience cache are stored in the project's directory allowing to resume training later from this point.

Training results can be tested by a human player by playing against the trained AI after specified number of episodes.
