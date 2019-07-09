from Game import sim_games, rendered_games
from CalculateRating import *
from DQNAgent import DQNAgent
from copy import deepcopy
import numpy as np
import time

# Function for training agent against itself
def train_DQN_agent(agent, train_episodes, test_episodes, train_test_epochs, board_size = (6, 7), connect_to_win = 4, display_stats_frequency = 1000):

    train_stats             = np.zeros(5)
    test_stats              = np.zeros(5)
    avg_moves_test          = 0
    avg_moves_train         = 0

    # Agent plays against itself and recalculates its rating frequently
    current_time = time.time()
    for i in range(train_test_epochs):

        avg_moves_train, train_stats = sim_games(agent, agent, board_size, connect_to_win, episodes=train_episodes, display_results = False)

        # Display results only after certain amount of games
        if (agent.experiencedModel.games_trained) % display_stats_frequency == 0:
            print("===========================================================================")
            avg_moves_test, test_stats = sim_games(agent, agent, board_size, connect_to_win, episodes=test_episodes, doTraining=False, epsilon_greedy=False, display_results = True)
        else:
            avg_moves_test, test_stats = sim_games(agent, agent, board_size, connect_to_win, episodes=test_episodes, doTraining=False, epsilon_greedy=False, display_results = False)

        if (agent.experiencedModel.games_trained) % display_stats_frequency == 0:

            print("Training Games: ", agent.experiencedModel.games_trained)
            print("Training Loss: ", '{0:.6f}'.format(agent.experiencedModel.last_loss) )
            elapsed_time = time.time() - current_time
            print("Elapsed Time: ", '{0:.3f}'.format(elapsed_time) )
            current_time = time.time()

            # Store current information for plots
            agent.experiencedModel.episode_list.append(agent.experiencedModel.games_trained)
            agent.experiencedModel.loss_history.append(agent.experiencedModel.last_loss)
            agent.experiencedModel.game_length_history.append(avg_moves_test)
            agent.experiencedModel.last_game_results[agent.experiencedModel.last_game_index] = test_stats[0:3]
            agent.experiencedModel.last_game_index = (agent.experiencedModel.last_game_index + 1) % 100

    agent.save()
