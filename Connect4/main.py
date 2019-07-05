import numpy as np
from KerasModels import *
from DQNAgent import DQNAgent
from HumanAgent import HumanAgent
from RandomAgent import RandomAgent
from Game import sim_games, rendered_games
from CalculateRating import calculate_rating
from copy import deepcopy


board_size      = (6,7)
connect_to_win  = 4

model1, model1_name            = compile_model1(board_size)
model2, model2_name            = compile_model2(board_size)
model_simple, model_simle_name = compile_model_simple(board_size)

dqn_agent_1      = DQNAgent(board_size, "models/dqn_model1_6_7", model1, model1_name)
#dqn_agent_2      = DQNAgent(board_size, "models/dqn_model2_6_7", model2, model2_name)
rand_agent       = RandomAgent()
human_agent      = HumanAgent()

train_episodes    = 1000
test_episodes     = 1000
test_train_epochs = 99
train_stats       = np.zeros(5)
test_stats        = np.zeros(5)
avg_moves_test    = 0
avg_moves_train   = 0

# Calibrate starting rating using with the Random Agent
# Random Agent has a rating of 0
if dqn_agent_1.experiencedModel.games_trained == 0:
    avg_moves_test, test_stats = sim_games(dqn_agent_1, rand_agent, board_size, connect_to_win, episodes=test_episodes, doTraining=False, epsilon_greedy=True)

    old_rating      = 0
    new_agent_score = (test_stats[3] + 0.5*test_stats[2])/test_episodes
    new_rating      = calculate_rating(old_rating, new_agent_score)
    
    dqn_agent_1.experiencedModel.rating = new_rating

    print("Starting Elo Rating: ", int(new_rating))
    print("Training games: ", dqn_agent_1.experiencedModel.games_trained)

# Agent plays against itself and recalculates its rating frequently
for i in range(test_train_epochs):
    old_dqn_agent_1 = deepcopy(dqn_agent_1)
    avg_moves_train, train_stats = sim_games(dqn_agent_1, dqn_agent_1, board_size, connect_to_win, episodes=train_episodes)
    avg_moves_test, test_stats = sim_games(dqn_agent_1, old_dqn_agent_1, board_size, connect_to_win, episodes=test_episodes, doTraining=False, epsilon_greedy=True)
    
    old_rating      = old_dqn_agent_1.experiencedModel.rating
    new_agent_score = (test_stats[3] + 0.5*test_stats[2])/test_episodes
    new_rating      = calculate_rating(old_rating, new_agent_score)
    
    dqn_agent_1.experiencedModel.rating = new_rating
    print("Elo Rating: ", int(new_rating))
    print("Training games: ", dqn_agent_1.experiencedModel.games_trained)

dqn_agent_1.save()
#rendered_games(dqn_agent_1, human_agent, board_size, connect_to_win)
