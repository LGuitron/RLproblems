import numpy as np
from KerasModels import *
from DQNAgent import DQNAgent
from HumanAgent import HumanAgent
from RandomAgent import RandomAgent
from Game import sim_games, rendered_games
from CalculateRating import *
from copy import deepcopy
import time

board_size      = (6,7)
connect_to_win  = 4

model1, model1_name            = compile_model1(board_size)
model2, model2_name            = compile_model2(board_size)
model_simple, model_simle_name = compile_model_simple(board_size)

dqn_agent_1      = DQNAgent(board_size, "models/dqn_model1_6_7", model1, model1_name)
#dqn_agent_2      = DQNAgent(board_size, "models/dqn_model2_6_7", model2, model2_name)
rand_agent       = RandomAgent()
human_agent      = HumanAgent()

train_episodes          = 1000
test_episodes           = 2
test_train_epochs       = 100
display_stats_frequency = 1000              # Display stats after this amount of games
train_stats             = np.zeros(5)
test_stats              = np.zeros(5)
avg_moves_test          = 0
avg_moves_train         = 0


# Agent plays against itself and recalculates its rating frequently
current_time = time.time()
for i in range(test_train_epochs):
    
    # Store old agent network for measuring rating increase
    old_dqn_agent_1 = DQNAgent(board_size, None, model1, model1_name)
    old_dqn_agent_1.experiencedModel.model  = deepcopy(dqn_agent_1.experiencedModel.model)
    old_dqn_agent_1.experiencedModel.rating =  dqn_agent_1.experiencedModel.rating
    
    avg_moves_train, train_stats = sim_games(dqn_agent_1, dqn_agent_1, board_size, connect_to_win, episodes=train_episodes, display_results = False)
    
    # Display results only after certain amount of games
    if (dqn_agent_1.experiencedModel.games_trained) % display_stats_frequency == 0:
        print("===========================================================================")
        avg_moves_test, test_stats = sim_games(dqn_agent_1, old_dqn_agent_1, board_size, connect_to_win, episodes=test_episodes, doTraining=False, epsilon_greedy=False, display_results = True)
    else:
        avg_moves_test, test_stats = sim_games(dqn_agent_1, old_dqn_agent_1, board_size, connect_to_win, episodes=test_episodes, doTraining=False, epsilon_greedy=False, display_results = False)
    
    old_rating      = old_dqn_agent_1.experiencedModel.rating
    new_agent_score = (test_stats[3] + 0.5*test_stats[2])/test_episodes
    new_rating      = calculate_rating_two_games(old_rating, new_agent_score)
    dqn_agent_1.experiencedModel.rating = new_rating
    
    if (dqn_agent_1.experiencedModel.games_trained) % display_stats_frequency == 0:
        
        print("Player Strength: ", int(new_rating))
        print("Training Games: ", dqn_agent_1.experiencedModel.games_trained)
        elapsed_time = time.time() - current_time
        print("Elapsed Time: ", '{0:.3f}'.format(elapsed_time) )
        current_time = time.time()
        
dqn_agent_1.save()
#rendered_games(dqn_agent_1, human_agent, board_size, connect_to_win)
