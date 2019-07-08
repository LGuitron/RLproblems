import numpy as np
import matplotlib.pyplot as plt
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
rand_agent       = RandomAgent()
human_agent      = HumanAgent()

train_episodes          = 1000
test_episodes           = 2
test_train_epochs       = 5
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
        print("Training Loss: ", '{0:.3f}'.format(dqn_agent_1.experiencedModel.last_loss) )
        elapsed_time = time.time() - current_time
        print("Elapsed Time: ", '{0:.3f}'.format(elapsed_time) )
        current_time = time.time()
        
        # Store current information for plots
        dqn_agent_1.experiencedModel.episode_list.append(dqn_agent_1.experiencedModel.games_trained)
        dqn_agent_1.experiencedModel.rating_history.append(new_rating)
        dqn_agent_1.experiencedModel.loss_history.append(dqn_agent_1.experiencedModel.last_loss)
        dqn_agent_1.experiencedModel.game_length_history.append(avg_moves_test)
        dqn_agent_1.experiencedModel.last_game_results[dqn_agent_1.experiencedModel.last_game_index] = test_stats[0:3]
        dqn_agent_1.experiencedModel.last_game_index = (dqn_agent_1.experiencedModel.last_game_index + 1) % 100

dqn_agent_1.save()

episodes            = dqn_agent_1.experiencedModel.episode_list
rating              = dqn_agent_1.experiencedModel.rating_history
loss                = dqn_agent_1.experiencedModel.loss_history
game_length         = dqn_agent_1.experiencedModel.game_length_history
game_results        = np.sum(dqn_agent_1.experiencedModel.last_game_results, axis=0)
game_results_labels = 'P1', 'P2', 'T'
print(game_results)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

ax1.set_title('Agent Strength')
ax2.set_title('Training Loss')
ax3.set_title('Game Length')
ax4.set_title('Game Results')

ax1.plot(episodes, rating)
ax2.plot(episodes, loss)
ax3.plot(episodes, game_length)
ax4.pie(game_results, explode=None, labels=game_results_labels)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()

#rendered_games(dqn_agent_1, human_agent, board_size, connect_to_win)

