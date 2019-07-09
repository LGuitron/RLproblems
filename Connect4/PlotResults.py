import numpy as np
import matplotlib.pyplot as plt

def plot_results(agent):
    episodes            = agent.experiencedModel.episode_list
    loss                = agent.experiencedModel.loss_history
    game_length         = agent.experiencedModel.game_length_history
    game_results        = np.sum(agent.experiencedModel.last_game_results, axis=0)
    game_results_labels = 'P1', 'P2', 'T'

    grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.3)

    ax1 = plt.subplot(grid[0, 0])
    ax2 = plt.subplot(grid[0, 1])
    ax3 = plt.subplot(grid[1, :])

    ax1.set_title('Training Loss')
    ax2.set_title('Game Length')
    ax3.set_title('Game Results')
    
    ax1.plot(episodes, loss)
    ax2.plot(episodes, game_length)
    ax3.pie(game_results, explode=None, labels=game_results_labels)
    
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()
