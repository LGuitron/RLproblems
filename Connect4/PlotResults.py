import numpy as np
import matplotlib.pyplot as plt

def plot_results(agent):
    episodes            = agent.experiencedModel.episode_list
    rating              = agent.experiencedModel.rating_history
    loss                = agent.experiencedModel.loss_history
    game_length         = agent.experiencedModel.game_length_history
    game_results        = np.sum(agent.experiencedModel.last_game_results, axis=0)
    game_results_labels = 'P1', 'P2', 'T'

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
