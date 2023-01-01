import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import src.configs
from src.agent import Dinosaur
from src.cnn_model import NeuralNetwork
from src.configs import PathVariables
from src.game import ChromeDinoGame
from src.interaction import GameState

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
src.configs.PHASE = 'test'
score_path = PathVariables.RECORD_SAVE_PATH + "/random_play_records/random_play_score.csv"
scores_df = pd.read_csv(score_path) if os.path.isfile(score_path) else pd.DataFrame(columns=['scores'])


def play_randomly(model):
    # instantiate game
    game = ChromeDinoGame()
    dino = Dinosaur(game)
    game_state = GameState(dino, game)

    iteration = 0
    game_played = 0
    while iteration < model.number_of_iterations:
        # initial action is do nothing
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        imdex = np.random.choice([0, 1], size=1, p=[.5, .5])[0]
        if imdex == 1:
            print('jump')
        else:
            print('nothing')
        action[imdex] = 1
        _, _, is_over = game_state.get_state(action)  # image_data, reward, terminal
        if is_over:
            scores_df.loc[game_played] = game_state.score()
            game_played += 1
        iteration += 1

        if game_played % 10 == 0:
            scores_df.to_csv(score_path, index=False)

        time.sleep(0.5)  # Sleep for letting the agent take an action. Action should be completed.


def main():
    # We need NeuralNetwork() for only some parameters.
    # The CNN is not used for this case.
    model = NeuralNetwork()
    play_randomly(model)


def evaluate():
    start = 0
    interval = 10
    step = 10
    scores_df_ = pd.read_csv(score_path)
    mean_scores = pd.DataFrame(columns=['score'])
    max_scores = pd.DataFrame(columns=['max_score'])

    while interval <= len(scores_df_):
        mean_scores.loc[len(mean_scores)] = (scores_df_.loc[start:interval].mean()['scores'])
        max_scores.loc[len(max_scores)] = (scores_df_.loc[start:interval].max()['scores'])
        start = interval
        interval = interval + step

    mean_scores.plot()
    plt.show()
    max_scores.plot()
    plt.show()
    print("len(mean_scores)", len(mean_scores))
    print("len(max_scores)", len(max_scores))


if __name__ == "__main__":
    main()
    evaluate()
