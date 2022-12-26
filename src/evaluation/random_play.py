import numpy as np
import torch

import src.configs
from src.agent import Dinosaur
from src.cnn_model import NeuralNetwork
from src.game import ChromeDinoGame
from src.interaction import GameState
import pandas as pd
from src.configs import PathVariables
import os

src.configs.PHASE = 'test'
score_path = PathVariables.RECORD_SAVE_PATH + "/random_play_records/random_play_score.csv"
scores_df = pd.read_csv(score_path) if os.path.isfile(score_path) else pd.DataFrame(columns=['scores'])


def play_randomly(model):
    # instantiate game
    game = ChromeDinoGame()
    dino = Dinosaur(game)
    game_state = GameState(dino, game)

    iteration = 0
    while iteration < model.number_of_iterations:
        # initial action is do nothing
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        imdex = np.random.choice([0, 1], size=1, p=[.5, .5])[0]
        action[imdex] = 1
        _, _, _ = game_state.get_state(action)  # image_data, reward, terminal
        scores_df.loc[iteration] = game_state.score()
        iteration += 1

        if iteration % 10 == 0:
            scores_df.to_csv(score_path, index=False)


def main():
    # We need NeuralNetwork() for only some parameters.
    # The CNN is not used for this case.
    model = NeuralNetwork()
    play_randomly(model)


if __name__ == "__main__":
    main()
