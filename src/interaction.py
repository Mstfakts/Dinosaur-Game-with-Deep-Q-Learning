import numpy as np
import torch

from src.configs import PHASE, DISPLAY_GAME_SCREEN
from src.util_methods import screen_shot_game, image_to_tensor, show_img


class GameState:

    def __init__(self, agent, game):
        """
        This class is for adjusting whether game is over or not.
        Also, responsible to determine the reward for the current state.

        :param agent: 'Dinosaur'
        :param game: 'ChromeDinoGame'
        """
        self._agent = agent
        self._game = game
        self._display = show_img()
        self._display.__next__()
        self.__win_reward = 0.1
        self.__die_reward = -1
        self.__score = 0

    def score(self):
        return self.__score

    def get_state(self, actions):

        if PHASE == 'train':
            from main import actions_df
            actions_df.loc[len(actions_df)] = actions[1].tolist()

        score = self._game.get_score()
        self.__score = score
        reward = self.__win_reward
        is_over = False

        if actions[1] == 1:
            self._agent.jump()

        image = screen_shot_game(self._game._driver)

        if DISPLAY_GAME_SCREEN:
            self._display.send(image)

        image = image_to_tensor(image)

        if self._agent.is_crashed():

            if PHASE == 'train':
                from main import loss_df, scores_df
                scores_df.loc[len(loss_df)] = score

            self._game.restart()
            reward = self.__die_reward
            is_over = True

        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)
        return image, reward, is_over
