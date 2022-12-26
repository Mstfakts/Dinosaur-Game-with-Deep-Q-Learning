import os

import pandas as pd

"""
This file is for storing some constants and parameters.
"""

PHASE = 'test'
DISPLAY_GAME_SCREEN = True
DISPLAY_LAYER_OUTPUT = True
SAVE_RECORDS_AT_ITERATION = 1000


class GameImage:
    HEIGHT = 84
    WIDTH = 84


class DriverVariables:
    GAME_URL = "chrome://dino"  # URL address for the game.
    CHROME_DRIVER_PATH = "../chromedriver"  # Chrome driver path
    GETBASE64SCRIPT = "canvasRunner = document.getElementById('runner-canvas'); " \
                      "return canvasRunner.toDataURL().substring(22)"  # To get image from the canvas


class PathVariables:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_SAVE_PATH = ROOT_DIR + "/saved_models"
    RECORD_SAVE_PATH = ROOT_DIR + "/records"
    LOSS_FILE = RECORD_SAVE_PATH + "/loss_df.csv"
    ACTION_FILE = RECORD_SAVE_PATH + "/actions_df.csv"
    Q_VALUE_FILE = RECORD_SAVE_PATH + "/q_values.csv"
    SCORES_FILE = RECORD_SAVE_PATH + "/scores_df.csv"


def initialize_recorders():
    loss_df = pd.read_csv(PathVariables.LOSS_FILE) \
        if os.path.isfile(PathVariables.LOSS_FILE) else pd.DataFrame(columns=['loss'])
    scores_df = pd.read_csv(PathVariables.SCORES_FILE) \
        if os.path.isfile(PathVariables.SCORES_FILE) else pd.DataFrame(columns=['scores'])
    actions_df = pd.read_csv(PathVariables.ACTION_FILE) \
        if os.path.isfile(PathVariables.ACTION_FILE) else pd.DataFrame(columns=['actions'])
    q_values_df = pd.read_csv(PathVariables.Q_VALUE_FILE) \
        if os.path.isfile(PathVariables.Q_VALUE_FILE) else pd.DataFrame(columns=['qvalues'])

    return loss_df, scores_df, actions_df, q_values_df
