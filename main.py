import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agent import Dinosaur
from src.cnn_model import NeuralNetwork
from src.configs import PathVariables, initialize_recorders, PHASE, SAVE_RECORDS_AT_ITERATION
from src.game import ChromeDinoGame
from src.interaction import GameState
from src.util_methods import save_obj

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
loss_df, scores_df, actions_df, q_values_df = initialize_recorders()


def run(model, start, pretrained):
    """
    1 - It is initial steps until the while-loop
    2 - Very first part of the while loop is 'Forward Pass Section'
    3 - After the forward pass, the 'Experience Replay Section' starts for the 'stable target'
    4 - Whenever the target is also found, 'Loss Calculation Section' starts
    5 - Regarding the loss, the parameters are updated in 'Backward Pass Section'
    6, 7 - Finally, the loop adjustments are completed at the very last part of the loop

    :param model:
    :param start:
    :param pretrained:
    :return:
    """

    # ------ Initial Adjustments Section (1) STARTS ------
    # define Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    # initialize mean squared error loss
    criterion = nn.MSELoss()

    # instantiate game
    game = ChromeDinoGame()
    dino = Dinosaur(game)
    game_state = GameState(dino, game)

    # initialize replay memory
    replay_memory = []

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, _, _ = game_state.get_state(action)  # image_Data, reward, terminal
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)  # stacking 4 images

    # initialize epsilon value
    epsilon = model.initial_epsilon
    iteration = 0
    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)
    # ------ Initial Adjustments Section (1) ENDS ------

    while iteration < model.number_of_iterations:

        # ------ Forward Pass Section (2) STARTS ------
        output = model(state)[0]  # Get the q-value from the CNN.

        # Initialize action again, since it is in a loop.
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)

        # Adjust the 'exploration'
        random_action = random.random() <= epsilon if PHASE == 'train' else False
        if random_action:
            action_index = torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
            print(" -- Random Action! -- ")
        else:
            action_index = torch.argmax(output)
        action[action_index] = 1

        # Get the next state and reward, regarding the action
        image_data_1, reward, terminal = game_state.get_state(action)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)
        action = action.unsqueeze(0)
        # ------ Forward Pass Section (2) ENDS ------

        # ------ Experience Replay Section (3) STARTS ------
        # Experience replay application.
        replay_memory.append((state, action, reward, state_1, terminal))

        # If replay memory is full, remove the oldest transition
        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)
            if len(replay_memory) == model.replay_memory_size + 1:
                print(f"Replay memory reached to its max capacity: {model.replay_memory_size}")

        # sample random minibatch
        minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

        # unpack minibatch
        state_batch = torch.cat(tuple(mb[0] for mb in minibatch))
        action_batch = torch.cat(tuple(mb[1] for mb in minibatch))
        reward_batch = torch.cat(tuple(mb[2] for mb in minibatch))
        state_1_batch = torch.cat(tuple(mb[3] for mb in minibatch))
        terminal_batch = [mb[4] for mb in minibatch]

        # get output for the next state
        output_1_batch = model(state_1_batch)

        y_batch = list()
        for i in range(len(minibatch)):
            if terminal_batch[i]:
                # For Terminal state (if terminal is true), set y_j = r_j
                y_batch.append(reward_batch[i])
            else:
                # For Non-Terminal state, y_j = r_j + gamma * max(batch_on_action)
                y_batch.append(reward_batch[i] + model.gamma * torch.max(output_1_batch[i]))
        y_batch = torch.cat(tuple(y_batch))
        # ------ Experience Replay Section (3) ENDS ------

        # ------ Loss Calculation Section (4) STARTS ------
        pred = model(state_batch)  # Get the q-value from the CNN.
        q_value = torch.sum(pred * action_batch, dim=1)

        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        optimizer.zero_grad()

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()

        # Loss calculation
        loss = criterion(q_value, y_batch)
        # ------ Loss Calculation Section (4) ENDS ------

        # ------ Backward Pass Section (5) STARTS ------
        loss.backward()
        optimizer.step()
        # ------ Backward Pass Section (5) ENDS ------

        # ------ Loop Update Section (6) STARTS ------
        state = state_1  # Update state (Upcoming state is now the present state)
        epsilon = epsilon_decrements[iteration]  # epsilon annealing
        iteration += 1
        # ------ Loop Update Section (6) ENDS ------

        # ------ Parameters Save Section (7) STARTS ------
        if PHASE == 'train':

            print(f"iteration: {iteration}, "
                  f"elapsed time: {time.time() - start}, "
                  f"epsilon: {epsilon}, "
                  f"action: {action_index.cpu().detach().numpy()}, "
                  f"reward: {reward.numpy()[0][0]}, "
                  f"Q max: {np.max(output.cpu().detach().numpy())}")

            if iteration % SAVE_RECORDS_AT_ITERATION == 0:
                loss_df.loc[len(loss_df)] = loss.tolist()
                q_values_df.loc[len(q_values_df)] = np.max(pred.tolist())

                # Save the model
                torch.save(model,
                           PathVariables.MODELS_SAVE_PATH + "/current_model_" + str(iteration + pretrained) + ".pth")

                # Save other parameters
                save_obj(replay_memory, "replay_memory")  # saving episodes
                save_obj(iteration, "iteration")  # caching time steps
                save_obj(epsilon, "epsilon")  # cache epsilon to avoid repeated randomness in actions
                loss_df.to_csv(PathVariables.LOSS_FILE, index=False)
                scores_df.to_csv(PathVariables.SCORES_FILE, index=False)
                actions_df.to_csv(PathVariables.ACTION_FILE, index=False)
                q_values_df.to_csv(PathVariables.Q_VALUE_FILE, index=False)
        # ------ Parameters Save Section (7) ENDS ------


def main(pretrained=None):
    model = NeuralNetwork()
    model = torch.load(PathVariables.MODELS_SAVE_PATH + '/current_model_' + str(pretrained) + '.pth')
    run(model, time.time(), pretrained)


if __name__ == "__main__":
    main(pretrained=1635000)
