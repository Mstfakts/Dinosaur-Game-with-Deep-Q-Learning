import torch.nn as nn

from src.configs import DISPLAY_LAYER_OUTPUT
from src.util_methods import show_conv_outputs


class NeuralNetwork(nn.Module):
    """
    This class builds the CNN model for the game.
    Also determines some variables.
    """

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.number_of_actions = 2  # [Do nothing, Jump]
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 5000000
        self.replay_memory_size = 10000
        self.minibatch_size = 32

        self.conv1 = nn.Sequential(nn.Conv2d(4, 32, kernel_size=(8, 8), stride=(4, 4)), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)), nn.ReLU(inplace=True))
        self.fc4 = nn.Sequential(nn.Linear(7 * 7 * 64, 512), nn.ReLU(inplace=True))
        self.fc5 = nn.Linear(512, self.number_of_actions)

        self._initialize_weights()

    def forward(self, x):
        c1_out = self.conv1(x)
        c2_out = self.conv2(c1_out)
        c3_out = self.conv3(c2_out)
        out = c3_out.view(c3_out.size(0), -1)
        out = self.fc4(out)
        out = self.fc5(out)

        if DISPLAY_LAYER_OUTPUT:
            im1 = c1_out.permute(2, 3, 1, 0).detach().numpy()
            im2 = c2_out.permute(2, 3, 1, 0).detach().numpy()
            im3 = c3_out.permute(2, 3, 1, 0).detach().numpy()
            display = show_conv_outputs()
            display.__next__()
            display.send((im1, im2, im3))

        """
        # To display the outputs of the layers:
        import matplotlib.pyplot as plt
        im = out.permute(2, 3, 1, 0).detach().numpy()
        plt.imshow(im[:, :, 0, 0])
        plt.show()
        """
        """
        # To display the filters:
        im = self.conv1.weight.permute(2, 3, 1, 0).detach().numpy()
        plt.imshow(im[:, :, 0, 0])
        plt.show()
        """
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                m.bias.data.fill_(0.01)
