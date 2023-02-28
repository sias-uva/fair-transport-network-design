from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
from actor import Encoder

class StateCritic(nn.Module): # ststic+ dynamic + matrix present
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, static_size, dynamic_size, hidden_size, grid_size):
        super(StateCritic, self).__init__()

        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)

        # Define the encoder & decoder models
        self.fc1 = nn.Conv2d(hidden_size * 2, 20, kernel_size=5, stride=1, padding=2)
        self.fc2 = nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2)
        self.fc3 = nn.Linear(20 * grid_size, 1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic, hidden_size, grid_x_size, grid_y_size):

        # Use the probabilities of visiting each
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)

        static_hidden = static_hidden.view(hidden_size, grid_x_size, grid_y_size)
        dynamic_hidden = dynamic_hidden.view(hidden_size, grid_x_size, grid_y_size)

        hidden = torch.cat((static_hidden, dynamic_hidden), 0).unsqueeze(0)

        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        output = output.view(output.size(0), -1)
        output = self.fc3(output)
        # output = self.fc4(output)
        return output
