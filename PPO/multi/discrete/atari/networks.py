import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class AtariActorCritic(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1=512,
                 chkpt_dir='models/'):
        super(AtariActorCritic, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'atari_ppo')

        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)
        self.fc1 = nn.Linear(fc_input_dims, fc1)
        self.pi = nn.Linear(fc1, n_actions)
        self.v = nn.Linear(fc1, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv_state = conv3.view(conv3.size()[0], -1)
        # conv_state shape is BS x (n_filters * H * W)
        flat1 = F.relu(self.fc1(conv_state))
        pi = self.pi(flat1)
        v = self.v(flat1)

        return pi, v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
