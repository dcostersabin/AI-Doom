# importing the libraries

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Importing the packages for openai and doom
import gym
from gym.wrappers import SkipWrapper
import gym_pull
import experience_replay, image_preprocessing

gym_pull.pull('github.com/ppaquette/gym-doom')


#  CNN

class CNN(nn.Module):

    def __init__(self, number_action):
        super(CNN, self).__init__()
        self.convolution1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.convolution2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.convolution3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        self.fc1 = nn.Linear(in_features=self.count_neurons((1, 80, 80)), out_features=40)
        self.fc2 = nn.Linear(in_features=40, out_features=number_action)

    def count_neurons(self, image_dim):
        x = Variable(torch.rand(1, *image_dim))
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        return x.data.view(1, -1).size(1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SoftmaxBody(nn.Module):

    def __init__(self, temp):
        super(SoftmaxBody, self).__init__()
        self.temp = temp

    def forward(self, outputs):
        prob = F.softmax(outputs * self.temp)
        actions = prob.multinomial()
        return actions


class AI:

    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def __call__(self, inputs):
        inp = Variable(torch.from_numpy(np.array(inputs, dtype=np.float32)))
        output = self.brain(inp)
        actions = self.body(output)
        return actions.data.numpy()


doom_env = image_preprocessing.PreprocessImage(
    SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width=80, height=80, grayscale=True)
doom_env = gym.wrappers.Monitor(doom_env, "videos", force=True)
