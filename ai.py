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
number_action = doom_env.action_space.n

# building an ai
cnn = CNN(number_action)
softmax_body = SoftmaxBody(temp=1.0)
ai = AI(brain=cnn, body=softmax_body)

# setting up Experience Replay
n_steps = experience_replay.NStepProgress(doom_env, ai, 10)
memory = experience_replay.ReplayMemory(n_steps=n_steps, capacity=10000)


# implementing eligibility trace
def eligibility_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch:
        inp = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype=np.float32)))
        output = cnn(inp)
        cumulative_reward = 0.0 if series[-1].done else output[1].data.max
        for step in reversed(series[:-1]):
            cumulative_reward = step.reward + gamma * cumulative_reward
        state = series[0].state
        target = output[0].data
        target[series[0].action] = cumulative_reward
        inputs.append(state)
        targets.append(target)
    return torch.from_numpy(np.array(inputs, dtype=np.float32)), torch.stack(targets)


# making the move acerage on 100 steps
class MA:

    def __init__(self, size):
        self.list_of_rewards = []
        self.size = size

    def add(self, rewards):
        if isinstance(rewards, list):
            self.list_of_rewards += rewards
        else:
            self.list_of_rewards.append(rewards)

        while len(self.list_of_rewards > self.size):
            del self.list_of_rewards[0]

    def average(self):
        return np.mean(self.list_of_rewards)


ma = MA(100)

#  training the ai
loss = nn.MSELoss()
optimizer = optim.adam(cnn.parameters(), lt=0.001)
np_epochs = 100
for epoch in range(1, np_epochs + 1):
    memory.run_steps(200)
    for batch in memory.sample_batch(128):
        inputs, target = eligibility_trace(batch)
        inputs, target = Variable(inputs), Variable(target)
        prediction = cnn(inputs)
        loss_error = loss(prediction, target)
        optimizer.zero_grad()
        loss_error.backward()
        optimizer.step()
    rewards_step = n_steps.rewards_steps()
    ma.add(rewards_step)
    avg_reward = ma.average()
    print('Epoch: %s , Average: %s' % (str(epoch), str(avg_reward)))
    if avg_reward > 1500:
        print("Congratulation, Your AI wins")
        break

# closing the doom environment
doom_env.close()