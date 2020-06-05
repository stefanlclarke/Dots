import numpy as np
import torch as T
import torch.nn as nn
import numpy as np

class Memory:
    def __init__(self, maxframes):
        self.memory = []
        self.maxframes = maxframes

    def addstate(self, prev_state, net_out, movemade, step_out, new_state):
        reward = step_out[0]
        done = step_out[1]
        self.memory.append([prev_state, net_out, movemade, reward, done, new_state])

    def erase_old(self):
        self.memory = self.memory[-self.maxframes:-1]

    def reset(self):
        self.memory = []

class Actor(T.nn.Module):
    def __init__(self, input_dim, l1, l2, output_dim):
        super(Actor, self).__init__()
        self.indim = input_dim
        self.l1 = l1
        self.l2 = l2
        self.out = output_dim

        self.fc1 = nn.Linear(self.indim, self.l1)
        self.fc2 = nn.Linear(self.l1, self.l2)
        self.fc_out = nn.Linear(self.l2, self.out)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, input):
        o1 = self.relu(self.fc1(input.double()))
        o2 = self.relu(self.fc2(o1))
        out = self.softmax(self.fc_out(o2))
        return out

class Critic(T.nn.Module):
    def __init__(self, input_dim, l1, l2, l3):
        super(Critic, self).__init__()
        self.indim = input_dim
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

        self.fc1 = nn.Linear(self.indim, self.l1)
        self.fc2 = nn.Linear(self.l1, self.l2)
        self.fc3 = nn.Linear(self.l2, self.l3)
        self.fc4 = nn.Linear(self.l3, 1)
        self.relu = nn.ReLU()

    def forward(self, input):
        o1 = self.relu(self.fc1(input.double()))
        o2 = self.relu(self.fc2(o1))
        o3 = self.relu(self.fc3(o2))
        out = self.fc4(o3)
        return out

class AC(T.nn.Module):
    def __init__(self, input_dim, actor_layers, critic_layers, output_dim):
        super(AC, self).__init__()
        self.actor = Actor(input_dim, actor_layers[0], actor_layers[1], output_dim)
        self.critic = Critic(input_dim, critic_layers[0], critic_layers[1], critic_layers[2])
