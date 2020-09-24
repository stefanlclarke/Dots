import torch as T
import numpy as np
import gym
import gym_dots
from ACagent import ACagent
from networks import Memory, AC
from matplotlib import pyplot as plt

env = gym.make('dots-v0')

actor_layers = [10, 10]
critic_layers = [10, 10, 10]
memory = Memory(2000)
ac = AC(21, actor_layers, critic_layers, 2).double()
gamma = 0.9
learning_rate = 3e-4

agent = ACagent(ac, env, memory, gamma, learning_rate)

agent.train(800,1000)
agent.score_plot()
