import torch as T
import numpy as np
import gym
import gym_dots
from matplotlib import pyplot as plt

class ACagent:
    def __init__(self, net, env, memory, gamma, lr):
        self.ac = net
        self.game = env
        self.memory = memory
        self.gamma = gamma
        self.lr = lr
        self.optimizer = T.optim.Adam(self.ac.parameters(), lr=lr)
        self.epoch_scores =[]

    def score_plot(self):
        plt.plot(self.epoch_scores)

    def train(self, epochs, steps_per_epoch):
        self.epoch_scores = []
        for epoch in range(epochs):
            self.game.reset_frames()
            self.game.reset()
            self.memory.reset()
            epoch_scores = []
            playing = True
            while playing:
                old_state = self.game.get_state()
                print('state: ', old_state)
                actor_out = self.ac.actor.forward(old_state.double())
                move_chosen = np.random.choice(2, p=actor_out.clone().detach().numpy())
                reward, done, score = self.game.step(move_chosen)
                new_state = self.game.get_state()
                self.memory.addstate(old_state, actor_out, move_chosen, (reward, done), new_state)
                if done:
                    epoch_scores.append(score)
                if self.game.frames > steps_per_epoch:
                    playing = False

            num_steps = len(self.memory.memory)
            Qerrors = []
            nextQs = []
            logprobs = []
            for state in self.memory.memory:
                prevq = self.ac.critic.forward(state[0].double())
                newq = self.ac.critic.forward(state[5].double()).detach()
                reward = state[3]
                Qerror = (prevq - reward - self.gamma*newq)**2
                Qerrors.append(Qerror)
                nextQs.append(newq)
                move_chosen = state[2]
                probs = state[1]
                if move_chosen == 0:
                    ido = T.tensor([1,0])
                elif move_chosen == 1:
                    ido = T.tensor([0,1])
                else:
                    raise ValueError("Impossible move chosen")
                print('probs: ', probs)
                print('move: ', move_chosen)
                print('ido: ', ido)
                pty = (ido*probs).sum()
                print('pty: ', pty)
                logprob = T.log(pty)
                logprobs.append(logprob)

            Qerrors = T.stack(Qerrors)
            nextQs = T.stack(nextQs)
            nextQs = nextQs.detach()
            logprobs = T.stack(logprobs)
            actor_loss = (-logprobs*nextQs).mean()
            critic_loss = Qerrors.mean()
            loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            avg_score = (sum(epoch_scores) + self.game.score)/max(len(epoch_scores),1)
            print(f"EPOCH: {epoch}")
            print(f"AVG SCORE: {avg_score}")
            print(f"LOSS: {loss}")
            self.epoch_scores.append(avg_score)
