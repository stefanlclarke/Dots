import gym
from gym import error, spaces, utils
import numpy as np
from gym.utils import seeding
import torch as T

class DotsEnvironment(gym.Env):
    """Game environment made by Stefan"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.game_length = 5
        self.game_width = 5
        self.reward = 1
        self.punishment = 3
        self.board = np.zeros((self.game_length,self.game_width - 1))
        self.player_pos = 1
        self.score = 0
        self.player_direction = 1
        self.apple_pos = np.array([0, np.random.randint(self.game_width//2)*2+1])
        self.board[self.game_length - 1,1] = 1
        self.board[self.apple_pos[0], self.apple_pos[1]] = 1
        self.frames = 0

    def reset(self):
        self.board = np.zeros((self.game_length,self.game_width - 1))
        self.player_pos = 1
        self.score = 0
        self.player_direction = 1
        self.apple_pos = np.array([0, np.random.randint(self.game_width//2)*2+1])
        self.board[self.game_length - 1,1] = 1
        self.board[self.apple_pos[0], self.apple_pos[1]] = 1

    def reset_frames(self):
        self.frames = 0

    def render(self, mode='human', close=False):
        print(f'SCORE: {self.score}')
        print(f'DIRECTION: {self.player_direction}')
        print(self.board)

    def update_board(self):
        self.board = np.zeros((self.game_length,self.game_width - 1))
        self.board[self.apple_pos[0], self.apple_pos[1]] = 1
        self.board[self.game_length - 1, self.player_pos] = 1

    def step(self, action):
        score, miss, death = self._take_action(action)
        reward = 0
        if miss:
            self.new_apple()
        if score:
            self.score += 1
            self.new_apple()
            reward += self.reward
        if death:
            done = True
            reward -= self.punishment
        else:
            done = False

        self.frames += 1
        oldscore = self.score
        if done:
            self.reset()
        self.update_board()
        return reward, done, oldscore

    def new_apple(self):
        self.apple_pos = np.array([0, np.random.randint(self.game_width//2)*2+1])

    def get_events(self):
        if self.player_pos >= self.game_width - 1 or self.player_pos < 0:
            death = True
        else:
            death = False
        if self.apple_pos[0] == self.game_length - 1 and self.apple_pos[1] == self.player_pos:
            score = True
        else:
            score = False
        if self.apple_pos[0] > self.game_length - 1:
            miss = True
        else:
            miss = False
        return score, miss, death

    def _next_observation(self):
        return self.get_state()

    def get_state(self):
        vec_board = T.from_numpy(self.board)
        if self.player_direction == -1:
            dir = 0
        elif self.player_direction == 1:
            dir = 1
        else:
            raise ValueError("Invalid direction type")
        vec_direction = T.tensor([dir]).double()
        vec_board = T.flatten(vec_board).double()
        vec_state = T.cat((vec_board, vec_direction))
        return vec_state

    def _take_action(self, action):
        if action == 1:
            self.player_direction *= -1
        elif action != 0:
            raise ValueError("Input should be 0 or 1")
        self.player_pos += self.player_direction
        self.apple_pos[0] += 1
        score, miss, death = self.get_events()
        return score, miss, death
