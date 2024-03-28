from typing import List

from itertools import product
from random import choice

import numpy as np
import matplotlib.pyplot as plt


class Cube3D:

    def __init__(self, N) -> None:
        self.N = N
        self.cube = self.generate_cube(self.N)
        self.ACTIONS = None

    @staticmethod
    def generate_cube(n):
        return np.array([x for x in range(1, n**3+1)]).reshape(n, n, n)
    
    @property
    def action_keys(self):
        if self.ACTIONS is None:
            self.ACTIONS = self.get_actions_possible()
        return self.ACTIONS.keys()

    def sort_loss(self) -> float:
        loss = 0
        input = self.cube.reshape(self.N**3)
        for i in range(len(input)):
            for j in range(i, len(input)):
                if input[i] > input[j] and all([input[i] > 0, input[j] > 0]):
                    loss += 1
        return loss

    def plot_cube(self):
        N = self.cube.shape[0]
        axes = [N, N, N] # change to 64
        alpha = 0.5
        colors = np.empty(axes + [4], dtype=np.float32)
        colors[self.cube < 10] = [1, 1, 0, alpha]
        colors[(self.cube > 9) & (self.cube < 19)] = [1, 0, 0, alpha]
        colors[self.cube > 18] = [0, 0, 1, alpha]
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.voxels(
            self.cube,
            facecolors=colors,
            edgecolors='black'
        )
        plt.show()

    def rotate_cube(self, axi, level, clockwise=True):
        _r = 1 if clockwise is True else -1
        if axi not in (0, 1, 2):
            raise ValueError('`axi` must be one of {0, 1, 2}')
        if level > self.cube.shape[0] - 1:
            raise ValueError('`level` must be less then maximal index of a cube\'s side')
        if axi == 0:
            self.cube = np.rot90(self.cube, k=1, axes=(0,1))
            self.cube[level] = np.rot90(self.cube[level], k=_r, axes=(0, 1))
            self.cube = np.rot90(self.cube, k=-1, axes=(0,1))
        elif axi == 1:
            self.cube = np.rot90(self.cube, k=1, axes=(0,2))
            self.cube[level] = np.rot90(self.cube[level], k=_r, axes=(0, 1))
            self.cube = np.rot90(self.cube, k=-1, axes=(0,2))
        else:
            self.cube = np.rot90(self.cube, k=1, axes=(1,2))
            self.cube[level] = np.rot90(self.cube[level], k=_r, axes=(0, 1))
            self.cube = np.rot90(self.cube, k=-1, axes=(1,2))
        return self.cube

    def mix_cube(self, steps=50):
        for _ in range(steps):
            self.rotate_cube(
                choice(list(range(self.N))),
                choice(list(range(self.N))),
                clockwise=choice([True, False])
            )

    def get_actions_possible(self):
        """N is the cude dimension."""
        if self.ACTIONS is None:
            ACTIONS = {
                0: {
                    'f': lambda: self.cube,
                    'args': ()
                }
            }
            for i, (axis, level, clockwise) in enumerate(product(range(3), range(self.N), [True, False]), 1):
                ACTIONS[i] = {
                    'f': self.rotate_cube,
                    'args': (axis, level, clockwise)
                }
        return ACTIONS

    def apply_action(self, action_num: int) -> np.ndarray:
        if action_num not in self.action_keys:
            raise ValueError(f'`action_num` should be in {self.action_keys}')
        action = self.ACTIONS[action_num]
        return action['f'](*action['args'])
