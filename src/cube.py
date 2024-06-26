from typing import List

from itertools import product
from random import choice

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns


class Cube3D:

    def __init__(self, N) -> None:
        self.N = N
        self.cube = self.generate_cube(self.N)
        self.ACTIONS = None

    @staticmethod
    def generate_cube(n):
        cube = np.array([x for x in range(1, n**3+1)]).reshape(n, n, n)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if all([i>0 and i < n-1, j>0 and j < n-1, k>0 and k < n-1]):
                        cube[i][j][k] = 0
        return cube
    
    @property
    def action_keys(self):
        if self.ACTIONS is None:
            self.ACTIONS = self.get_actions_possible()
        return self.ACTIONS.keys()
    
    @property
    def flattened(self) -> np.ndarray:
        return self.cube.reshape(self.N**3)

    def sort_loss(self) -> float:
        loss = 0
        input = self.flattened
        for i in range(len(input)):
            for j in range(i, len(input)):
                if input[i] > input[j] and all([input[i] > 0, input[j] > 0]):
                    loss += 1
        return loss

    def plot(self, alpha=0.9, colors_set=None):
        N = self.cube.shape[0]
        axes = [N, N, N] # change to 64
        colors = np.empty(axes + [4], dtype=np.float32)
        if colors_set is None:
            colors_set = sns.color_palette("husl", N+1).as_hex()
        COLORS_LIST  = [list(mcolors.to_rgba_array(list(colors_set)[i], alpha=alpha)[0]) for i in range(N+1)]
        STEPS = [0] + [i*(N**3//N) for i in range(1, N+1)] + [10**10]
        for i, s in enumerate(STEPS[0:len(STEPS)-1]):
            colors[(self.cube > s) & (self.cube <= STEPS[i+1])] = COLORS_LIST[i]
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.voxels(
            self.cube,
            facecolors=colors,
            edgecolors='black'
        )
        plt.show()

    def rotate(self, axi, level, clockwise=True):
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

    def mix(self, steps=50):
        for _ in range(steps):
            self.rotate(
                choice([0, 1, 2]),
                choice(list(range(self.N))),
                clockwise=choice([True, False])
            )

    def get_actions_possible(self):
        """N is the cude dimension."""
        if self.ACTIONS is None:
            self.ACTIONS = {
                0: {
                    'f': lambda: self.cube,
                    'args': ()
                }
            }
            for i, (axis, level, clockwise) in enumerate(product(range(3), range(self.N), [True, False]), 1):
                self.ACTIONS[i] = {
                    'f': self.rotate,
                    'args': (axis, level, clockwise)
                }
        return self.ACTIONS

    def apply_action(self, action_num: int) -> np.ndarray:
        if action_num not in self.action_keys:
            raise ValueError(f'`action_num` should be in {self.action_keys}')
        action = self.ACTIONS[action_num]
        return action['f'](*action['args'])

    def solve(self, model):
        pass
