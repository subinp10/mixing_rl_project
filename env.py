import gym
import numpy as np
from gym import spaces

class MixingEnv(gym.Env):
    def __init__(self):
        super(MixingEnv, self).__init__()

        self.n = 3
        self.max_pool = 10
        self.max_steps = 10

        self.base_fluids = [self._one_hot(i) for i in range(self.n)]

        self.action_space = spaces.MultiDiscrete([self.max_pool, self.max_pool])

        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.max_pool * self.n + self.n + 4,),
            dtype=np.float32
        )

    def _one_hot(self, idx):
        v = np.zeros(self.n)
        v[idx] = 1.0
        return v

    def reset(self):
        self.pool = self.base_fluids.copy()
        self.target = np.random.dirichlet(np.ones(self.n))
        self.steps = 0

        self.K = 0
        self.B = 0
        self.L = 0

        self.history = []

        return self._get_obs()

    def _pad_pool(self):
        padded = self.pool.copy()
        while len(padded) < self.max_pool:
            padded.append(np.zeros(self.n))
        return padded[:self.max_pool]

    def _get_obs(self):
        pool_flat = np.array(self._pad_pool()).flatten()
        return np.concatenate([
            pool_flat,
            self.target,
            [self.steps, self.K, self.B, self.L]
        ])

    def compute_error(self, mix):
        return np.sum(np.abs(mix - self.target))

    def step(self, action):
        i, j = action

        if i >= len(self.pool) or j >= len(self.pool):
            return self._get_obs(), -20, False, {}

        fluid1 = self.pool[i]
        fluid2 = self.pool[j]

        new_mix = (fluid1 + fluid2) / 2

        if i < self.n:
            self.K += 1
        if j < self.n:
            self.K += 1

        if i != j:
            self.B += 1

        self.L += abs(i - j)

        if len(self.pool) < self.max_pool:
            self.pool.append(new_mix)

        self.history.append((i, j))

        self.steps += 1

        error = self.compute_error(new_mix)

        reward = (
            - 25 * error
            - 2 * self.steps
            - 0.5 * self.K
            - 0.3 * self.B
            - 0.1 * self.L
        )

        done = False
        if error < 0.05 or self.steps >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, {}
