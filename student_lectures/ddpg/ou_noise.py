import numpy as np


class OUNoise(object):

    def __init__(self,
                 mu=0.0,
                 theta=0.15,
                 max_sigma=0.3,
                 min_sigma=0.3,
                 decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = 1
        self.low = -2.0
        self.high = 2.0
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        left = self.theta * (self.mu - x)
        right = self.sigma * np.random.randn(self.action_dim)
        dx = left + right
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        left = self.max_sigma - (self.max_sigma - self.min_sigma)
        right = min(1.0, t / self.decay_period)
        self.sigma = left * right
        return np.clip(action + ou_state, self.low, self.high)
