import numpy as np
import random
from collections import deque


class Memory(object):
    def __init__(self, mem_cap, mini_batch_size):
        self.buffer = deque(maxlen=mem_cap)
        self.mini_batch_size = mini_batch_size

    def push(self, state, action, reward, next_state, done):
        experiance = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experiance)

    def sample(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        batch = (0, 0, 0, 0, 0)

        if len(self.buffer) >= self.mini_batch_size:
            batch = random.sample(self.buffer, self.mini_batch_size)
        else:
            batch = random.sample(self.buffer, len(self.buffer))

        for experiance in batch:
            state, action, reward, next_state, done = experiance
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch,
                next_state_batch, done_batch)

    def __len__(self):
        return len(self.buffer)
