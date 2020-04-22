import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class Critic(object):
    def __init__(self, input_size, hidden_size, output_size):
        self.model = Sequential()
        self.model.add(Dense(hidden_size, input_dim=input_size,
                       activation='relu'))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(output_size, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=0.001))

    def act(self, states, actions):
        state_action = np.array([states, actions])
        act_value = self.model.predict(state_action)
        return act_value


class Actor(object):
    def __init__(self, input_size, hidden_size, output_size):
        self.model = Sequential()
        self.model.add(Dense(hidden_size, input_dim=input_size,
                       activation='relu'))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(output_size, activation='tanh'))
        self.model.compile(loss='mse', optimizer=Adam(lr=0.001))

    def act(self, state):
        state = np.array([state])
        act_values = self.model.predict(state)
        return np.array(act_values[0])
