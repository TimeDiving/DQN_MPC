import os
import numpy as np
import random
import matplotlib.pyplot as plt

from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

from TestNet1 import TestNet
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class DQN(object):
    def __init__(
            self,
            n_features,
            n_actions,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            epsilon_decay = 0.995,
            epsilon_min = 0.01,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
    ):
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.learn_step_counter = 0

        # initialize memory
        self.memory = deque(maxlen=self.memory_size)
        # create q network
        self.q_network = self._build_model(self.n_features, self.n_actions)

        # create target network
        self.target_network = self._build_model(self.n_features, self.n_actions)
        self.replace_target_op()

    def _build_model(self, input_size, output_size, hidden_size0=50, hidden_size1=25):
        model = Sequential()
        model.add(Dense(hidden_size0, input_shape=(input_size,), activation='relu'))
        model.add(Dense(hidden_size1, activation='relu'))
        model.add(Dense(output_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def act(self, state):
        if np.random.uniform() <= self.epsilon or len(self.memory) < self.batch_size:
            action = np.random.randint(0, self.n_actions)
        else:
            act_values = self.q_network.predict(state)
            action = np.argmax(act_values[0])
        return action

    def remember(self, states):
        self.memory.append(states)

    def update_epsilon(self):
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replace_target_op(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def replay(self):
        # check if replace target_net
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.replace_target_op()

        minibatch = random.sample(self.memory, self.batch_size)
        assert len(minibatch) == self.batch_size

        states = np.array([d[0] for d in minibatch])
        next_states = np.array([d[3] for d in minibatch])

        y = self.q_network.predict(states)
        q = self.target_network.predict(next_states)

        for i , (_, action, reward, _) in enumerate(minibatch):
            target = reward + self.gamma*np.amax(q[i])
            y[i][0][action] = target

        loss = self.q_network.train_on_batch(states, y)
        return loss

if __name__ == "__main__":
    env = TestNet()
    agent = DQN(34, 3)
    history = list()

    env.draw_network()
    print(env.MPC_connectivity)

    episodes = 5
    for e in range(episodes):
        env.reset_network()
        env.operate(env.sampling_period)
        state = env.observer()

        for time_t in range(100):

            for i in range(1, env.node_number+1):
                state[0][-1] = i
                action = agent.act(state)
                state_, reward = env.step(i, action)

                state_[0][-1] = i%env.node_number + 1
                agent.remember((state, action, reward, state_))

                state = state_

        hist = agent.replay()
        print("Episode {}: {}".format(e, hist))
        history.append(hist)

    print(env.MPC_connectivity)


    plt.ion()
    env.draw_network()
    plt.figure(2)
    plt.plot(np.array(history), c='b')
    plt.ylabel('Loss')
    plt.xlabel('training steps')
    plt.grid()
    plt.ioff()
    plt.show()

