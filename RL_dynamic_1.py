import os
import numpy as np
import random
import matplotlib.pyplot as plt

from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Lambda, Subtract, Add
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from TestNet1 import TestNet
from Network_dynamic_b import NetworkMPC
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Use cpu
#os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"    # Limit GPU storage

class DQN(object):
    def __init__(
            self,
            n_features,
            n_actions,
            learning_rate=0.001,
            reward_decay=0.9,
            e_greedy=1,
            epsilon_decay = 0.995,
            epsilon_min = 0.001,
            replace_target_iter=10,
            memory_size=2000,
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
        self.ddqn = True
        self.dueling_dqn = False

        # initialize memory
        self.memory = list()
        # create q network
        self.q_network = self._build_model(self.n_features, self.n_actions)

        # create target network
        self.target_network = self._build_model(self.n_features, self.n_actions)

    def _build_model(self, input_size, output_size, hidden_size0=200, hidden_size1=10):
        if self.dueling_dqn:
            inputs = Input(shape=(input_size,))
            x = Dense(hidden_size0, activation='relu')(inputs)
            x = Dense(hidden_size1, activation='relu')(x)

            value = Dense(3, activation='linear')(x)
            a = Dense(3, activation='linear')(x)
            mean = Lambda(lambda  x: K.mean(x, axis=1, keepdims=True))(a)
            advantage = Subtract()([a, mean])

            q = Add()([value, advantage])

            model = Model(inputs=inputs, outputs=q)
            model.compile(loss='mse', optimizer=Adam(self.lr))
        else:
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
        if len(self.memory) > self.memory_size:
            del self.memory[0]

    def update_epsilon(self):
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replace_target_op(self):
        #K.clear_session()
        self.target_network.set_weights(self.q_network.get_weights())

    def replay(self):
        # check whether replace target_net
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.replace_target_op()

        minibatch = random.sample(self.memory, self.batch_size)
        assert len(minibatch) == self.batch_size

        states = np.array([d[0] for d in minibatch])
        next_states = np.array([d[3] for d in minibatch])

        y = self.q_network.predict(states)
        q = self.target_network.predict(next_states)
        if self.ddqn:
            next_action = np.argmax(self.q_network.predict(next_states), axis=-1)
            for i , (_, action, reward, _) in enumerate(minibatch):
                target = reward + self.gamma*q[i][0][next_action[i]]
                y[i][0][action] = target
        else:
            for i , (_, action, reward, _) in enumerate(minibatch):
                target = reward + self.gamma*np.amax(q[i])
                y[i][0][action] = target
        loss = self.q_network.train_on_batch(states, y)
        self.learn_step_counter += 1
        return loss

if __name__ == "__main__":
    env = NetworkMPC(True)
    env.draw_network()
    print(env.MPC_connectivity)

    loss_history = list()
    reward_history = list()

    agent = DQN((env.node_number-1)**2+(env.node_number-1)*1+1, 3)

    episodes = 300
    step = 0
    for e in range(episodes):
        env.reset_network()
        state = env.observer(1)
        reward_sum = 0
        tmp = 0

        for t in range(20):
            for i in range(1, env.node_number):
                action = agent.act(state)
                state_, reward = env.step(i, action)
                reward_sum += reward
                agent.remember((state, action, reward, state_))
                state = state_
                tmp = reward
            if (t+1)%5 == 0:
                step += 1
                print(tmp)
                reward_sum = reward_sum/(5*(env.node_number-1))

                loss = agent.replay()
                agent.update_epsilon()
                print("Step {}: {} | Reward: {}".format(step, loss, reward_sum))
                print("State: {} | Epsilon：{}".format(env.act_hist, agent.epsilon))

                loss_history.append(loss)
                reward_history.append(reward_sum)
                reward_sum = 0

    print(env.MPC_connectivity)
    print("Act_history:{}".format(env.act_hist))

    plt.ion()
    env.draw_network()
    plt.figure(2)
    graph1 = plt.subplot(1, 2, 1)
    plt.plot(np.array(loss_history), c='b')
    plt.ylabel('Loss')
    plt.xlabel('training steps')
    graph2 = plt.subplot(1, 2, 2)
    plt.plot(np.array(reward_history), c='b')
    plt.ylabel('Reward')
    plt.xlabel('training steps')
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.grid()
    plt.ioff()
    plt.show()





