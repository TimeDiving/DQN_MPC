import numpy as np
import random
import copy
import networkx as nx
import matplotlib.pyplot as plt
from Network_dynamic_object import NetworkMPC

GRID_DISTANCE = 10         # The network is divided into numerous of grids, each grid contains one node
GRID_NODE_NUM_X = 5        # Number of girds/nodes in x axis
GRID_NODE_NUM_Y = 2        # Number of grids/nodes in y axis

class TestNet(NetworkMPC):
    def __init__(self):
        super(TestNet, self).__init__()
        #self.MPC_connectivity[4:, :].fill(0)

    def scatter_node_random_position(self):
        self.state_xcor.clear()
        self.state_ycor.clear()
        self.state_xcor.append(self.deploy_range_x - 1)
        self.state_ycor.append(self.deploy_range_y / 2)  # Set sink node position
        for i in range(1, 4):
            self.state_xcor.append(self.deploy_range_x - 12 * i)
            self.state_ycor.append(self.deploy_range_y / 2)  # Set sink node position

        for i in range(1, 4):
            self.state_xcor.append(self.deploy_range_x - 12 * i)
            self.state_ycor.append(self.deploy_range_y / 2 - 2.5 * i)  # Set sink node position
            self.state_xcor.append(self.deploy_range_x - 12 * i)
            self.state_ycor.append(self.deploy_range_y / 2 + 2.5 * i)  # Set sink node position
        self.state_xcor.append(self.deploy_range_x - 12 * 4)
        self.state_ycor.append(self.deploy_range_y / 2)  # Set sink node position

    def set_network_connectivity(self):
        self.state_link.clear()
        for i in range(0, self.node_number + self.server_number):
            node_link = [0] * (self.node_number + self.server_number)
            self.state_link.append(node_link)
        self.state_link[0][1] = 1
        self.state_link[1][0] = 1

        self.state_link[1][2] = 1
        self.state_link[2][1] = 1
        self.state_link[1][4] = 1
        self.state_link[4][1] = 1
        self.state_link[1][5] = 1
        self.state_link[5][1] = 1

        self.state_link[2][3] = 1
        self.state_link[3][2] = 1
        self.state_link[2][6] = 1
        self.state_link[6][2] = 1
        self.state_link[2][7] = 1
        self.state_link[7][2] = 1

        self.state_link[3][8] = 1
        self.state_link[8][3] = 1
        self.state_link[3][9] = 1
        self.state_link[9][3] = 1
        self.state_link[3][10] = 1
        self.state_link[10][3] = 1

    def draw_network(self):
        plt.figure(1)
        pos = nx.get_node_attributes(self.state_G, 'pos')
        used_color = 0
        color = [1] * (self.node_number + self.server_number)
        for i in range(1, 4):
            used_color = (used_color + 0.2) % 1
            color[i] = used_color
            connect = self.MPC_connectivity[i, :]
            index = list(np.where(connect)[0])
            if 1 in index: index.remove(1)
            if 2 in index: index.remove(2)
            if 3 in index: index.remove(3)
            for id in index:
                color[id] = used_color

        nx.draw(self.state_G, pos, with_labels=True,
                node_color=color, node_size=100)
        plt.show()

    def reset_network(self):
        self.time = 0
        self.MPC_connectivity.fill(0)

        """======= Performance analysis parameters ======="""
        self.raw_throughput.clear()
        self.fine_throughput.clear()
        self.transmit_delay.clear()
        self.mean_delay.clear()
        self.traffic.clear()
        self.act_hist.fill(0)

        for i in range(0, self.node_number + self.server_number):
            self.node_object[str(i)].reset()
        self.init_node()
        self.init_mpc()
        #self.MPC_connectivity[4:, :].fill(0)

    def observer(self):
        state = list()
        for i in range(1, self.node_number+self.server_number):
            nei = [n for n in self.state_G.neighbors(i)]
            if 0 in nei:
                nei.remove(0)
            num_sharing_node = np.count_nonzero(self.MPC_connectivity[i])
            state.append(num_sharing_node / len(nei))
        state = np.reshape(state, [1, self.node_number])[0]
        #state = np.append(state, self.act_hist)
        '''
        state = self.MPC_connectivity[1:, 1:].copy()
        state = np.reshape(state, [1, self.node_number**2])[0]
        #state = np.append(state, (1, 2, 3))
        """
        load = self.traffic[0:3]
        # load = (load - np.mean(load))/np.std(load)
        load = (load - np.min(load)) / (np.max(load) - np.min(load))
        state = np.append(state, load)
        state = np.append(state, self.mean_delay[-1])
        # state = np.append(state, self.mean_delay[-1])
        """
        '''

        state = np.append(state, 0)
        state = state[np.newaxis]
        return state

    def step(self, node_id, action):
        self.act_action(node_id, action)
        self.time = 0
        self.raw_throughput.clear()
        self.fine_throughput.clear()
        self.transmit_delay.clear()
        self.mean_delay.clear()
        self.traffic.clear()
        self.init_node()
        self.operate(self.sampling_period)

        reward = self.get_reward()
        state_ = self.observer()
        return state_, reward

    def get_reward(self):
        privacy = list()
        for i in range(1, 11):
            nei = [n for n in self.state_G.neighbors(i)]
            if 0 in nei:
                nei.remove(0)
            num_sharing_node = np.count_nonzero(self.MPC_connectivity[i])
            privacy.append(num_sharing_node / len(nei))

        load = self.traffic.copy()
        load = (load - np.mean(load))/np.std(load)
        #load = (load - np.mean(load)) / (np.max(load) - np.min(load))
        load = -1*load
        reward = np.dot(privacy, load)
        #reward = -1*self.mean_delay[-1]
        #reward = np.sum(privacy)/len(privacy)
        return reward

