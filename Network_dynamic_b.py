import numpy as np
import random
import copy
import networkx as nx
import matplotlib.pyplot as plt

GRID_DISTANCE = 10         # The network is divided into numerous of grids, each grid contains one node
GRID_NODE_NUM_X = 4         # Number of girds/nodes in x axis
GRID_NODE_NUM_Y = 4        # Number of grids/nodes in y axis


class Network(object):
    """
    Establish wireless mesh network,
    Define node parameters and mesh topology
    """
    def __init__(self, Test):
        """====== Node parameters ======"""
        self.grid_distance = GRID_DISTANCE
        self.grid_xcor_node_number = GRID_NODE_NUM_X
        self.grid_ycor_node_number = GRID_NODE_NUM_Y
        self.node_number = self.grid_xcor_node_number*self.grid_ycor_node_number    # Total nodes num except the sink
        self.server_number = 1                                            # Num of sink node

        self.transmit_range = 15
        self.transmit_energy = 5
        self.min_distance_between_nodes = 7
        self.deploy_range_x = self.grid_xcor_node_number * self.grid_distance     # Range of network in x axis
        self.deploy_range_y = self.grid_ycor_node_number * self.grid_distance     # Range of network in y axis

        """"======= Network initialization parameters ======="""
        self.max_find_good_position_time = 5

        """"======= Network positions storage ======="""
        self.state_G = nx.Graph()
        self.state_G_no_sink = nx.Graph()
        self.state_xcor = []
        self.state_ycor = []
        self.state_link = []

        self.test = Test

    def setup_network(self):
        # Initiate node positions
        self.set_network_topology()
        # Ensure the graph is fully connected
        while not self.all_nodes_connected():
            self.set_network_topology()

    def all_nodes_connected(self):
        for i in range(0, self.node_number):
            for j in range(0, self.node_number):
                check = nx.has_path(self.state_G, i, j)
                if not check:
                    return False
        return True

    def set_network_topology(self):
        self.scatter_node_random_position()
        self.set_network_connectivity()
        self.set_graph()

    def scatter_node_random_position(self):
        self.state_xcor.clear()
        self.state_ycor.clear()
        if self.test:
            self.transmit_range = self.grid_distance
            for i in range(0, self.grid_xcor_node_number):
                for j in range(0, self.grid_ycor_node_number):  # Set other nodes positions
                    self.state_xcor.append(i*self.grid_distance)
                    self.state_ycor.append(j*self.grid_distance)
        else:
            for i in range(0, self.grid_xcor_node_number):
                for j in range(0, self.grid_ycor_node_number):  # Set other nodes positions
                    self.state_xcor.append(0)
                    self.state_ycor.append(0)
                    if not (i ==0 and j == 0):
                        for k in range(0, self.max_find_good_position_time):    # Constrain the distance between nodes
                            self.state_xcor[-1] = random.random() * self.grid_distance + i*self.grid_distance
                            self.state_ycor[-1] = random.random() * self.grid_distance + j*self.grid_distance
                            good_position = self.check_neighbor_distance(i*self.grid_ycor_node_number+j)
                            if good_position == 1:
                                break

    def check_neighbor_distance(self, node_id):
        good_position = 1
        ax = self.state_xcor[node_id]
        ay = self.state_ycor[node_id]
        for j in range(0, len(self.state_xcor)-1):
            bx = self.state_xcor[j]
            by = self.state_ycor[j]
            distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            if distance < self.min_distance_between_nodes:
                good_position = 0
        return good_position

    def set_network_connectivity(self):
        self.state_link.clear()
        for i in range(0, self.node_number):
            node_link = []
            for j in range(0, self.node_number):
                distance = ((self.state_xcor[i]-self.state_xcor[j])**2 +
                            (self.state_ycor[i]-self.state_ycor[j])**2)**0.5
                if (i != j) and (distance <= self.transmit_range):
                    node_link.append(1)
                else:
                    node_link.append(0)
            self.state_link.append(node_link)

    def set_graph(self):
        self.state_G.clear()
        for i in range(0, self.node_number):
            self.state_G.add_node(i, pos=(self.state_xcor[i], self.state_ycor[i]))
        for i in range(0, self.node_number):
            for j in range(i, self.node_number):
                if self.state_link[i][j] == 1 and self.state_link[j][i] == 1:
                    self.state_G.add_edge(i, j)
        self.state_G_no_sink = self.state_G.copy()
        self.state_G_no_sink.remove_node(0)

    def draw_network(self):
        plt.figure(1)
        pos = nx.get_node_attributes(self.state_G, 'pos')
        nx.draw(self.state_G, pos, with_labels=True, cmap=plt.get_cmap('Accent'),
                node_color='w', node_size=100)
        plt.show()

    def store_network(self):
        np.save("x_cor.npy", np.array(self.state_xcor[:]))
        np.save("y_cor.npy", np.array(self.state_ycor[:]))
        np.save("link.npy", np.array(self.state_link[:]))

    def reuse_network(self):
        self.state_xcor = np.load("x_cor.npy").tolist()
        self.state_ycor = np.load("y_cor.npy").tolist()
        self.state_link = np.load("link.npy").tolist()
        self.set_graph()


class Node(object):
    def __init__(self, node_id):
        self.node_id = node_id
        self.node_energy = 5000

        self.receive_queue = []
        self.send_queue = []
        self.pending_queue = []
        self.neighbor = []
        self.sleep_mode = 0

        self.sampling_time = -1
        self.trace_time = -1
        self.hop_dis = 0

    def reset(self):
        self.node_energy = 5000
        self.receive_queue.clear()
        self.send_queue.clear()
        self.pending_queue.clear()
        self.sleep_mode = 0

        self.sampling_time = -1
        self.trace_time = -1


class NetworkMPC(Network):
    def __init__(self, reuse=False):
        super(NetworkMPC, self).__init__(reuse)
        """======= Workload parameters ======="""
        self.time = 0
        self.reuse = reuse
        self.MPC_connectivity = np.zeros((self.node_number, self.node_number),
                                         dtype=int)

        """======= Performance analysis parameters ======="""
        #self.sink_analysis = pd.DataFrame(index=list(np.arange(0, time_scope)),
                                          #columns=["raw_throughput", "fine_throughput", "transmit_delay"])
        self.traffic = []

        self.node_id_sapce = [n for n in range(1, self.node_number)]
        self.node_id_sapce = (self.node_id_sapce - np.mean(self.node_id_sapce)) \
                             /(np.max(self.node_id_sapce - np.min(self.node_id_sapce)))
        self.action_space = [n for n in range(0, 3)]
        self.action_space = (self.action_space - np.mean(self.action_space)) \
                            /(np.max(self.action_space) - np.min(self.action_space))
        self.traffic_space = [n for n in range(0, self.node_number)]
        self.traffic_space = (self.traffic_space - np.mean(self.traffic_space)) / self.node_number

        self.act_hist = np.array([self.action_space[0]]*(self.node_number-1))

        self.setup_network()
        self.node_object = {}
        for i in range(0, self.node_number):
            self.node_object[str(i)] = Node(i)
        self.mpc_candidate()
        self.init_node()
        self.init_mpc()
        self.get_load()

    def init_node(self):
        for i in range(1, self.node_number):
            self.node_object[str(i)].reset()
            self.traffic.append(0)

    def mpc_candidate(self):
        dtype = [('nei', int), ('length', int), ('tosink', int)]
        for i in range(1, self.node_number):
            values = []
            for j in range(1, self.node_number):
                if i != j:
                    route1 = self.find_route(i, j)
                    route2 = self.find_route(j, 0)
                    values.append((j, len(route1), len(route2)))

            #a = np.array(values, dtype=dtype)
            #sorted = np.sort(a, order='length')
            values.sort(key=lambda x:(x[1], x[2]))

            #print("node:{}:{}".format(i, sorted[:3]))
            tmp = []
            for k in range(5):
                tmp.append(values[k][0])
            self.node_object[str(i)].neighbor = tmp
            print("node:{}:{}".format(i, tmp))

    def init_mpc(self):
        # Find a one-hop neighbor for each node to initialize MPC connectivity matrix
        self.MPC_connectivity = np.zeros((self.node_number, self.node_number),
                                         dtype=int)
        for i in range(1, self.node_number):
            #rand = random.randint(0, 2)
            rand = 0
            mpc_node = self.node_object[str(i)].neighbor[rand]
            self.MPC_connectivity[i, mpc_node] = 1
            self.node_object[str(i)].hop_dis = len(self.find_route(i, 0)) - 1

    def find_route(self, s, t):
        if t == 0:
            check = nx.has_path(self.state_G, source=s, target=t)
            if check:
                path = nx.dijkstra_path(self.state_G, source=s, target=t)
            else:
                path = []
        else:
            check = nx.has_path(self.state_G_no_sink, source=s, target=t)
            if check:
                path = nx.dijkstra_path(self.state_G_no_sink, source=s, target=t)
            else:
                path = []
            #print('The path is: ' + str(s) + '-' + str(t) + ' = ' + str(path))
        return path

    def generate_source(self):
        # Generate source
        # [ [MPC_num, TYPE, R], [s, t, ST, ET], [[ROUTE], i]] ]
        for i in range(1, self.node_number):
            tmp_load = 0
            num_mpc = list(self.MPC_connectivity[i][1:]).count(1)   # Number of sharing nodes
            #self.traffic[i - 1] += num_mpc
            for j in range(num_mpc):
                t = list(self.MPC_connectivity[i]).index(1, j)  # Target node
                s2sh_route = self.find_route(i, t)
                tmp_load += len(s2sh_route) - 1
                sh2sink_route = self.find_route(t, 0)
                tmp_load += len(sh2sink_route) - 1

            s2sink_route = self.find_route(i, 0)
            tmp_load += len(s2sink_route) - 1
            self.traffic[i-1] = tmp_load

    def operate(self, time_scope=10):
        """
        Simulate network transmission within given time steps
        """
        self.generate_source()
        self.time += 1

    def store_network(self):
        np.save("x_cor.npy", np.array(self.state_xcor[:]))
        np.save("y_cor.npy", np.array(self.state_ycor[:]))
        np.save("link.npy", np.array(self.state_link[:]))
        np.save("mpc_connectivity.npy", np.array(self.MPC_connectivity[:]))

    def reuse_network(self):
        self.state_xcor = np.load("x_cor.npy").tolist()
        self.state_ycor = np.load("y_cor.npy").tolist()
        self.state_link = np.load("link.npy").tolist()
        self.MPC_connectivity = np.load("mpc_connectivity.npy").tolist()
        self.MPC_connectivity = np.array(self.MPC_connectivity)
        self.set_graph()
        self.mpc_candidate()
        self.reset_network()

    def reset_network(self):
        self.time = 0
        self.MPC_connectivity.fill(0)

        """======= Performance analysis parameters ======="""
        self.traffic.clear()
        self.act_hist.fill(self.action_space[0])

        self.init_node()
        self.init_mpc()

    def step(self, node_id, action):
        self.act_action(node_id, action)
        self.traffic.clear()
        self.init_node()
        self.operate()
        reward = self.get_reward(node_id)
        #state_ = self.observer(node_id)
        state_ = self.observer(node_id%(self.node_number-1)+1)
        return state_, reward

    def observer(self, node_id):
        state = np.reshape(self.MPC_connectivity[1:, 1:]-0.5, [1, (self.node_number-1)**2])[0]
        #state = list()
        #privacy = list()
        #for i in range(1, self.node_number):
        #    nei = self.node_object[str(i)].neighbor
        #    num_sharing_node = np.count_nonzero(self.MPC_connectivity[i])
        #    privacy.append(num_sharing_node / len(nei))

        #state = np.append(state, privacy)
        state = np.append(state, self.act_hist)
        state = np.append(state, self.node_id_sapce[node_id-1])
        state = state[np.newaxis]
        return state

    def act_action(self, node_id, action):
        self.act_hist[node_id-1] = self.action_space[action]
        connectivity = self.MPC_connectivity[node_id]
        num_mpc = np.count_nonzero(connectivity)  # Number of sharing nodes
        nei = self.node_object[str(node_id)].neighbor

        if action == 1 and num_mpc > 1:
            connectivity[nei[num_mpc-1]] = 0
        elif action == 2 and num_mpc < len(nei):
            connectivity[nei[num_mpc]] = 1

    def get_reward(self, node_id):
        p = list()
        for i in range(1, self.node_number):
            nei = self.node_object[str(i)].neighbor
            num_sharing_node = np.count_nonzero(self.MPC_connectivity[i])
            p.append(num_sharing_node / (len(nei)))

        b = 1 - abs(self.traffic- np.mean(self.traffic)) / np.mean(self.traffic)

        H = 1 - (np.array(self.traffic) - np.array(self.min_load))/(np.array(self.max_load) - np.array(self.min_load))
        #H = 1 - (np.sum(self.traffic) - np.sum(self.min_load))/(np.sum(self.max_load) - np.sum(self.min_load))
        #print(H)

        #reward = H*np.sum(p*b)/(self.node_number-1)
        reward = H[node_id-1]*p[node_id-1]*b[node_id-1]
        return reward

    def get_load(self):
        self.operate()
        self.min_load = self.traffic.copy()
        print("Min Load" + str(self.min_load))
        #print(self.MPC_connectivity)
        for n in range(5):
            for i in range(1, self.node_number):
                self.act_action(i, 2)
        self.operate()
        self.max_load = self.traffic.copy()
        print("Max Load" + str(self.max_load))
        #print(self.MPC_connectivity)
        self.init_mpc()

if __name__ == "__main__":
    net = NetworkMPC()
    plt.ion()
    net.store_network()
    net.draw_network()
    net.step(1,0)
    print(net.get_reward(1))


    '''
    x = [n for n in range(1, net.node_number)]
    print(net.min_load)
    print(np.mean(net.min_load))
    print(net.max_load)
    print(np.mean(net.max_load))
    for j in range(10):
        for i in range(1, net.node_number):
            net.act_action(i, j)
        state_, reward = net.step(1, j)
        print("---")
        #y = net.traffic
        a = np.array([(j+1)/10]*(net.node_number-1))
        b = 1 - (np.array(net.traffic) - np.array(net.min_load))/(np.array(net.max_load) - np.array(net.min_load))
        c = 1 - abs(net.traffic- np.mean(net.traffic)) / np.mean(net.traffic)
        y = a
        y = b
        y = c
        y = c
        if j==0:
            plt.plot(x, y, 'g--')
        elif j == 9:
            plt.plot(x, y, 'r--')
        else:
            plt.plot(x, y, 'b-')
        print(np.mean(y))
        print(net.MPC_connectivity)

    for j in range(1):
        for i in range(1, net.node_number-1):
            net.act_action(i, 1)
            #net.act_action(i, random.randint(0,4))
        state_, reward = net.step(1, 4)
        state_, reward = net.step(2, 4)
        state_, reward = net.step(3, 4)
        print("---")
        #y = net.traffic
        a = np.array([(j+1)/5]*(net.node_number-1))
        b = 1 - (np.sum(net.traffic) - np.sum(net.min_load))/(np.sum(net.max_load) - np.sum(net.min_load))
        c = 1 - abs(net.traffic- np.mean(net.traffic)) / np.mean(net.traffic)
        y = a
        y = b
        y = c
        y = a*b*c
        plt.plot(x, y, 'p-')
    '''
    plt.ioff()
    plt.show()
