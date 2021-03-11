import numpy as np
import random
import copy
import networkx as nx
import matplotlib.pyplot as plt

GRID_DISTANCE = 10         # The network is divided into numerous of grids, each grid contains one node
GRID_NODE_NUM_X = 4        # Number of girds/nodes in x axis
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
        for i in range(0, self.node_number + self.server_number):
            for j in range(0, self.node_number + self.server_number):
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
            self.state_xcor.append((self.deploy_range_x-self.grid_distance) / 2)
            self.state_ycor.append((self.deploy_range_y-self.grid_distance) / 2)  # Set sink node position
            for i in range(0, self.grid_xcor_node_number):
                for j in range(0, self.grid_ycor_node_number):  # Set other nodes positions
                    self.state_xcor.append(i*self.grid_distance)
                    self.state_ycor.append(j*self.grid_distance)

        else:
            self.state_xcor.append(self.deploy_range_x / 2)
            self.state_ycor.append(self.deploy_range_y / 2)  # Set sink node position
            for i in range(0, self.grid_xcor_node_number):
                for j in range(0, self.grid_ycor_node_number):  # Set other nodes positions
                    self.state_xcor.append(0)
                    self.state_ycor.append(0)
                    for k in range(0, self.max_find_good_position_time):    # Constrain the distance between nodes
                        self.state_xcor[-1] = random.random() * self.grid_distance + i*self.grid_distance
                        self.state_ycor[-1] = random.random() * self.grid_distance + j*self.grid_distance
                        good_position = self.check_neighbor_distance(i+1)
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
        for i in range(0, self.node_number + self.server_number):
            node_link = []
            for j in range(0, self.node_number + self.server_number):
                distance = ((self.state_xcor[i]-self.state_xcor[j])**2 +
                            (self.state_ycor[i]-self.state_ycor[j])**2)**0.5
                if (i != j) and (distance <= self.transmit_range):
                    node_link.append(1)
                else:
                    node_link.append(0)
            self.state_link.append(node_link)

    def set_graph(self):
        self.state_G.clear()
        for i in range(0, self.node_number + self.server_number):
            self.state_G.add_node(i, pos=(self.state_xcor[i], self.state_ycor[i]))
        for i in range(0, self.node_number + self.server_number):
            for j in range(i, self.node_number + self.server_number):
                if self.state_link[i][j] == 1 and self.state_link[j][i] == 1:
                    self.state_G.add_edge(i, j)
        self.state_G_no_sink = self.state_G.copy()
        self.state_G_no_sink.remove_node(0)

    def draw_network(self):
        plt.figure(1)
        pos = nx.get_node_attributes(self.state_G, 'pos')
        nx.draw(self.state_G, pos, with_labels=True, cmap=plt.get_cmap('Accent'),
                node_color='g', node_size=100)
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
        super(NetworkMPC, self).__init__(True)
        """======= Workload parameters ======="""
        self.time = 0
        self.reuse = reuse
        self.sampling_period = 100
        self.trace_period = 50
        self.MPC_connectivity = np.zeros((self.node_number+self.server_number, self.node_number+self.server_number),
                                         dtype=int)

        """======= Performance analysis parameters ======="""
        #self.sink_analysis = pd.DataFrame(index=list(np.arange(0, time_scope)),
                                          #columns=["raw_throughput", "fine_throughput", "transmit_delay"])
        self.raw_throughput = []
        self.fine_throughput = []
        self.transmit_delay = [0]*self.node_number
        self.mean_delay = []
        self.traffic = []

        self.node_id_sapce = [n for n in range(1, self.node_number+1)]
        self.node_id_sapce = (self.node_id_sapce - np.mean(self.node_id_sapce)) \
                             /(np.max(self.node_id_sapce - np.min(self.node_id_sapce)))
        self.action_space = [n for n in range(1, 5)]
        self.action_space = (self.action_space - np.mean(self.action_space)) \
                            /(np.max(self.action_space) - np.min(self.action_space))
        self.traffic_space = [n for n in range(0, self.node_number)]
        self.traffic_space = (self.traffic_space - np.mean(self.traffic_space)) / self.node_number

        self.act_hist = np.array([self.action_space[1]]*self.node_number)

        self.setup_network()
        self.node_object = {}
        for i in range(0, self.node_number + self.server_number):
            self.node_object[str(i)] = Node(i)
        self.mpc_candidate()
        self.init_node()
        self.get_load()
        self.init_mpc()

    def init_node(self):
        for i in range(1, self.node_number+self.server_number):
            self.node_object[str(i)].reset()
            #self.node_object[str(i)].sampling_time = 0
            #self.node_object[str(i)].sampling_time = random.randint(0, 5)
            self.node_object[str(i)].trace_time = 0
            self.traffic.append(0)
        #self.node_object[str(10)].sampling_time = 0

    def mpc_candidate(self):
        dtype = [('nei', int), ('length', int), ('tosink', int)]
        for i in range(1, self.node_number+self.server_number):
            values = []
            for j in range(1, self.node_number+self.server_number):
                if i != j:
                    route1 = self.find_route(i, j)
                    route2 = self.find_route(j, 0)
                    values.append((j, len(route1), len(route2)))

            #a = np.array(values, dtype=dtype)
            #sorted = np.sort(a, order='length')
            values.sort(key=lambda x:(x[1], x[2]))

            #print("node:{}:{}".format(i, sorted[:3]))
            tmp = []
            for k in range(10):
                tmp.append(values[k][0])
            self.node_object[str(i)].neighbor = tmp
            print("node:{}:{}".format(i, tmp))

    def init_mpc(self):
        # Find a one-hop neighbor for each node to initialize MPC connectivity matrix
        self.MPC_connectivity = np.zeros((self.node_number+self.server_number, self.node_number+self.server_number),
                                         dtype=int)
        for i in range(1, self.node_number+self.server_number):
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
        for i in range(1, self.node_number+self.server_number):
            if self.node_object[str(i)].trace_time == self.time:
                num_mpc = list(self.MPC_connectivity[i][1:]).count(1)   # Number of sharing nodes
                type_packet = 1                                     # 0:to the sink; 1:MPC_ping 2:MPC_pong
                rec_mpc = 0                                         # Number of received pong messages
                s = i                                               # Source node id
                st = self.time                                      # Start time
                et = -1                                             # End time
                for j in range(num_mpc):
                    t = list(self.MPC_connectivity[i]).index(1, j)  # Target node
                    route = self.find_route(s, t)
                    packet = [[num_mpc, type_packet, rec_mpc], [s, t, st, et], [route, 1]]
                    self.node_object[str(i)].send_queue.append(packet)
                self.node_object[str(i)].trace_time += self.trace_period

                route = self.find_route(s, 0)
                packet = [[num_mpc, type_packet, rec_mpc], [s, 0, st, et], [route, 1]]
                self.node_object[str(i)].send_queue.append(packet)      # Send to sink

        for i in range(1, self.node_number+self.server_number):
            if self.node_object[str(i)].sampling_time == self.time:
                num_mpc = 0                                         # Number of sharing nodes
                type_packet = 0                                     # 0:to the sink; 1:MPC_ping 2:MPC_pong
                rec_mpc = 0                                         # Number of received pong messages
                s = i                                               # Source node id
                st = self.time                                      # Start time
                et = -1                                             # End time
                if num_mpc == 0:
                    type_packet = 0
                    t = 0
                    route = self.find_route(s, t)
                    packet = [[num_mpc, type_packet, rec_mpc], [s, t, st, et], [route, 1]]
                    self.node_object[str(i)].send_queue.append(packet)
                self.node_object[str(i)].sampling_time += self.sampling_period

    def parse_send_queue(self):
        # [ [MPC_num, TYPE, R], [s, t, ST, ET], [[ROUTE], i]] ]
        for i in range(1, self.node_number+self.server_number):
            # Check if itself is in sleep mode
            if self.node_object[str(i)].sleep_mode != 1:
                for tmp_send in self.node_object[str(i)].send_queue[:]:
                    #print(tmp_send)
                    index = tmp_send[2][1]
                    target_id = tmp_send[2][0][index]
                    # If the target node is the sink, receive the packet
                    if target_id == 0:
                        self.node_object[str(0)].receive_queue.append(tmp_send)
                        self.node_object[str(i)].send_queue.remove(tmp_send)
                        #self.node_object[str(i)].sleep_mode = 1
                        self.node_object[str(i)].node_energy -= self.transmit_energy
                    else:
                        # If the target node is others, check whether the target node
                        if self.node_object[str(target_id)].sleep_mode != 1:
                            self.node_object[str(i)].send_queue.remove(tmp_send)
                            tmp_send[2][1] += 1
                            self.node_object[str(target_id)].receive_queue.append(tmp_send)
                            self.node_object[str(target_id)].sleep_mode = 1
                            self.node_object[str(target_id)].node_energy -= self.transmit_energy
                            self.node_object[str(i)].sleep_mode = 1
                            self.node_object[str(i)].node_energy -= self.transmit_energy
                            break

    def parse_received_queue(self):
        # [ [MPC_num, TYPE, R], [s, t, ST, ET], [[ROUTE}, i]] ]
        # Parse received queue
        for i in range(1, self.node_number + self.server_number):
            for tmp_receive in self.node_object[str(i)].receive_queue[:]:
                self.node_object[str(i)].receive_queue.remove(tmp_receive)
                # Check if itself is the target node, if not append the packet to the send queue
                if tmp_receive[1][1] != self.node_object[str(i)].node_id:
                    self.node_object[str(i)].send_queue.append(tmp_receive)
                else:
                    # Check the type of packet, generate pong packet and source to the sink if TYPE = 1
                    if tmp_receive[0][1] == 1:
                        tmp_send = copy.deepcopy(tmp_receive)
                        tmp_send[1][1] = 0
                        tmp_send[2][0] = self.find_route(i, 0)
                        tmp_send[2][1] = 1
                        self.node_object[str(i)].send_queue.append(tmp_send)

        for i in range(1, self.node_number+self.server_number):
            self.node_object[str(i)].sleep_mode = 0

    def performance_analysis(self):
        self.sink_node_analysis()
        self.network_analysis()

    def sink_node_analysis(self):
        # Calculate data throughput
        self.raw_throughput.append(len(self.node_object[str(0)].receive_queue))
        self.fine_throughput.append(0)
        self.mean_delay.append(0)

        for tmp_receive in self.node_object[str(0)].receive_queue[:]:
            # [ [MPC_num, TYPE, R], [s, t, ST, ET], [[ROUTE}, i]] ]
            self.node_object[str(0)].receive_queue.remove(tmp_receive)
            has_rece = 0
            for tmp_pend in self.node_object[str(0)].pending_queue:
                if tmp_pend[1][0] == tmp_receive[1][0] and tmp_pend[1][2] == tmp_receive[1][2]:
                    tmp_pend[0][2] += 1
                    has_rece = 1
                    break
            if not has_rece:
                tmp_receive[0][2] = 1
                self.node_object[str(0)].pending_queue.append(tmp_receive)

        for tmp_pend in self.node_object[str(0)].pending_queue[:]:
            # [ [MPC_num, TYPE, R], [s, t, ST, ET], [[ROUTE}, i]] ]
            if tmp_pend[0][0] == tmp_pend[0][2] - 1:
                self.fine_throughput[-1] += 1
                delay = self.time - tmp_pend[1][2] + 1
                #if tmp_pend[0][0] != 0:
                    #self.transmit_delay[tmp_pend[1][0]-1] += delay
                self.transmit_delay[tmp_pend[1][0] - 1] += delay
                self.node_object[str(0)].pending_queue.remove(tmp_pend)
        #self.mean_delay[-1] = np.mean(self.transmit_delay) if self.transmit_delay else 0
        self.mean_delay[-1] = np.sum(self.transmit_delay)/self.node_number

    def network_analysis(self):
        for i in range(1, self.node_number+self.server_number):
            consume = self.node_object[str(0)].node_energy - self.node_object[str(i)].node_energy
            self.traffic[i-1] = consume/self.transmit_energy

    def _debug(self):
        print("===============" + "TIME: " + str(self.time) + "==================")
        #print(self.raw_throughput, self.fine_throughput, self.transmit_delay)
        for i in range(1):
        #for i in range(0, self.node_number+self.server_number):
            print("Node " + str(i) +":")
            print(self.node_object[str(i)].receive_queue)
            print(self.node_object[str(i)].send_queue)
            print(self.node_object[str(i)].pending_queue)

    def _plot(self):
        plt.clf()
        graph1 = plt.subplot(2, 2, 1)
        graph1.set_title('Data Throughput')
        graph1.set_xlabel('Time', fontsize=10)
        graph1.set_ylabel('Packets Num', fontsize=10)
        x1 = [n for n in range(self.time)]
        plt.plot(x1, self.raw_throughput, 'g-')

        graph2 = plt.subplot(2, 2, 2)
        graph2.set_title("Packet Delay")
        graph2.set_xlabel('Time', fontsize=10)
        graph2.set_ylabel('Average Packets Delay', fontsize=10)
        plt.plot(x1, self.mean_delay, 'r-')

        graph3 = plt.subplot(2, 2, 3, ylim=(0, 200), xticks=np.arange(0, self.node_number+1, 1))
        graph3.set_title("Node traffic")
        graph3.set_xlabel('Node ID', fontsize=10)
        graph3.set_ylabel('Num packet to be sent', fontsize=10)
        x3 = [n for n in range(self.server_number, self.node_number+self.server_number)]
        plt.bar(x3, self.traffic, color="blue")

        graph4 = plt.subplot(2, 2, 4, ylim=(0, 5000), xticks=np.arange(0, self.node_number+1, 1))
        graph4.set_title("Node energy")
        graph4.set_xlabel('Node ID', fontsize=10)
        graph4.set_ylabel('Available energy', fontsize=10)
        x4 = [n for n in range(self.server_number, self.node_number+self.server_number)]
        y4 = [0] * self.node_number
        for i in range(self.server_number, self.node_number+self.server_number):
            y4[i-self.server_number] = self.node_object[str(i)].node_energy
        plt.bar(x4, y4, color="green")

        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.pause(0.1)

    def operate(self, time_scope=10):
        """
        Simulate network transmission within given time steps
        """
        #plt.figure(2, figsize=(9, 9))
        while self.time < time_scope:
            self.generate_source()
            self.parse_send_queue()
            self.parse_received_queue()
            self.performance_analysis()
            self.time += 1

            #self._debug()
            #self._plot()

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
        self.raw_throughput.clear()
        self.fine_throughput.clear()
        self.transmit_delay = [0]*self.node_number
        #self.transmit_delay.clear()
        self.mean_delay.clear()
        self.traffic.clear()
        self.act_hist.fill(self.action_space[0])

        self.init_node()
        self.init_mpc()

    def step(self, node_id, action):
        self.act_action(node_id, action)
        self.time = 0
        self.raw_throughput.clear()
        self.fine_throughput.clear()
        #self.transmit_delay.clear()
        self.transmit_delay = [0]*self.node_number
        self.mean_delay.clear()
        self.traffic.clear()
        self.init_node()
        self.operate(2000)
        reward = self.get_reward(node_id)
        #state_ = self.observer(node_id)
        state_ = self.observer(node_id%self.node_number+1)
        return state_, reward

    def observer(self, node_id):
        #state = np.reshape(self.MPC_connectivity[node_id][1:], [1, self.node_number])[0]
        #state = np.reshape(self.MPC_connectivity[1:, 1:]-0.5, [1, self.node_number**2])[0]
        '''
        state = np.zeros((self.node_number, 4)) - 0.5
        for i in range(0, self.node_number):
            num_sharing_node = np.count_nonzero(self.MPC_connectivity[i+1])
            state[i][:num_sharing_node] = 0.5
        state = np.reshape(state, [1, self.node_number*4])[0]
        #state = np.append(state, self.node_object[str(node_id)].neighbor[index])
        #state = np.append(state, self.act_hist[node_id-1])
        state = np.append(state, self.act_hist)
        '''
        #state = np.reshape(self.MPC_connectivity[1:, 1:]-0.5, [1, self.node_number**2])[0]

        state = list()
        privacy = list()
        for i in range(1, self.node_number+self.server_number):
            nei = self.node_object[str(i)].neighbor
            num_sharing_node = np.count_nonzero(self.MPC_connectivity[i])
            privacy.append(num_sharing_node / len(nei))

        state = np.append(state, privacy)
        state = np.append(state, self.act_hist)
        state = np.append(state, self.node_id_sapce[node_id-1])
        state = state[np.newaxis]

        return state

    def act_action(self, node_id, action):
        self.act_hist[node_id-1] = self.action_space[action]
        connectivity = self.MPC_connectivity[node_id]

        nei = self.node_object[str(node_id)].neighbor
        connected = nei[:2*(action+1)]
        dis_connected = nei[2*(action+1):]

        for n in connected:
            connectivity[n] = 1
        for n in dis_connected:
            connectivity[n] = 0

    def get_reward(self, node_id):
        privacy = list()
        delay = self.transmit_delay.copy()
        load = self.traffic.copy()
        eff = (np.mean(load) - self.avr_load)/self.avr_load
        print("eff {}".format(eff))
        for i in range(1, self.node_number+self.server_number):
            nei = self.node_object[str(i)].neighbor
            num_sharing_node = np.count_nonzero(self.MPC_connectivity[i])
            hop = self.node_object[str(i)].hop_dis
            privacy.append(num_sharing_node / (len(nei)))
        #a = np.sum(privacy)/16

        #b = 1 -abs(load-np.mean(load))/np.mean(load)
        inc_load = [m-n for m,n in zip(load, self.init_load)]

        a = [m*n/o for m,n,o in zip(privacy, self.init_delay, inc_load)]
        b = 1 - abs(load - np.mean(load)) / np.mean(load)
        #print(b)
        c = 1 - abs(delay - np.mean(delay)) / np.mean(delay)
        d = [m/n for m,n in zip(self.init_load, inc_load)]
        #b = 1- (load - np.mean(load)) / (np.max(load) - np.min(load))
        #b = 1 -(load-np.mean(load))/np.mean(load)
        reward = np.dot(b, privacy)/16
        #load = (load-np.min(load))/(np.max(load)-np.min(load))
        #b = np.sum(load)/16
        #load = self.transmit_delay.copy()
        #load = (load - np.mean(load)) / (np.max(load) - np.min(load))
        #load = (load - np.mean(load)) / np.std(load)
        #load = -1*load
        #reward = np.dot(privacy, load)/16

        '''
        reward = 0
        for i in range(1, self.node_number+1):
            reward += -1*self.transmit_delay[i-1] / self.node_object[str(i)].hop_dis
        reward /= self.node_number
        reward -= self.average
        #reward += 5
        #reward = (-1*self.mean_delay[-1] + 5)
        #reward = -1*self.mean_delay[-1]
        '''
        return reward

    def get_average(self):
        self.average = 0
        for i in range(3):
            self.reset_network()
            for j in range(1, self.node_number + 1):
                self.act_action(j, i)
            self.operate(self.batch_period)

            reward = 0
            for k in range(1, self.node_number+1):
                reward += -1*self.transmit_delay[k-1] / self.node_object[str(k)].hop_dis
            reward /= self.node_number
            #reward = -1 * self.mean_delay[-1]
            print("{}: {}".format(i, reward))
            self.average += reward

        self.average /= 3
        print("Average delay: {}".format(self.average))

    def get_load(self):
        self.MPC_connectivity.fill(0)
        self.operate(100)
        self.init_load = self.traffic.copy()
        self.avr_load = np.mean(self.init_load)
        self.init_delay = self.transmit_delay.copy()

if __name__ == "__main__":
    """
    net = Network(False)
    net.setup_network()
    print(net.state_xcor)
    print(net.state_ycor)
    net.draw_network()
    """
    net = NetworkMPC(True)
    plt.ion()
    #net.store_network()
    net.draw_network()
    print(net.init_load)
    print(net.init_delay)

    for i in range(0, net.node_number):
        net.act_action(i+1, 0)
    state_, reward = net.step(1, 0)
    print("---")
    print(reward)
    print(net.traffic)
    print(net.transmit_delay)

    for i in range(0, net.node_number):
        net.act_action(i+1, 3)
    state_, reward = net.step(1, 3)
    print("---")
    print(reward)
    print(net.traffic)
    print(net.transmit_delay)

    for i in range(0, net.node_number):
        net.act_action(i+1, 0)
    net.act_action(1, 3)
    net.act_action(4, 3)
    net.act_action(13, 3)
    net.act_action(16, 3)
    state_, reward = net.step(1, 3)
    print("---")
    print(reward)
    print(net.traffic)
    print(net.transmit_delay)
    '''
    state_, reward = net.step(1, 4)
    print(net.MPC_connectivity)
    print(net.traffic)
    '''
    plt.ioff()
    plt.show()
