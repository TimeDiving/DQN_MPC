import os
import numpy as np
import random
import copy
import json
import math
import networkx as nx
import matplotlib.pyplot as plt


class Network(object):
    """
    Establish wireless mesh network,
    Define node parameters and mesh topology
    """
    def __init__(self):
        self.init_states()
        self.setup_network()

    def init_states(self):
        #===== Node parameters ======
        self.grid_distance = 10                     # The network is divided into numerous of grids, each grid contains a node
        self.grid_xcor_node_number = 10              # Number of girds/nodes in x axis
        self.grid_ycor_node_number = 10              # Number of grids/nodes in y axis
        self.node_number = self.grid_xcor_node_number*self.grid_ycor_node_number    # Total nodes num except the sink
        self.server_number = 1                      # NUm of sink node

        self.transmit_range = 13
        self.transmit_energy = 10
        self.min_distance_between_nodes = 8
        self.deploy_range_x = (self.grid_xcor_node_number - 1) * self.grid_distance     # Range of network in x axis
        self.deploy_range_y = (self.grid_ycor_node_number - 1) * self.grid_distance     # Range of network in y axis

        #===== Network initialization parameters =====
        self.max_find_good_position_time = 3

        #===== Network positions storage =====
        self.state_G = nx.Graph()
        self.state_G_no_sink = nx.Graph()
        self.state_xcor = []
        self.state_ycor = []
        self.state_link = []

    def setup_network(self):
        # Initiate node positions
        self.set_network_topology()
        # Ensure the graph is fully connected
        while self.all_nodes_connected() == False:
            self.set_network_topology()

    def all_nodes_connected(self):
        for i in range(0, self.node_number + self.server_number):
            for j in range(0, self.node_number + self.server_number):
                check = nx.has_path(self.state_G, i, j)
                if check == False:
                    return False
        return True

    def set_network_topology(self):
        self.state_xcor = []
        self.state_ycor = []

        self.scatter_node_random_position()
        self.set_network_connectivity()

    def scatter_node_random_position(self):
        self.state_xcor.append(self.deploy_range_x/2)
        self.state_ycor.append(self.deploy_range_y/2)
        for i in range(0, self.grid_xcor_node_number):
            for j in range(0, self.grid_ycor_node_number):
                self.state_xcor.append(0)
                self.state_ycor.append(0)
                for k in range(0, self.max_find_good_position_time):
                    self.state_xcor[-1] = random.random() * self.grid_distance + i*self.grid_distance
                    self.state_ycor[-1] = random.random() * self.grid_distance + j*self.grid_distance
                    good_position = self.check_neighbor_distance(i+1)
                    if good_position == 1:
                        break

    def check_neighbor_distance(self, node_id):
        good_position = 1
        ax = self.state_xcor[node_id]
        ay = self.state_ycor[node_id]
        for j in range(1, len(self.state_xcor)-1):
            bx = self.state_xcor[j]
            by = self.state_ycor[j]
            distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            if distance < self.min_distance_between_nodes:
                good_position = 0
        return good_position

    def set_network_connectivity(self):
        self.state_link = []
        for i in range(0, self.node_number + self.server_number):
            node_link = []
            for j in range(0, self.node_number + self.server_number):
                if i!=j and ((self.state_xcor[i]-self.state_xcor[j])**2 + (self.state_ycor[i]-self.state_ycor[j])**2)**0.5 <= self.transmit_range:
                    node_link.append(1)
                else:
                    node_link.append(0)
            self.state_link.append(node_link)
        self.set_graph()

    def set_graph(self):
        self.state_G = nx.Graph()
        for i in range(0, self.node_number + self.server_number):
            self.state_G.add_node(i, pos=(self.state_xcor[i], self.state_ycor[i]))
        for i in range(0, self.node_number + self.server_number):
            for j in range(i, self.node_number + self.server_number):
                if self.state_link[i][j] == 1 and self.state_link[j][i] == 1:
                    self.state_G.add_edge(i, j)
        self.state_G_no_sink = self.state_G.copy()
        self.state_G_no_sink.remove_node(0)


    def draw_network(self):
        labeldict = {}
        for i in range(0, self.node_number + self.server_number):
            #labeldict[i] = str(i) " | " + str(self.state_cluster_id[i])
            labeldict[i] = str(i)

        #pos = nx.kamada_kawai_layout(self.state_G)
        pos = nx.get_node_attributes(self.state_G, 'pos')
        #print(pos)
        nx.draw(self.state_G, pos, with_labels=True, cmap=plt.get_cmap('Accent'),
                node_color='g', node_size=100)
        plt.show()

    def store_network(self):
        #nx.write_gml(self.state_G, "my_network")
        np.save("x_cor.npy", np.array(self.state_xcor))
        np.save("y_cor.npy", np.array(self.state_ycor))
        np.save("link.npy", np.array(self.state_link))

    def reuse_network(self):
        #self.state_G = nx.Graph()
        #self.state_G = nx.read_gml("my_network")
        self.state_xcor = np.load("x_cor.npy").tolist()
        self.state_ycor = np.load("y_cor.npy").tolist()
        self.state_link = np.load("link.npy").tolist()
        self.set_graph()

class Node(object):
    def __init__(self, id):
        self.node_id = id
        self.node_energy = 5000

        self.receive_queue= []
        self.send_queue = []
        self.pending_queue = []
        self.sleep_mode = 0

        self.sampling_time = -1

class MPC_network(Network):
    def __init__(self, reuse=False):
        Network.__init__(self)
        #===== Workload parameters =====
        self.time = 0
        self.reuse = reuse
        self.sampling_period = 20
        self.MPC_connectivity = np.zeros((self.node_number+self.server_number, self.node_number+self.server_number), dtype=int)

        #===== Performacne analysis parameters =====
        self.raw_throughput = []
        self.fine_throughput = []
        self.transmit_delay = []
        self.mean_delay = []

        if self.reuse:
            self.reuse_network()
        self.init_node()
        self.init_MPC()

    def init_node(self):
        self.node_object = {}
        for i in range(0, self.node_number + self.server_number):
            self.node_object[str(i)]= Node(i)

        for i in range(1, self.node_number+self.server_number):
            self.node_object[str(i)].sampling_time = random.randint(0, 10)
        #for i in range(1, 4):
            #self.node_object[str(i)].sampling_time = 0

    def init_MPC(self):
        # Find a one-hop neighbor for each node to initialize MPC connectivity matrix
        if not self.reuse:
            for i in range(1, self.node_number+self.server_number):
                nei = [n for n in self.state_G.neighbors(i)]
                if 0 in nei: nei.remove(0)
                #print("Node " + str(i))
                #print(nei)
                rand = random.randint(0, len(nei)-1)
                self.MPC_connectivity[i, nei[rand]] = 1
            #print(self.MPC_connectivity)

    def find_route(self, s, t):
        if t == 0:
            check = nx.has_path(self.state_G, source=s, target=t)
            if check == True:
                path = nx.dijkstra_path(self.state_G, source=s, target=t)
            else:
                path = []
        else:
            check = nx.has_path(self.state_G_no_sink, source=s, target=t)
            if check == True:
                path = nx.dijkstra_path(self.state_G_no_sink, source=s, target=t)
            else:
                path = []
            #print('The path is: ' + str(s) + '-' + str(t) + ' = ' + str(path))
        return path

    def generate_source(self):
        # Generate source
        # [ [MPC_num, TYPE, R], [s, t, ST, ET], [[ROUTE}, i]] ]
        for i in range(1, self.node_number+self.server_number):
            if self.node_object[str(i)].sampling_time == self.time:
                MPC_num = list(self.MPC_connectivity[i]).count(1)   # Number of sharing nodes
                type = 1                                            # 0:to the sink; 1:MPC_ping 2:MPC_pong
                MPC_rec =  0                                        # Number of received pong messages
                s = i                                               # Source node id
                st = self.time                                      # Start time
                et = -1                                             # End time
                for j in range(MPC_num):
                    t = list(self.MPC_connectivity[i]).index(1, j)  # Target node
                    route = self.find_route(s, t)
                    packet = [[MPC_num, type, MPC_rec], [s, t, st, et], [route, 1]]
                    self.node_object[str(i)].send_queue.append(packet)
                self.node_object[str(i)].sampling_time += self.sampling_period
                #print(self.node_object[str(i)].send_queue)

    def parse_send_queue(self):
        for i in range(1, self.node_number+self.server_number):
            # Check if itself is in sleep mode
            if self.node_object[str(i)].sleep_mode != 1:
                for tmp_send in self.node_object[str(i)].send_queue[:]:
                    # If the target node is the sink, receive the packet
                    if tmp_send[1][1] == 0:
                        self.node_object[str(0)].receive_queue.append(tmp_send)
                        self.node_object[str(i)].send_queue.remove(tmp_send)
                        self.node_object[str(i)].sleep_mode = 1
                        self.node_object[str(i)].node_energy -= self.transmit_energy
                        break
                    else:
                        index = tmp_send[2][1]
                        target_id = tmp_send[2][0][index]
                        # If the target node is others, check whether the target node
                        if self.node_object[str(target_id)].sleep_mode != 1:
                            self.node_object[str(i)].send_queue.remove(tmp_send)
                            if tmp_send[0][1] == 1:
                                tmp_send[2][1] += 1
                            else:
                                tmp_send[2][1] -= 1
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
                    tmp_receive[2][1] += 1
                    self.node_object[str(i)].send_queue.append(tmp_receive)
                else:
                    # Check the type of packet, generate pong packet and source to the sink if TYPE = 1
                    if tmp_receive[0][1] == 1:
                        tmp_send = copy.deepcopy(tmp_receive)
                        tmp_send[0][1] = 0
                        tmp_send[1][1] = 0
                        tmp_send[2][0] = self.find_route(i, 0)
                        tmp_send[2][1] = 1
                        self.node_object[str(i)].send_queue.append(tmp_send)

                        tmp_receive[0][1] = 2
                        tmp_receive[2][1] = -2
                        tmp_receive[1][0], tmp_receive[1][1] = tmp_receive[1][1], tmp_receive[1][0]
                        self.node_object[str(i)].send_queue.append(tmp_receive)
                    elif tmp_receive[0][1] == 2:
                        # Check the type of packet, if TYPE = 2
                        has_rece = 0
                        for tmp_pend in self.node_object[str(i)].pending_queue:
                            if tmp_receive[2][0] == tmp_pend[2][0]:
                                tmp_pend[0][2] += 1
                                has_rece = 1
                                break
                        if not has_rece:
                            tmp_receive[0][2] = 1
                            self.node_object[str(i)].pending_queue.append(tmp_receive)

            for tmp_pend in self.node_object[str(i)].pending_queue[:]:
                if tmp_pend[0][2] == tmp_pend[0][0]:
                    self.node_object[str(i)].pending_queue.remove(tmp_pend)
                    tmp_pend[0][1] = 0
                    tmp_pend[0][2] = 0
                    tmp_pend[1][0] = i
                    tmp_pend[1][1] = 0
                    tmp_pend[2][0] = self.find_route(i, 0)
                    tmp_pend[2][1] = 1
                    self.node_object[str(i)].send_queue.append(tmp_pend)

        for i in range(1, self.node_number+self.server_number):
            self.node_object[str(i)].sleep_mode = 0

    def performance_analysis(self):
        self.sink_node_analysis()

    def sink_node_analysis(self):
        # Calculate data throughput
        self.raw_throughput.append(len(self.node_object[str(0)].receive_queue))
        self.fine_throughput.append(0)
        self.mean_delay.append(0)

        for tmp_receive in self.node_object[str(0)].receive_queue[:]:
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
            if tmp_pend[0][0] == tmp_pend[0][2] - 1:
                self.fine_throughput[-1] += 1
                delay = self.time - tmp_pend[1][2]
                self.transmit_delay.append(delay)
                self.node_object[str(0)].pending_queue.remove(tmp_pend)
        self.mean_delay[-1] = np.mean(self.transmit_delay) if self.transmit_delay else 0

    def _debug(self):
        print("===============" + "TIME: "+ str(self.time) + "==================")
        print(self.raw_throughput, self.fine_throughput, self.transmit_delay)
        '''for i in range(0, self.node_number+self.server_number):
            self.node_object[str(i)].sleep_mode = 0
            print("Node " + str(i) +":")
            print(self.node_object[str(i)].receive_queue)
            print(self.node_object[str(i)].send_queue)
            print(self.node_object[str(i)].pending_queue)
            '''
    def _plot(self):
        plt.ion()
        plt.clf()
        graph1= plt.subplot(2, 1, 1)
        graph1.set_title('Data Throughput')
        graph1.set_xlabel('Time', fontsize=10)
        graph1.set_ylabel('Packets Num', fontsize=10)
        ax = [n for n in range(self.time)]
        plt.plot(ax, self.raw_throughput, 'g-')

        graph2 = plt.subplot(2, 1, 2)
        graph2.set_title("Packet Delay")
        graph2.set_xlabel('Time', fontsize=10)
        graph2.set_ylabel('Average Packets Delay', fontsize=10)
        plt.plot(ax, self.mean_delay, 'r-')

        plt.pause(0.1)


    def operate(self, time_scope):
        '''
        Simulate network transmission within given time steps
        '''
        while self.time <= time_scope:
            self.generate_source()
            self.parse_send_queue()
            self.parse_received_queue()
            self.performance_analysis()
            self.time += 1

            #self._debug()
            self._plot()
        plt.ioff()
        plt.show()

    def store_network(self):
        #nx.write_gml(self.state_G, "my_network")
        np.save("x_cor.npy", np.array(self.state_xcor))
        np.save("y_cor.npy", np.array(self.state_ycor))
        np.save("link.npy", np.array(self.state_link))
        np.save("mpc_connectivity.npy", np.array(self.MPC_connectivity))

    def reuse_network(self):
        #self.state_G = nx.Graph()
        #self.state_G = nx.read_gml("my_network")
        self.state_xcor = np.load("x_cor.npy").tolist()
        self.state_ycor = np.load("y_cor.npy").tolist()
        self.state_link = np.load("link.npy").tolist()
        self.MPC_connectivity = np.load("mpc_connectivity.npy").tolist()
        self.set_graph()

if __name__ == "__main__":
    #net = MPC_network(reuse=True)
    #net.operate(50)
    #net.draw_network()
    #net.store_network()
    net = Network()
    net.draw_network()