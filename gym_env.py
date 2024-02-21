from typing import Any

import gymnasium as gym
import numpy as np
import os
import networkx as nx

DATA_DIR = './resources'

class Topology():
    def __init__(self, data_dir='./resources/', topology='Abilene'):
        # topology = 'topology_0'
        self.topology_file = data_dir + 'topologies/' + topology
        self.shortest_paths_file = f'./resources/shortest_path/{topology}'

        self.DG = nx.DiGraph()

        self.load_topology()
        self.calculate_paths()

    def load_topology(self):
        print('[*] Loading topology...', self.topology_file)

        f = open(self.topology_file, 'r')
        header = f.readline()
        self.num_nodes = int(header[header.find(':') + 2:header.find(',')])
        self.num_links = int(header[header.find(':', 10) + 2:])
        f.readline()
        self.link_idx_to_sd = {}
        self.link_sd_to_idx = {}
        self.link_capacities = np.empty((self.num_links))
        self.link_weights = np.empty((self.num_links))
        for line in f:
            # print(line)
            link = line.split(',')
            # print(link)
            i, s, d, w, c = link
            self.link_idx_to_sd[int(i)] = (int(s), int(d))
            self.link_sd_to_idx[(int(s), int(d))] = int(i)
            self.link_capacities[int(i)] = float(c)
            self.link_weights[int(i)] = int(w)
            self.DG.add_weighted_edges_from([(int(s), int(d), int(w))])

        assert len(self.DG.nodes()) == self.num_nodes and len(
            self.DG.edges()) == self.num_links, f'DG.nodes: {len(self.DG.nodes())}, num_nodes : {self.num_nodes}, \n edges: {len(self.DG.edges())} == {self.num_links}'
        f.close()
        print('nodes: %d, links: %d\n' % (self.num_nodes, self.num_links))

        """plt.figure()
        print(self.DG)
        nx.draw(self.DG, pos=nx.circular_layout(self.DG), with_labels=True)
        plt.show()"""

    def get_topology(self):
        return nx.to_numpy_matrix(self.DG)

    def calculate_paths(self):
        print(f"Calculating paths for: {self.shortest_paths_file}")
        self.pair_idx_to_sd = []
        self.pair_sd_to_idx = {}
        # Shortest paths
        self.shortest_paths = []
        # self.shortest_paths_file = f'./resources/shortest_path/topology_0'
        if os.path.exists(self.shortest_paths_file):
            #print('[*] Loading shortest paths...', self.shortest_paths_file)
            f = open(self.shortest_paths_file, 'r')
            self.num_pairs = 0
            for line in f:
                sd = line[:line.find(':')]
                s = int(sd[:sd.find('-')])
                d = int(sd[sd.find('>') + 1:])
                self.pair_idx_to_sd.append((s, d))
                self.pair_sd_to_idx[(s, d)] = self.num_pairs
                self.num_pairs += 1
                self.shortest_paths.append([])
                paths = line[line.find(':') + 1:].strip()[1:-1]
                while paths != '':
                    idx = paths.find(']')
                    path = paths[1:idx]
                    node_path = np.array(path.split(',')).astype(np.int16)
                    assert node_path.size == np.unique(node_path).size
                    self.shortest_paths[-1].append(node_path)
                    paths = paths[idx + 3:]
        else:
            #print('[!] Calculating shortest paths...')
            f = open(self.shortest_paths_file, 'w+')
            self.num_pairs = 0
            for s in range(self.num_nodes):
                for d in range(self.num_nodes):
                    if s != d:
                        # print(f'{s}:{d}')
                        self.pair_idx_to_sd.append((s, d))
                        self.pair_sd_to_idx[(s, d)] = self.num_pairs
                        self.num_pairs += 1
                        self.shortest_paths.append(list(nx.all_shortest_paths(self.DG, s, d, weight='weight')))
                        line = str(s) + '->' + str(d) + ': ' + str(self.shortest_paths[-1])
                        f.writelines(line + '\n')

        assert self.num_pairs == self.num_nodes * (self.num_nodes - 1), f'{self.num_pairs} {self.num_nodes}'
        f.close()

        print('pairs: %d, nodes: %d, links: %d\n' \
              % (self.num_pairs, self.num_nodes, self.num_links))


class Traffic(object):
    def __init__(self, config, num_nodes, data_dir='./resources/', topology='Abilene', is_training=False):
        if is_training:
            self.traffic_file = './resources/tms/' + topology + '_TM'

        else:
            self.traffic_file = './resources/tms/AbileneTM2'
        # self.traffic_file = './resources/tms/AbileneTM'

        print(self.traffic_file)
        # print("Is training")
        # print(is_training)
        # print(self.traffic_file)
        self.num_nodes = num_nodes
        self.load_traffic(config)

    def load_traffic(self, config):
        assert os.path.exists(self.traffic_file)
        print('[*] Loading traffic matrices...', self.traffic_file)

        f = open(self.traffic_file, 'r')
        traffic_matrices = []
        for line in f:
            volumes = line.strip().split(' ')
            total_volume_cnt = len(volumes)
            assert total_volume_cnt == self.num_nodes * self.num_nodes
            matrix = np.zeros((self.num_nodes, self.num_nodes))
            for v in range(total_volume_cnt):
                i = int(v / self.num_nodes)
                j = v % self.num_nodes
                if i != j:
                    matrix[i][j] = float(volumes[v])
            # print(matrix + '\n')
            traffic_matrices.append(matrix)

        f.close()
        self.traffic_matrices = np.array(traffic_matrices)

        tms_shape = self.traffic_matrices.shape
        self.tm_cnt = tms_shape[0]
        #print('Traffic matrices dims: [%d, %d, %d]\n' % (tms_shape[0], tms_shape[1], tms_shape[2]))


class GameEnv(gym.Env):


    def __init__(self, config, topology, is_training = True):



    def step(self, action):
        pass


    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        pass

    def render(self):
        raise NotImplementedError

    def close(self):
        pass


def set_seed(seed: int):
    np.random.seed(seed)



