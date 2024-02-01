from typing import Any

import gymnasium as gym
import numpy as np
import os
import networkx as nx
from env import Environment
from pulp import LpProblem, LpVariable, lpSum, LpStatus, GLPK

DATA_DIR = './resources'
OBJ_EPSILON = 1e-12


class Topology():
    def __init__(self, data_dir='./resources/', topology='Abilene'):
        # topology = 'topology_0'
        self.topology_file = data_dir + 'topologies/' + topology
        self.shortest_paths_file = f'./resources/shortest_path/{topology}'

        self.DG = nx.DiGraph()

        self.load_topology()
        self.calculate_paths()

    def load_topology(self):
        #print('[*] Loading topology...', self.topology_file)

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
        #print('nodes: %d, links: %d\n' % (self.num_nodes, self.num_links))

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
            print('[!] Calculating shortest paths...')
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
        print('Traffic matrices dims: [%d, %d, %d]\n' % (tms_shape[0], tms_shape[1], tms_shape[2]))


class GameEnv(gym.Env):
    seed = None
    random_state: np.random.RandomState = None
    tms: np.array = None

    def __init__(self, config, env: Environment, seed=None):
        self.set_seed(seed)
        self.env = env
        self.data_dir = env.data_dir
        self.num_links = env.num_links
        self.DG = env.topology.DG
        self.traffic_matrices = env.traffic_matrices
        self.indexes = np.arange(0,self.env.tm_cnt)
        self.tm_idx = 0
        self.action_dim = self.env.num_pairs
        self.max_moves = int(self.action_dim * (config.max_moves / 100.))
        self.get_ecmp_next_hops()
        self.baseline = {}
        self.pair_idx_to_sd = env.pair_idx_to_sd
        self.link_sd_to_idx = env.link_sd_to_idx
        self.link_capacities = env.link_capacities
        self.shortest_paths_link = env.shortest_paths_link

        self.lp_pairs = [p for p in range(self.env.num_pairs)]
        self.lp_nodes = [n for n in range(self.env.num_nodes)]
        self.links = [e for e in range(self.num_links)]
        self.lp_links = [e for e in self.env.link_sd_to_idx]
        self.pair_links = [(pr, e[0], e[1]) for pr in self.lp_pairs for e in self.lp_links]

    def get_state(self):
        mat = nx.adjacency_matrix(self.DG).toarray()
        tm = self.traffic_matrices[self.indexes[self.tm_idx]]
        return mat, tm

    def update_baseline(self, reward):
        if self.indexes[self.tm_idx] in self.baseline:
            total_v, cnt = self.baseline[self.indexes[self.tm_idx]]

            total_v += reward
            cnt += 1

            self.baseline[self.indexes[self.tm_idx]] = (total_v, cnt)
        else:
            self.baseline[self.indexes[self.tm_idx]] = (reward, 1)

    def step(self, actions):
        reward = self.calculate_reward(actions)
        self.tm_idx += 1
        if self.tm_idx >= len(self.traffic_matrices):
            self.reset()
        state = self.get_state()
        return state[0],state[1], reward

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Starts the environment and returns a state"""
        self.shuffle()
        self.tm_idx = 0
        return self.get_state()

    def render(self):
        raise NotImplementedError

    def close(self):
        pass


    def set_seed(self, seed):
        self.seed = seed
        set_seed(seed)
        self.random_state = np.random.RandomState(seed=seed)

    def shuffle(self):
        self.random_state.shuffle(self.indexes)

    def calculate_reward(self, actions):
        mlu, _ = self.optimal_routing_mlu_critical_pairs(actions[0].tolist())
        ecmp_mlu = self.ecmp_mlu()

        crit_topk = self.get_critical_topK_flows()
        topk_mlu, _ = self.optimal_routing_mlu_critical_pairs(crit_topk)

        if mlu < topk_mlu:
            reward = 1 + (topk_mlu - mlu) / topk_mlu
        elif (abs(topk_mlu - ecmp_mlu) < 0.005):
            reward = 0
        else:
            reward = (1 - 2 * abs(mlu - topk_mlu) / abs(topk_mlu - ecmp_mlu))
        return reward

    def optimal_routing_mlu_critical_pairs(self, critical_pairs):
        """
        Calculate the Maximum Link Utilization (MLU) and routing solution for critical flow pairs in the network

        Optimally routes the traffic flow associated with critical flow pairs in the network's traffic matrix
        to minimize MLU while ensuring flow conservation.

        :param tm_idx (int): The index of the traffic matrix
        :param critical_pairs (list of int): A list of indices representing critical flow pairs to optimize routing for
        :return:
            - obj_r (float): The calculated MLU for the optimized routing
            - solution (dict) A dictionary representing the optimized routing solution
        """

        tm = self.traffic_matrices[self.indexes[self.tm_idx]]
        name = self.env.topology_name
        pairs = critical_pairs

        # If the flow is critical it gets inserted into demands
        demands = {}
        background_link_loads = np.zeros((self.num_links))
        for i in range(self.env.num_pairs):
            s, d = self.env.pair_idx_to_sd[i]
            # background link load
            if i not in critical_pairs:
                self.ecmp_next_hop_distribution(background_link_loads, tm[s][d], s, d)
            else:
                demands[i] = tm[s][d]

        # Initializes a Linear Programming problem ?
        model = LpProblem(name=f"routing_{name}")


        pair_links = [(pr, e[0], e[1]) for pr in pairs for e in self.lp_links]

        ratio = LpVariable.dicts(name=f"ratio_{name}", indexs=pair_links, lowBound=0, upBound=1)

        link_load = LpVariable.dicts(name=f"link_load_{name}", indexs=self.links)

        r = LpVariable(name=f"congestion_ratio_{name}")

        # Flow into source nodes = Flow out of source nodes
        #print()
        #print(ratio)
        #i = input("Waiting")
        for pr in pairs:
            #print(pr)
            model += (
                lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == self.env.pair_idx_to_sd[pr][0]]) - lpSum(
                    [ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == self.env.pair_idx_to_sd[pr][0]]) == -1,
                f"{name}_flow_conservation_constr1_{pr}")

        # Flow into destination node = Flow out of destination node
        for pr in pairs:
            model += (
                lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == self.pair_idx_to_sd[pr][1]]) - lpSum(
                    [ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == self.pair_idx_to_sd[pr][1]]) == 1,
                f"{name}_flow_conservation_constr2_{pr}")

        # Flow conservation for intermediate nodes.
        for pr in pairs:
            for n in self.lp_nodes:
                if n not in self.pair_idx_to_sd[pr]:
                    model += (lpSum([ratio[pr, e[0], e[1]] for e in self.lp_links if e[1] == n]) - lpSum(
                        [ratio[pr, e[0], e[1]] for e in self.lp_links if e[0] == n]) == 0,
                              f"{name}_flow_conservation_constr3_{pr}_{n}")

        # Adds constraints to the links / ensures that the capacity is not exceeded
        for e in self.lp_links:
            ei = self.link_sd_to_idx[e]
            model += (
                link_load[ei] == background_link_loads[ei] + lpSum(
                    [demands[pr] * ratio[pr, e[0], e[1]] for pr in pairs]),
                f"{name}_link_load_constr{ei}")
            model += (link_load[ei] <= self.env.link_capacities[ei] * r, f"{name}_congestion_ratio_constr{ei}")

        # Objective function, minimize r (Congestion Ratio)
        model += r + OBJ_EPSILON * lpSum([link_load[ei] for ei in self.links])

        # Solve the problem
        model.solve(solver=GLPK(msg=False))
        assert LpStatus[model.status] == 'Optimal', LpStatus[model.status]

        obj_r = r.value()
        solution = {}
        for k in ratio:
            solution[k] = ratio[k].value()

        return obj_r, solution

    def get_ecmp_next_hops(self):
        self.ecmp_next_hops = {}
        for src in range(self.env.num_nodes):
            for dst in range(self.env.num_nodes):
                if src == dst:
                    continue
                self.ecmp_next_hops[src, dst] = []
                for p in self.env.shortest_paths_node[self.env.pair_sd_to_idx[(src, dst)]]:
                    if p[1] not in self.ecmp_next_hops[src, dst]:
                        self.ecmp_next_hops[src, dst].append(p[1])

    def get_critical_topK_flows(self, critical_links=5):
        link_loads = self.ecmp_traffic_distribution()
        critical_link_indexes = np.argsort(-(link_loads / self.link_capacities))[:critical_links]

        cf_potential = []
        for pair_idx in range(self.env.num_pairs):
            for path in self.shortest_paths_link[pair_idx]:
                if len(set(path).intersection(critical_link_indexes)) > 0:
                    cf_potential.append(pair_idx)
                    break

        # print(cf_potential)
        assert len(cf_potential) >= self.max_moves, \
            ("cf_potential(%d) < max_move(%d), please increse critical_links(%d)" % (
            len(cf_potential), self.max_moves, critical_links))

        return self.get_topK_flows(cf_potential)

    def ecmp_next_hop_distribution(self, link_loads, demand, src, dst):
        if src == dst:
            return

        ecmp_next_hops = self.ecmp_next_hops[src, dst]

        next_hops_cnt = len(ecmp_next_hops)
        # if next_hops_cnt > 1:
        # print(self.shortest_paths_node[self.pair_sd_to_idx[(src, dst)]])

        ecmp_demand = demand / next_hops_cnt
        for np in ecmp_next_hops:
            link_loads[self.env.link_sd_to_idx[(src, np)]] += ecmp_demand
            self.ecmp_next_hop_distribution(link_loads, ecmp_demand, np, dst)

    def ecmp_traffic_distribution(self):
        link_loads = np.zeros((self.num_links))
        tm = self.traffic_matrices[self.indexes[self.tm_idx]]
        for pair_idx in range(self.env.num_pairs):
            s, d = self.pair_idx_to_sd[pair_idx]
            demand = tm[s][d]
            if demand != 0:
                self.ecmp_next_hop_distribution(link_loads, demand, s, d)

        return link_loads

    def ecmp_mlu(self):
        link_loads = self.ecmp_traffic_distribution()
        mlu = np.max(link_loads / self.link_capacities)
        return mlu

    def advantage(self, reward):
        if self.indexes[self.tm_idx] not in self.baseline:
            return reward

        total_v, cnt = self.baseline[self.indexes[self.tm_idx]]

        # print(reward, (total_v/cnt))

        return reward - (total_v / cnt)

    def get_topK_flows(self, pairs):
        tm = self.traffic_matrices[self.tm_idx]
        f = {}
        for p in pairs:
            s, d = self.pair_idx_to_sd[p]
            f[p] = tm[s][d]

        sorted_f = sorted(f.items(), key = lambda kv: (kv[1], kv[0]), reverse=True)

        cf = []
        for i in range(self.max_moves):
            cf.append(sorted_f[i][0])

        return cf
def set_seed(seed):
    np.random.seed(seed)
