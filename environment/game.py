from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import networkx as nx
import pulp

import wandb
from tqdm import tqdm
import numpy as np
from pulp import LpMinimize, LpMaximize, LpProblem, LpStatus, lpSum, LpVariable, value, GLPK
import wandb
OBJ_EPSILON = 1e-12


class Game(object):
    """
        Game class that simulates the network

        Attributes:
            num_pairs: int
                Number of origin destination pairs.
            config: Environment
                The configuration provided in the file
            env:  The created enviroment
            random_seed = 1000 : a parameter to randomize the state.

        Methods:
            generate_inputs(normalization=True)
                Generate input data for a matrix prediction model.
                Prepares input data via creating normalized or unnormalized traffic matrices
                based on historic data.

            get_topK_flows(tm_index, pairs)
                Get the top-K flows in a matrix at a given index

            get_ecmp_next_hops()
                Compute the Equal-Cost Multipath (ECMP) next hops for each source-destination pair.

            ecmp_next_hop_distribution(link_loads, demand, src, dst):
                Distribute demand evenly across ECMP next hops between source and destination

            eval_ecmp_traffic_distribution(tm_idx, eval_delay=False):

            get_critical_topK_flows(tm_idx, critical_links=5):

            eval_ecmp_traffic_distribution(tm_idx, eval_delay=False):

            optimal_routing_mlu(tm_idx):

            eval_optimal_routing_mlu(tm_idx, solution, eval_delay=False):

            optimal_routing_mlu_critical_pairs(tm_idx, critical_pairs):

            eval_critical_flow_and_ecmp(tm_idx, critical_pairs, solution, eval_delay=False):

            optimal_routing_delay(tm_idx)

            eval_optimal_routing_delay(self, tm_idx, solution):
    """

    num_nodes: int
    """Number of nodes."""
    num_links: int
    """Number of links."""
    num_pairs: int
    """Number of pairs."""
    link_idx_to_sd: dict
    """Convert from index to source destination pair"""
    link_idx_to_sd: dict
    """Convert from source destination pair to index"""
    pair_links: list
    """List containing (index, source, destination)"""

    def __init__(self, config, env, random_seed=1000):
        self.random_state = np.random.RandomState(seed=random_seed)

        self.env = env
        self.data_dir = env.data_dir
        self.DG = env.topology.DG
        self.traffic_file = env.traffic_file
        self.traffic_matrices = env.traffic_matrices
        self.traffic_matrices_dims = self.traffic_matrices.shape
        self.tm_cnt = env.tm_cnt
        self.num_pairs = env.num_pairs
        self.pair_idx_to_sd = env.pair_idx_to_sd
        self.pair_sd_to_idx = env.pair_sd_to_idx
        self.num_nodes = env.num_nodes
        self.num_links = env.num_links
        self.link_idx_to_sd = env.link_idx_to_sd
        self.link_sd_to_idx = env.link_sd_to_idx
        self.link_capacities = env.link_capacities
        self.link_weights = env.link_weights
        self.shortest_paths_node = env.shortest_paths_node              # paths with node info
        self.shortest_paths_link = env.shortest_paths_link              # paths with link info

        self.get_ecmp_next_hops()

        self.model_type = config.model_type

        #for LP
        self.pairs_idx = [p for p in range(self.num_pairs)] 
        self.node_idx = [n for n in range(self.num_nodes)]
        self.links_idx = [e for e in range(self.num_links)]
        self.pairs = [e for e in self.link_sd_to_idx]
        self.pair_links = [(index, pair[0], pair[1]) for index in self.pairs_idx for pair in self.pairs]
        self.load_multiplier = {}


    def get_topology(self, normalize=True):
        mat = nx.adjacency_matrix(self.DG).toarray()
        if normalize:
            mat = (mat.flatten() / np.max(mat.flatten())).reshape(12,12)
        return mat.astype('float')

    def generate_inputs(self, normalization=True):
        self.normalized_traffic_matrices = np.zeros((self.valid_tm_cnt, self.traffic_matrices_dims[1], self.traffic_matrices_dims[2], self.tm_history), dtype=np.float32)   #tm state  [Valid_tms, Node, Node, History]
        idx_offset = self.tm_history - 1
        for tm_idx in self.tm_indexes:
            for h in range(self.tm_history):
                if normalization:
                    tm_max_element = np.max(self.traffic_matrices[tm_idx-h])
                    self.normalized_traffic_matrices[tm_idx-idx_offset,:,:,h] = self.traffic_matrices[tm_idx-h] / tm_max_element        #[Valid_tms, Node, Node, History]
                else:
                    self.normalized_traffic_matrices[tm_idx-idx_offset,:,:,h] = self.traffic_matrices[tm_idx-h]                         #[Valid_tms, Node, Node, History]

    def get_topK_flows(self, tm_idx, pairs):
        tm = self.traffic_matrices[tm_idx]
        f = {}
        for p in pairs:
            s, d = self.pair_idx_to_sd[p]
            f[p] = tm[s][d]

        sorted_f = sorted(f.items(), key = lambda kv: (kv[1], kv[0]), reverse=True)

        cf = []
        for i in range(self.max_moves):
            cf.append(sorted_f[i][0])

        return cf

    def get_ecmp_next_hops(self):
        self.ecmp_next_hops = {}
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                if src == dst:
                    continue
                self.ecmp_next_hops[src, dst] = []
                for p in self.shortest_paths_node[self.pair_sd_to_idx[(src, dst)]]:
                    if p[1] not in self.ecmp_next_hops[src, dst]:
                        self.ecmp_next_hops[src, dst].append(p[1])

    def ecmp_next_hop_distribution(self, link_loads, demand, src, dst):
        if src == dst:
            return

        ecmp_next_hops = self.ecmp_next_hops[src, dst]

        next_hops_cnt = len(ecmp_next_hops)
        #if next_hops_cnt > 1:
            #print(self.shortest_paths_node[self.pair_sd_to_idx[(src, dst)]])

        ecmp_demand = demand / next_hops_cnt
        for np in ecmp_next_hops:
            link_loads[self.link_sd_to_idx[(src, np)]] += ecmp_demand
            self.ecmp_next_hop_distribution(link_loads, ecmp_demand, np, dst)

    def ecmp_traffic_distribution(self, tm_idx):
        link_loads = np.zeros((self.num_links))
        tm = self.traffic_matrices[tm_idx]
        for pair_idx in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[pair_idx]
            demand = tm[s][d]
            if demand != 0:
                self.ecmp_next_hop_distribution(link_loads, demand, s, d)

        return link_loads

    def get_critical_topK_flows(self, tm_idx, critical_links=5):
        link_loads = self.ecmp_traffic_distribution(tm_idx)
        critical_link_indexes = np.argsort(-(link_loads / self.link_capacities))[:critical_links]

        cf_potential = []
        for pair_idx in range(self.num_pairs):
            for path in self.shortest_paths_link[pair_idx]:
                if len(set(path).intersection(critical_link_indexes)) > 0:
                    cf_potential.append(pair_idx)
                    break

        #print(cf_potential)
        assert len(cf_potential) >= self.max_moves, \
                ("cf_potential(%d) < max_move(%d), please increse critical_links(%d)"%(len(cf_potential), self.max_moves, critical_links))

        return self.get_topK_flows(tm_idx, cf_potential)

    def eval_ecmp_traffic_distribution(self, tm_idx, eval_delay=False):
        eval_link_loads = self.ecmp_traffic_distribution(tm_idx)
        eval_max_utilization = np.max(eval_link_loads / self.link_capacities)
        self.load_multiplier[tm_idx] = 0.9 / eval_max_utilization  # Where does 0.9 come from?
        delay = 0
        if eval_delay:
            eval_link_loads *= self.load_multiplier[tm_idx]
            delay = sum(eval_link_loads / (self.link_capacities - eval_link_loads))

        return eval_max_utilization, delay

    def optimal_routing_mlu(self, tm_idx):
        tm = self.traffic_matrices[tm_idx]
        demands = {}
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            demands[i] = tm[s][d]
        # demands[i] (size N*N) is the flattened traffic matrix.
        
        model = LpProblem(name="routing")

        # MLU? -> VARIABLE BETWEEN 0 AND 1
        ratio = LpVariable.dicts(name="ratio", indexs=self.pair_links, lowBound=0, upBound=1)

        # link load
        link_load = LpVariable.dicts(name="link_load", indexs=self.links_idx)

        congestion_ratio = LpVariable(name="congestion_ratio")

        for index in self.pairs_idx: 
            #o,d = origin, destination
                    # (sum of ratios where destination == origin)  - (sum of ratios where origin == origin) == -1
            model += (lpSum([ratio[index, o, d] for o, d in self.pairs if d == self.pair_idx_to_sd[index][0]]) - 
                      lpSum([ratio[index, o, d] for o, d in self.pairs if o == self.pair_idx_to_sd[index][0]]) == -1, 
                      f"flow_conservation_constr1_{index}")

        for index in self.pairs_idx:
                    # (Sum of ratios where destination == destination) - (Sum of ratios where origin == destionation) == 1
            model += (lpSum([ratio[index, o, d] for o,d in self.pairs if d == self.pair_idx_to_sd[index][1]]) - 
                      lpSum([ratio[index, o, d] for o,d in self.pairs if o == self.pair_idx_to_sd[index][1]]) == 1,
                      f"flow_conservation_constr2_{index}")

        for index in self.pairs_idx:
            for node in self.node_idx:
                if node not in self.pair_idx_to_sd[index]:
                    # Traffic in = Traffic Out 
                    model += (lpSum([ratio[index, s, d] for s,d in self.pairs if d == node]) - 
                              lpSum([ratio[index, s, d] for s,d in self.pairs if s == node]) == 0, 
                              f"flow_conservation_constr3_{index}_{node}")

        for s, d in self.pairs:
            idx = self.link_sd_to_idx[(s,d)]
            # link load definition
            model += (link_load[idx] == lpSum([demands[pair_idx]*ratio[pair_idx, s, d] for pair_idx in self.pairs_idx]), f"link_load_constr{idx}")
            # link load has to be smaller than capacities
            model += (link_load[idx] <= self.link_capacities[idx]*congestion_ratio, f"congestion_ratio_constr{idx}")

        # Operation to minize?
        model += congestion_ratio + OBJ_EPSILON*lpSum([link_load[idx] for idx in self.links_idx])

        model.solve(solver=GLPK(msg=False))


        assert LpStatus[model.status] == 'Optimal'

        obj_r = congestion_ratio.value()
        solution = {}
        for k in ratio:
            solution[k] = ratio[k].value()

        return obj_r, solution

    def eval_optimal_routing_mlu(self, tm_idx, solution, eval_delay=False):
        optimal_link_loads = np.zeros((self.num_links))
        eval_tm = self.traffic_matrices[tm_idx]
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            demand = eval_tm[s][d]
            for e in self.pairs:
                link_idx = self.link_sd_to_idx[e]
                optimal_link_loads[link_idx] += demand*solution[i, e[0], e[1]]

        optimal_max_utilization = np.max(optimal_link_loads / self.link_capacities)
        delay = 0
        if eval_delay:
            assert tm_idx in self.load_multiplier, (tm_idx)
            optimal_link_loads *= self.load_multiplier[tm_idx]
            delay = sum(optimal_link_loads / (self.link_capacities - optimal_link_loads))

        return optimal_max_utilization, delay

    def optimal_routing_mlu_critical_pairs(self, tm_idx, critical_pairs):
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

        tm = self.traffic_matrices[tm_idx]
        name = self.env.topology_name
        pairs = critical_pairs

        # If the flow is critical it gets inserted into demands
        demands = {}
        background_link_loads = np.zeros((self.num_links))
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            #background link load
            if i not in critical_pairs:
                self.ecmp_next_hop_distribution(background_link_loads, tm[s][d], s, d)
            else:
                demands[i] = tm[s][d]

        # Initializes a Linear Programming problem ?
        model = LpProblem(name=f"routing_{name}")

        pair_links = [(pr, e[0], e[1]) for pr in pairs for e in self.pairs]
        ratio = LpVariable.dicts(name=f"ratio_{name}", indexs=pair_links, lowBound=0, upBound=1)

        link_load = LpVariable.dicts(name=f"link_load_{name}", indexs=self.links_idx)

        r = LpVariable(name=f"congestion_ratio_{name}")

        # Flow into source nodes = Flow out of source nodes
        for pr in pairs:
            model += (lpSum([ratio[pr, e[0], e[1]] for e in self.pairs if e[1] == self.pair_idx_to_sd[pr][0]]) - lpSum([ratio[pr, e[0], e[1]] for e in self.pairs if e[0] == self.pair_idx_to_sd[pr][0]]) == -1, f"{name}_flow_conservation_constr1_{pr}")

        # Flow into destination node = Flow out of destination node
        for pr in pairs:
            model += (lpSum([ratio[pr, e[0], e[1]] for e in self.pairs if e[1] == self.pair_idx_to_sd[pr][1]]) - lpSum([ratio[pr, e[0], e[1]] for e in self.pairs if e[0] == self.pair_idx_to_sd[pr][1]]) == 1, f"{name}_flow_conservation_constr2_{pr}")

        # Flow conservation for intermediate nodes.
        for pr in pairs:
            for n in self.node_idx:
                if n not in self.pair_idx_to_sd[pr]:
                    model += (lpSum([ratio[pr, e[0], e[1]] for e in self.pairs if e[1] == n]) - lpSum([ratio[pr, e[0], e[1]] for e in self.pairs if e[0] == n]) == 0, f"{name}_flow_conservation_constr3_{pr}_{n}")

        # Adds constraints to the links / ensures that the capacity is not exceeded
        for e in self.pairs:
            ei = self.link_sd_to_idx[e]
            model += (link_load[ei] == background_link_loads[ei] + lpSum([demands[pr]*ratio[pr, e[0], e[1]] for pr in pairs]), f"{name}_link_load_constr{ei}")
            model += (link_load[ei] <= self.link_capacities[ei]*r, f"{name}_congestion_ratio_constr{ei}")

        # Objective function, minimize r (Congestion Ratio)
        model += r + OBJ_EPSILON*lpSum([link_load[ei] for ei in self.links_idx])

        # Solve the problem
        model.solve(solver=GLPK(msg=False))
        assert LpStatus[model.status] == 'Optimal', LpStatus[model.status]

        obj_r = r.value()
        solution = {}
        for k in ratio:
            solution[k] = ratio[k].value()

        return obj_r, solution

    def eval_critical_flow_and_ecmp(self, tm_idx, critical_pairs, solution, eval_delay=False):
        eval_tm = self.traffic_matrices[tm_idx]
        eval_link_loads = np.zeros((self.num_links))
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            if i not in critical_pairs:
                self.ecmp_next_hop_distribution(eval_link_loads, eval_tm[s][d], s, d)
            else:
                demand = eval_tm[s][d]
                for e in self.pairs:
                    link_idx = self.link_sd_to_idx[e]
                    eval_link_loads[link_idx] += eval_tm[s][d]*solution[i, e[0], e[1]]

        eval_max_utilization = np.max(eval_link_loads / self.link_capacities)
        delay = 0
        if eval_delay:
            assert tm_idx in self.load_multiplier, (tm_idx)
            eval_link_loads *= self.load_multiplier[tm_idx]
            delay = sum(eval_link_loads / (self.link_capacities - eval_link_loads))

        return eval_max_utilization, delay

    def optimal_routing_delay(self, tm_idx):
        # To do with multiple topologies
        assert tm_idx in self.load_multiplier, (tm_idx)
        tm = self.traffic_matrices[tm_idx]*self.load_multiplier[tm_idx]
        demands = {}
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            demands[i] = tm[s][d]

        model = LpProblem(name="routing")

        ratio = LpVariable.dicts(name="ratio", indexs=self.pair_links, lowBound=0, upBound=1)

        link_load = LpVariable.dicts(name="link_load", indexs=self.links_idx)

        f = LpVariable.dicts(name="link_cost", indexs=self.links_idx)

        for pr in self.pairs_idx:
            model += (lpSum([ratio[pr, e[0], e[1]] for e in self.pairs if e[1] == self.pair_idx_to_sd[pr][0]]) - lpSum([ratio[pr, e[0], e[1]] for e in self.pairs if e[0] == self.pair_idx_to_sd[pr][0]]) == -1, "flow_conservation_constr1_%d"%pr)

        for pr in self.pairs_idx:
            model += (lpSum([ratio[pr, e[0], e[1]] for e in self.pairs if e[1] == self.pair_idx_to_sd[pr][1]]) - lpSum([ratio[pr, e[0], e[1]] for e in self.pairs if e[0] == self.pair_idx_to_sd[pr][1]]) == 1, "flow_conservation_constr2_%d"%pr)

        for pr in self.pairs_idx:
            for n in self.node_idx:
                if n not in self.pair_idx_to_sd[pr]:
                    model += (lpSum([ratio[pr, e[0], e[1]] for e in self.pairs if e[1] == n]) - lpSum([ratio[pr, e[0], e[1]] for e in self.pairs if e[0] == n]) == 0, "flow_conservation_constr3_%d_%d"%(pr,n))

        for e in self.pairs:
            ei = self.link_sd_to_idx[e]
            model += (link_load[ei] == lpSum([demands[pr]*ratio[pr, e[0], e[1]] for pr in self.pairs_idx]), "link_load_constr%d"%ei)
            model += (f[ei] * self.link_capacities[ei] >= link_load[ei], "cost_constr1_%d"%ei)
            model += (f[ei] >= 3 * link_load[ei] / self.link_capacities[ei] - 2/3, "cost_constr2_%d"%ei)
            model += (f[ei] >= 10 * link_load[ei] / self.link_capacities[ei] - 16/3, "cost_constr3_%d"%ei)
            model += (f[ei] >= 70 * link_load[ei] / self.link_capacities[ei] - 178/3, "cost_constr4_%d"%ei)
            model += (f[ei] >= 500 * link_load[ei] / self.link_capacities[ei] - 1468/3, "cost_constr5_%d"%ei)
            model += (f[ei] >= 5000 * link_load[ei] / self.link_capacities[ei] - 16318/3, "cost_constr6_%d"%ei)

        model += lpSum(f[ei] for ei in self.links_idx)

        model.solve(solver=GLPK(msg=False))
        assert LpStatus[model.status] == 'Optimal'

        solution = {}
        for k in ratio:
            solution[k] = ratio[k].value()

        return solution

    def eval_optimal_routing_delay(self, tm_idx, solution):
        optimal_link_loads = np.zeros((self.num_links))
        assert tm_idx in self.load_multiplier, (tm_idx)
        eval_tm = self.traffic_matrices[tm_idx]*self.load_multiplier[tm_idx]
        for i in range(self.num_pairs):
            s, d = self.pair_idx_to_sd[i]
            demand = eval_tm[s][d]
            for e in self.pairs:
                link_idx = self.link_sd_to_idx[e]
                optimal_link_loads[link_idx] += demand*solution[i, e[0], e[1]]

        optimal_delay = sum(optimal_link_loads / (self.link_capacities - optimal_link_loads))

        return optimal_delay


class CFRRL_Game(Game):
    def __init__(self, config, env, random_seed=1000, baseline=True):
        super(CFRRL_Game, self).__init__(config, env, random_seed)
        
        self.project_name = config.project_name
        self.action_dim = env.num_pairs
        self.max_moves = int(self.action_dim * (config.max_moves / 100.))
        assert self.max_moves <= self.action_dim, (self.max_moves, self.action_dim)
        
        self.tm_history = 1
        self.tm_indexes = np.arange(self.tm_history-1, self.tm_cnt)
        self.valid_tm_cnt = len(self.tm_indexes)
        
        if baseline:
            self.baseline = {}

        self.generate_inputs(normalization=True)
        self.state_dims = self.normalized_traffic_matrices.shape[1:]
        #print('Input dims :', self.state_dims)
        #print('Max moves :', self.max_moves)

    def get_state(self, tm_idx):
        idx_offset = self.tm_history - 1
        return self.normalized_traffic_matrices[tm_idx-idx_offset]

    def reward(self, tm_idx, actions):
        mlu, _ = self.optimal_routing_mlu_critical_pairs(tm_idx, actions)

        #ecmp_mlu, _ = np.round(self.eval_ecmp_traffic_distribution(tm_idx, eval_delay=False),4)

        #quasi_optimal, _ = self.optimal_routing_mlu(tm_idx, True)
        #optimal_mlu, _ = self.optimal_routing_mlu(tm_idx)
        #print(optimal_mlu)

        # Critical MLU
        crit_topk = self.get_critical_topK_flows(tm_idx)
        crit_mlu, _ = self.optimal_routing_mlu_critical_pairs(tm_idx, crit_topk)
        #crit_mlu, _ = self.eval_critical_flow_and_ecmp(tm_idx, crit_topk, solution, eval_delay=False)

        reward = ((crit_mlu - mlu)/crit_mlu) + 1
        #reward = crit_mlu / mlu

        return reward

        """ print(f"Execution time: {execution_time} seconds")
        print(f"Quasi Optimal {quasi_optimal}, Optimal {optimal_mlu}, ECMP: {ecmp_mlu}")"""
        #_, solution = self.optimal_routing_mlu_critical_pairs(tm_idx, actions)
        '''print('======================================')
        if mlu < crit_mlu:
            print(1)
            reward = 1 + (crit_mlu - mlu) / crit_mlu
        elif (abs(crit_mlu - ecmp_mlu) == 0):
            print(2)
            reward = 0.9

        else:
            print(3)
            reward = (1 - 2 * abs(mlu - crit_mlu) / abs(crit_mlu - ecmp_mlu))



        print(reward)
        print('=======================================')'''
        reward = crit_mlu / mlu 
        #print(reward)
        #reward = (ecmp_mlu / mlu) - 1
        #print(f'mlu: {mlu}, optimal: {optimal_mlu}, ecmp: {ecmp_mlu}, reward: {reward}')

        #reward = optimal_mlu / mlu
        return reward

    def advantage(self, tm_idx, reward):
        if tm_idx not in self.baseline:
            return reward

        total_v, cnt = self.baseline[tm_idx]
        
        #print(reward, (total_v/cnt))

        return reward - (total_v/cnt)

    def update_baseline(self, tm_idx, reward):
        if tm_idx in self.baseline:
            total_v, cnt = self.baseline[tm_idx]

            total_v += reward
            cnt += 1

            self.baseline[tm_idx] = (total_v, cnt)
        else:
            self.baseline[tm_idx] = (reward, 1)

    def evaluate(self, tm_idx, actions=None, ecmp=True, eval_delay=False):

        # Evaluates traffic distribution, calculates MLU (Maximum Link Utilization) and the delay
        # Line : Traffic Matrix Index | Normalized MLU | MLU | Normalized Critical MLU | Critical MLU |
        #        Normalized TOPK MLU  | TOPK       MLU |
        if ecmp:
            ecmp_mlu, ecmp_delay = self.eval_ecmp_traffic_distribution(tm_idx, eval_delay=eval_delay)



        _, solution = self.optimal_routing_mlu_critical_pairs(tm_idx, actions)
        mlu, delay = self.eval_critical_flow_and_ecmp(tm_idx, actions, solution, eval_delay=eval_delay)

        crit_topk = self.get_critical_topK_flows(tm_idx)
        _, solution = self.optimal_routing_mlu_critical_pairs(tm_idx, crit_topk)
        crit_mlu, crit_delay = self.eval_critical_flow_and_ecmp(tm_idx, crit_topk, solution, eval_delay=eval_delay)

        topk = self.get_topK_flows(tm_idx, self.pairs_idx)
        _, solution = self.optimal_routing_mlu_critical_pairs(tm_idx, topk)
        topk_mlu, topk_delay = self.eval_critical_flow_and_ecmp(tm_idx, topk, solution, eval_delay=eval_delay)

        _, solution = self.optimal_routing_mlu(tm_idx)
        optimal_mlu, optimal_mlu_delay = self.eval_optimal_routing_mlu(tm_idx, solution, eval_delay=eval_delay)

        norm_mlu = optimal_mlu / mlu
        line = str(tm_idx) + ', ' + str(norm_mlu) + ', ' + str(mlu) + ', ' 
        
        norm_crit_mlu = optimal_mlu / crit_mlu
        line += str(norm_crit_mlu) + ', ' + str(crit_mlu) + ', ' 

        norm_topk_mlu = optimal_mlu / topk_mlu
        line += str(norm_topk_mlu) + ', ' + str(topk_mlu) + ', '

        log = {'Traffic_Matrix_Index': tm_idx,
               'Normalized MLU': norm_mlu,
               'MLU': mlu,
               'Normalized Critical MLU': norm_crit_mlu,
               'Critical MLU': crit_mlu,
               'Normalized TopK MLU': norm_topk_mlu,
               'TopK MLU': topk_mlu}

        if ecmp:
            norm_ecmp_mlu = optimal_mlu / ecmp_mlu
            line += str(norm_ecmp_mlu) + ', ' + str(ecmp_mlu) + ', '
            log['Normalized ECMP MLU'] = norm_ecmp_mlu
            log['ECMP MLU'] = ecmp_mlu

        if eval_delay:
            solution = self.optimal_routing_delay(tm_idx)
            optimal_delay = self.eval_optimal_routing_delay(tm_idx, solution) 

            line += str(optimal_delay/delay) + ', ' 
            line += str(optimal_delay/crit_delay) + ', ' 
            line += str(optimal_delay/topk_delay) + ', ' 
            line += str(optimal_delay/optimal_mlu_delay) + ', '
            if ecmp:
                line += str(optimal_delay/ecmp_delay) + ', '
        
            assert tm_idx in self.load_multiplier, (tm_idx)
            line += str(self.load_multiplier[tm_idx]) + ', '

        #print(line[:-2])
        wandb.log(log)