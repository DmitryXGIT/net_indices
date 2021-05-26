import numpy as np
import networkx as nx
from sklearn.model_selection import ParameterGrid
from scipy.stats import rankdata
from collections import defaultdict
from enum import Enum

from .opt_utils import compute_optimal_coverings
from .centralities_batch import CentralityBatchAnalyzer


def generate_graphs(param_grid, n):
    graphs_list = []
    for p in ParameterGrid(param_grid):
        if p['k'] < n:
            G = nx.connected_watts_strogatz_graph(n=n, k=p['k'], p=p['p'], tries=100, seed=None)
            graphs_list.append(G)

    non_iso_graphs_list = []
    for i, G in enumerate(graphs_list):
        for j in range(i):
            G_prev = graphs_list[j]
            if nx.is_isomorphic(G, G_prev):
                break
        else:
            non_iso_graphs_list.append(G)
    return non_iso_graphs_list


class CentralityComparator:

    class Method(Enum):
        ABS = 1
        SIGN = 2

    @staticmethod
    def node_rank_differences(node_list, c_values):
        values = [c_values[n] for n in node_list]
        ranks = rankdata(values)

        node_rank_diffs = np.subtract.outer(ranks, ranks)
        return node_rank_diffs

    @staticmethod
    def indicatory_node_pairs(node_list, c1_values, c2_values, method=Method.SIGN):
        diff1 = CentralityComparator.node_rank_differences(node_list, c1_values)
        diff2 = CentralityComparator.node_rank_differences(node_list, c2_values)

        if method == CentralityComparator.Method.SIGN:
            matrix = np.sign(diff1) != np.sign(diff2)
            indices = np.argwhere(matrix)
            if indices.size == 0:
                raise ValueError('All node rank diffs have same sign.')
        else:
            matrix = np.abs(np.abs(diff1) - np.abs(diff2))
            indices = np.argwhere(matrix == np.max(matrix))

        return [(node_list[idx[0]], node_list[idx[1]]) for idx in indices if idx[0] < idx[1]], matrix


class CentralitySeparator:

    class CPairs:

        def __init__(self, c_list):
            n = len(c_list)
            self.c_list = c_list
            self.all = {(i, j) for i in range(n) for j in range(i + 1, n)}
            self.separated = set()

        def add_separated(self, separated_new):
            self.separated |= separated_new

        def get_unseparated(self, keep_labels=True):
            un = self.all - self.separated
            if keep_labels:
                un = self.as_label_pairs(un)
            return un

        def as_label_pairs(self, pairs):
            return {(self.c_list[i1].id, self.c_list[i2].id) for i1, i2 in pairs}

        def as_label_pair(self, pair):
            i1, i2 = pair
            return self.c_list[i1].id, self.c_list[i2].id

    class Generation:

        def __init__(self, c_list, n, param_grid, sep_lt):
            self.c_list = c_list
            self.n = n
            self.param_grid = param_grid
            self.sep_lt = sep_lt

            self.sample = {
                'graphs': [],
                'analyzers': [],
                'matrices': []
            }

            self.separating_index_to_data = dict()

        def get_indicatory_node_pairs(self, separating_idx, c1, c2):
            node_list = [n for n in self.sample['graphs'][separating_idx].nodes()]
            cba = self.sample['analyzers'][separating_idx]
            c1_values = cba.centrality_to_nvalues[c1]
            c2_values = cba.centrality_to_nvalues[c2]
            indicative_node_pairs, _ = CentralityComparator.indicatory_node_pairs(node_list, c1_values, c2_values)
            return indicative_node_pairs

        def compute(self, already_separated_pairs, precision=6):
            self.sample['graphs'] = generate_graphs(param_grid=self.param_grid, n=self.n)
            _graph_preferences = []
            for G in self.sample['graphs']:
                cba = CentralityBatchAnalyzer(G=G, centralities=self.c_list)
                cba.compute_centralities(precision=precision)
                self.sample['analyzers'].append(cba)
                self.sample['matrices'].append(cba.get_correlation_matrix())
                _graph_preference = G.number_of_nodes() * (G.number_of_nodes()-1) / 2 - G.number_of_edges()
                _graph_preferences.append(_graph_preference)

            for p in already_separated_pairs:
                i, j = p
                for m in self.sample['matrices']:
                    m[j, i] = m[i, j] = np.nan

            pair_coverings = compute_optimal_coverings(
                self.sample['matrices'],
                matrix_preferences=_graph_preferences,
                max_t=self.sep_lt
            )

            self.separating_index_to_data = defaultdict(dict)
            if len(pair_coverings) > 0:
                pair_indices, mat_indices = zip(*pair_coverings)
                for p_c in pair_coverings:
                    separated_pair, separating_index = p_c
                    c1 = self.c_list[separated_pair[0]]
                    c2 = self.c_list[separated_pair[1]]
                    self.separating_index_to_data[separating_index][separated_pair] = {
                        'identifiers': (c1.id, c2.id),
                        'indicatory_node_pairs': self.get_indicatory_node_pairs(separating_index, c1, c2)
                    }
                return set(pair_indices)
            else:
                return set()

    def __init__(self, n0=5, param_grid=None, sep_lt=1.0):
        self.param_grid = param_grid or [{'k': [2, 4], 'p': [0.2, 0.5, 0.8], 'dup': [0, 1, 2]}]
        self.n0 = n0
        self.n = n0
        self.sep_lt = sep_lt

        self.c_list = CentralityBatchAnalyzer.generate_centralities()
        self.c_pairs = CentralitySeparator.CPairs(self.c_list)
        self.generations = []

    def next_generation(self):
        gen = CentralitySeparator.Generation(c_list=self.c_list,
                                             n=self.n,
                                             param_grid=self.param_grid,
                                             sep_lt=self.sep_lt)
        separated_pairs = gen.compute(self.c_pairs.separated)
        self.c_pairs.add_separated(separated_pairs)
        self.generations.append(gen)

        self.n += 1

