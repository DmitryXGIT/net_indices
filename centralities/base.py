from abc import abstractmethod
from enum import Enum
import numpy as np
import networkx as nx
from scipy.linalg import expm
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix

from .kernels import Kernel, spectral_radius
from .kernel_distances import KernelDistance
from .lin_utils import to_stochastic_matrix, to_sparse_matrix, dominant_eig, laplacian_matrix


def get_adj_matrix(G):
    return np.asarray(nx.to_numpy_matrix(G))


def create_centrality_dict(G, cc):
    c_dict = dict()
    for i, nid in enumerate(G.nodes()):
        c_dict[nid] = cc[i]
    return c_dict


def create_centrality_vec(G, c_dict):
    n = G.number_of_nodes()
    c = np.zeros(n)
    for i, nid in enumerate(G.nodes()):
        c[i] = c_dict[nid]
    return c


class Centrality:

    name = None
    default_params = None

    @property
    def id(self):
        return self._id

    def _set_parameters(self, params):
        self._params = self.default_params.copy()
        if not params:
            return None

        non_def_params = dict()
        for p in params:
            if p not in self.default_params:
                raise Exception(f'Unknown parameter {p} for {self.name} centrality.')
            elif params[p] != self.default_params[p]:
                non_def_params[p] = params[p]

        if len(non_def_params) > 0:
            self._params.update(non_def_params)
            return non_def_params
        else:
            return None

    def _set_id(self, non_def_params):
        self._id = self.name
        if not non_def_params:
            return

        keys = sorted(non_def_params.keys())
        s = ','.join([f'{k}={non_def_params[k]}' for k in keys])
        self._id = f'{self.name} [{s}]'

    def __init__(self, params=None):
        non_def_params = self._set_parameters(params)
        self._set_id(non_def_params)

    @abstractmethod
    def compute(self, G):
        return None

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.id == other.id
        return False


class GraphBasedCentrality(Centrality):

    @abstractmethod
    def compute(self, G):
        pass


class CentralityIntegrationRadiality(GraphBasedCentrality):

    name = 'integration/r'
    default_params = {
    }

    @staticmethod
    def centrality_integration_radiality_general(G, method='radiality'):
        D = np.asarray(nx.floyd_warshall_numpy(G))
        if method == 'radiality':
            D_ = D
        elif method == 'integration':
            D_ = D.T
        else:
            raise Exception(f'Method {method} is not supported.')

        n, _ = D_.shape
        c = np.zeros(n)
        max_D = np.max(np.where(D_ == np.inf, -np.inf, D_))
        for i in range(n):
            for j in range(n):
                if i != j and D_[i, j] != np.inf:
                    c[i] += 1 + max_D - D_[i, j]
        c = c / (n - 1)
        return c

    @staticmethod
    def centrality_integration_radiality_simple(D):
        n, _ = D.shape
        return 1 + np.max(D) - np.sum(D, axis=1) / (n - 1)

    def compute(self, G):
        D = np.asarray(nx.floyd_warshall_numpy(G))
        cc = self.centrality_integration_radiality_simple(D)
        return create_centrality_dict(G, cc)


class CentralityPMeans(GraphBasedCentrality):

    name = 'p-means'
    default_params = {
        'p': 1.0
    }

    def compute(self, G):
        D = np.asarray(nx.floyd_warshall_numpy(G))
        n, _ = D.shape
        p = self._params['p']
        D_ = D.copy()
        if p == 0:
            D_[D_ == 0] = 1
            cc = np.prod(D_, axis=1) ** (-1 / (n - 1))
        else:
            np.power(D, p, out=D_, where=D > 0)
            cc = (np.sum(D_, axis=1) / (n - 1)) ** (-1 / p)

        return create_centrality_dict(G, cc)


class CentralityHarmonicCloseness(GraphBasedCentrality):

    name = 'clo harmonic'
    default_params = {
    }

    def compute(self, G):
        D = np.asarray(nx.floyd_warshall_numpy(G))
        n, _ = D.shape
        D_ = D.copy()
        np.divide(1, D, out=D_, where=D > 0)  # 1/D
        cc = np.sum(D_, axis=1) / (n - 1)
        return create_centrality_dict(G, cc)


class CentralityWeightedDegree(GraphBasedCentrality):

    name = 'deg weighted'
    default_params = {
    }

    def compute(self, G):
        A = get_adj_matrix(G)
        D = np.asarray(nx.floyd_warshall_numpy(G))
        n, _ = D.shape
        D_ = D.copy()
        deg = np.sum(A, axis=0, keepdims=True)
        np.divide(deg, D, out=D_, where=D > 0)
        cc = np.sum(D_, axis=1)
        return create_centrality_dict(G, cc)


class CentralityDecayingDegree(GraphBasedCentrality):

    name = 'deg decaying'
    default_params = {
    }

    def compute(self, G):
        A = get_adj_matrix(G)
        D = np.asarray(nx.floyd_warshall_numpy(G))
        n, _ = D.shape
        D_ = D.copy()
        deg = np.sum(A, axis=0, keepdims=True)
        np.divide(deg, np.power(n, 2 * D), out=D_)  # where=D > 0
        cc = np.sum(D_, axis=1)  # / (n-1)
        return create_centrality_dict(G, cc)


class CentralityDecay(GraphBasedCentrality):

    name = 'decay'
    default_params = {
        'delta': 1.0
    }

    def compute(self, G):
        delta = self._params['delta']
        D = np.asarray(nx.floyd_warshall_numpy(G))
        cc = np.sum(np.power(delta, D), axis=1) - 1
        return create_centrality_dict(G, cc)


class CentralitySeeley(GraphBasedCentrality):

    name = 'seeley'
    default_params = {
    }

    def compute(self, G):
        M = to_stochastic_matrix(to_sparse_matrix(G))
        eigval, eigvec = dominant_eig(M, left=True)
        if np.round(eigval, 6) != 1.0:
            raise ValueError(f'Seeley centrality: eigenvalue ({eigval}) must be equal to 1')
        return create_centrality_dict(G, eigvec)


class CentralityEigenOnDissim(GraphBasedCentrality):

    name = 'eig_dissim'
    default_params = {
        'metric': 'jaccard'
    }

    def compute(self, G):
        A = get_adj_matrix(G)
        n, _ = A.shape
        A = np.asarray(A + np.eye(n), dtype=bool)
        D = squareform(pdist(A, self._params['metric']))
        W = csr_matrix(A * D)
        eigval, eigvec = dominant_eig(W, left=False)
        return create_centrality_dict(G, eigvec)


class CentralityBetaCurrentFlow(GraphBasedCentrality):

    name = 'bCF'
    default_params = {
        'beta': 1.0
    }

    def compute(self, G):
        A = np.asarray(nx.to_numpy_matrix(G))
        n, __ = A.shape
        L = laplacian_matrix(A)

        phi = np.linalg.matrix_power(np.eye(n) * self._params['beta'] + L, -1)

        y = np.zeros(n)
        for i in range(n):
            for k in range(n):
                s = 0.0
                for j in range(n):
                    s += np.abs(phi[i, j] - phi[k, j])
                y[i] += A[i, k] * s
        y = (y + 1) / (2 * n)

        return create_centrality_dict(G, y)


class CentralityBridging(GraphBasedCentrality):

    name = 'bridging'
    default_params = {
    }

    def compute(self, G):
        A = np.asarray(nx.to_numpy_matrix(G))
        n, __ = A.shape
        deg = np.sum(A, axis=1)

        BC = 1 / (deg * A.dot(1 / deg))
        bc_dict = dict(zip(G, BC))
        cr_dict = nx.betweenness_centrality(G)
        for k in cr_dict:
            cr_dict[k] = cr_dict[k] * bc_dict[k]
        return cr_dict


class CentralityEstrada(GraphBasedCentrality):

    name = 'estrada'
    default_params = {
    }

    def compute(self, G):
        A = np.asarray(nx.to_numpy_matrix(G))
        cc = np.diag(expm(A))
        return create_centrality_dict(G, cc)


class CentralityTotalComm(GraphBasedCentrality):

    name = 'total_comm'
    default_params = {
    }

    def compute(self, G):
        A = np.asarray(nx.to_numpy_matrix(G))
        cc = np.sum(expm(A), axis=1)
        return create_centrality_dict(G, cc)


class CentralityBetweenness(GraphBasedCentrality):

    name = 'bet'
    default_params = {
        'normalized': False
    }

    def compute(self, G):
        c_dict = nx.betweenness_centrality(G, **self._params)
        return c_dict


class CentralityEccentricity(GraphBasedCentrality):

    name = 'ecc'
    default_params = {
    }

    def compute(self, G):
        c_dict = nx.eccentricity(G)
        for k in c_dict:
            c_dict[k] = 1.0 / c_dict[k]
        return c_dict


class CentralityDegree(GraphBasedCentrality):

    name = 'deg'
    default_params = {
    }

    def compute(self, G):
        c_dict = nx.degree_centrality(G, **self._params)
        return c_dict


class CentralityCloseness(GraphBasedCentrality):

    name = 'clo'
    default_params = {
    }

    def compute(self, G):
        c_dict = nx.closeness_centrality(G, **self._params)
        return c_dict


class CentralityPagerank(GraphBasedCentrality):

    name = 'pr'
    default_params = {
        'alpha': 0.85,
        'personalization': None,
        'max_iter': 1000,
        'tol': 1e-07,
        'dangling': None
    }

    def compute(self, G):
        c_dict = nx.pagerank(G, **self._params)
        return c_dict


class CentralityEigenvector(GraphBasedCentrality):

    name = 'eig'
    default_params = {
    }

    def compute(self, G):
        c_dict = nx.eigenvector_centrality(G, **self._params)
        return c_dict


class CentralityGeneralizedDegree(GraphBasedCentrality):

    name = 'gen_deg'
    default_params = {
        'alpha': 2.0
    }

    def compute(self, G):
        alpha = self._params['alpha']
        A = get_adj_matrix(G)
        n = A.shape[0]
        I = np.eye(n)
        d = np.sum(A > 0, axis=0)

        Diag_d = np.diag(np.squeeze(np.matmul(A, np.ones((n, 1)))))
        L = Diag_d - A

        K = np.linalg.matrix_power(I + alpha * L, -1)
        cc = np.matmul(K, d)
        return create_centrality_dict(G, cc)


class CentralityBonacich(GraphBasedCentrality):

    name = 'bonacich'
    default_params = {
        'alpha': 2.0
    }

    def compute(self, G):
        alpha = self._params['alpha']
        A = get_adj_matrix(G)
        n = A.shape[0]
        I = np.eye(n)
        d = np.sum(A > 0, axis=0)

        alpha = 1.0 / (spectral_radius(A) + 1.0 / alpha)
        K = np.linalg.matrix_power(I - alpha * A, -1)
        cc = np.matmul(K, d)
        return create_centrality_dict(G, cc)


class CentralityKatz(GraphBasedCentrality):

    name = 'katz'
    default_params = {
        'alpha': 1.0,
        'beta': 1.0,
        'normalized': False
    }

    def compute(self, G):
        alpha = self._params['alpha']
        beta = self._params['beta']
        normalized = self._params['normalized']

        A = get_adj_matrix(G)
        if alpha is None:
            alpha = 1.0 / (spectral_radius(A) + 1.0)
        else:
            alpha = 1.0 / (spectral_radius(A) + 1.0 / alpha)

        return nx.katz_centrality(G, alpha=alpha, beta=beta, normalized=normalized)


class KernelBasedCentrality(Centrality):

    k_type = None
    name = None
    default_params = {
        'k_log': False,
        'k_a': 1.0,
    }

    def _set_id(self, non_def_params):
        self._id = self.name
        if self._params['k_log']:
            self._id = f'l_{self.name}'

        if not non_def_params:
            return
        elif 'k_a' in non_def_params:
            self._id += f' [a={non_def_params["k_a"]}]'

    def __init__(self, params=None):
        super().__init__(params)
        self.kernel = Kernel(self.k_type, self._params['k_log'], self._params['k_a'])

    @abstractmethod
    def compute(self, G):
        pass


class CentralityCommKii(KernelBasedCentrality):

    k_type = Kernel.Category.COMM
    name = f'{k_type.label} kii'

    def compute(self, G):
        K = self.kernel.compute(get_adj_matrix(G))
        cc = np.diag(K)
        c_dict = create_centrality_dict(G, cc)
        return c_dict


class CentralityWalkKii(KernelBasedCentrality):

    k_type = Kernel.Category.WALK
    name = f'{k_type.label} kii'

    def compute(self, G):
        K = self.kernel.compute(get_adj_matrix(G))
        cc = np.diag(K)
        c_dict = create_centrality_dict(G, cc)
        return c_dict


class CentralityCommKij(KernelBasedCentrality):

    k_type = Kernel.Category.COMM
    name = f'{k_type.label} kij'

    def compute(self, G):
        K = self.kernel.compute(get_adj_matrix(G))
        cc = np.sum(K - np.diag(np.diag(K)), axis=1)
        c_dict = create_centrality_dict(G, cc)
        return c_dict


class CentralityWalkKij(KernelBasedCentrality):

    k_type = Kernel.Category.WALK
    name = f'{k_type.label} kij'

    def compute(self, G):
        K = self.kernel.compute(get_adj_matrix(G))
        cc = np.sum(K - np.diag(np.diag(K)), axis=1)
        c_dict = create_centrality_dict(G, cc)
        return c_dict


class DistanceBasedCentrality(Centrality):

    class CType(Enum):
        CLOSENESS = 'clo'
        ECCENTRICITY = 'ecc'

        def __init__(self, label):
            self.label = label

    c_type = None
    k_type = None
    name = None

    default_params = {
        'k_log': False,
        'k_a': 1.0,
        'd_squared': False,
        'd_norm_func_name': None
    }

    def _set_id(self, non_def_params):
        self._id = self.c_type.label
        if self._params['k_log']:
            self._id += f' l_{self.k_type.label}'
        else:
            self._id += f' {self.k_type.label}'

        if not non_def_params:
            return
        else:
            keys = sorted(non_def_params.keys())
            if 'k_log' in keys:
                keys.remove('k_log')
            if len(keys) > 0:
                s = ','.join([f'{k}={non_def_params[k]}' for k in keys])
                self._id += f' [{s}]'

    def __init__(self, params=None):
        super().__init__(params)
        self.kernel = Kernel(
            self.k_type,
            self._params['k_log'],
            self._params['k_a']
        )
        self.distance = KernelDistance(
            self.kernel,
            self._params['d_squared'],
            self._params['d_norm_func_name']
        )

    def compute_distance(self, G):
        D = self.distance.compute(get_adj_matrix(G))
        return D

    def compute(self, G):
        D = self.distance.compute(get_adj_matrix(G))
        cc = None
        if self.c_type == self.CType.CLOSENESS:
            (n, _) = D.shape
            D_sum = np.matmul(D, np.ones((n, 1)))
            cc = (n - 1) / np.squeeze(D_sum)
        elif self.c_type == self.CType.ECCENTRICITY:
            cc = 1.0 / np.max(D, axis=1)

        c_dict = create_centrality_dict(G, cc)
        return c_dict


class CentralityClosenessComm(DistanceBasedCentrality):

    c_type = DistanceBasedCentrality.CType.CLOSENESS
    k_type = Kernel.Category.COMM
    name = f'{c_type.label} {k_type.label}'


class CentralityClosenessForest(DistanceBasedCentrality):

    c_type = DistanceBasedCentrality.CType.CLOSENESS
    k_type = Kernel.Category.FOREST
    name = f'{c_type.label} {k_type.label}'


class CentralityClosenessHeat(DistanceBasedCentrality):

    c_type = DistanceBasedCentrality.CType.CLOSENESS
    k_type = Kernel.Category.HEAT
    name = f'{c_type.label} {k_type.label}'


class CentralityClosenessWalk(DistanceBasedCentrality):

    c_type = DistanceBasedCentrality.CType.CLOSENESS
    k_type = Kernel.Category.WALK
    name = f'{c_type.label} {k_type.label}'


class CentralityEccentricityComm(DistanceBasedCentrality):

    c_type = DistanceBasedCentrality.CType.ECCENTRICITY
    k_type = Kernel.Category.COMM
    name = f'{c_type.label} {k_type.label}'


class CentralityEccentricityForest(DistanceBasedCentrality):

    c_type = DistanceBasedCentrality.CType.ECCENTRICITY
    k_type = Kernel.Category.FOREST
    name = f'{c_type.label} {k_type.label}'


class CentralityEccentricityHeat(DistanceBasedCentrality):

    c_type = DistanceBasedCentrality.CType.ECCENTRICITY
    k_type = Kernel.Category.HEAT
    name = f'{c_type.label} {k_type.label}'


class CentralityEccentricityWalk(DistanceBasedCentrality):

    c_type = DistanceBasedCentrality.CType.ECCENTRICITY
    k_type = Kernel.Category.WALK
    name = f'{c_type.label} {k_type.label}'


if __name__ == '__main__':

    class Centrality1(Centrality):
        name = 'centr1'
        default_params = {
            'aa': 11.0,
            'bb': 2.0,
        }

        def compute(self, G):
            pass

    c1 = Centrality1({'aa': 11, 'bb': 2})
    c2 = Centrality1({'aa': 20})
    c3 = Centrality1()
    c4 = CentralityCommKii({
        'k_log': True,
        'k_a': 1.0}
    )
    c5 = CentralityClosenessComm({'k_log': True})

    cset = set()
    cset.add(c1)
    cset.add(c2)
    cset.add(c3)
    cset.add(c4)
    cset.add(c5)

    for c in cset:
        print(c.id)

    print(c1 in cset, c2 in cset)
    print(c1.id, c2.id, hash(c1), hash(c2), c2 == c3)

    G = nx.lollipop_graph(m=3, n=2)
    print(c4.compute(G))
    print(c5.compute(G))


