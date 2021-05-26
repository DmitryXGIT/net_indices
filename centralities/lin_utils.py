import networkx as nx
import scipy as sp
import numpy as np
from scipy.sparse import linalg


def laplacian_matrix(A):
    n, __ = A.shape
    Diag_d = np.diag(np.squeeze(np.matmul(A, np.ones((n, 1)))))
    return Diag_d - A


def to_stochastic_matrix(M):
    M_sum = sp.sparse.diags(1 / M.sum(axis=1).A.ravel())
    return M_sum @ M


def to_sparse_matrix(G):
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept('Cannot compute centrality')
    return nx.to_scipy_sparse_matrix(G, nodelist=list(G), weight=None, dtype=float)


def dominant_eig(M, left=False, which='LR'):
    if left:
        eigenvalue, eigenvector = linalg.eigs(M.T, k=1, which=which, maxiter=50, tol=0)
    else:
        eigenvalue, eigenvector = linalg.eigs(M, k=1, which=which, maxiter=50, tol=0)

    largest = eigenvector.flatten().real
    norm = sp.sign(largest.sum()) * sp.linalg.norm(largest)
    return eigenvalue.real, largest / norm



