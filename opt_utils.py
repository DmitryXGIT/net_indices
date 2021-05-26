import numpy as np
import cvxpy


def solve_set_cover_problem(covering_mat, constraint_rp):
    (n, m) = covering_mat.shape
    selection = cvxpy.Bool(m)
    cover_constraint = covering_mat * selection >= constraint_rp
    cover_cost = cvxpy.sum_entries(selection)
    set_cover_problem = cvxpy.Problem(cvxpy.Minimize(cover_cost), [cover_constraint])
    val = set_cover_problem.solve(solver=cvxpy.GLPK_MI)
    return selection.value, val


def create_covering_matrix(matrix_list, matrix_preferences=None, max_t=1.0):
    n, _ = matrix_list[0].shape
    pair_to_matrix = np.array([m[np.triu_indices(n, 1)] < max_t for m in matrix_list]).T
    if matrix_preferences is None:
        return pair_to_matrix
    else:
        matrix_preferences = np.asarray(matrix_preferences).reshape((1, len(matrix_list)))
        pair_to_matrix = pair_to_matrix * matrix_preferences
        pair_max = np.max(pair_to_matrix, axis=1, keepdims=True)
        return np.where(pair_to_matrix > 0, np.where(pair_to_matrix == pair_max, True, False), False)


def get_coverings(element_count, covering_matrix, selection):
    n_pairs, n_mat = covering_matrix.shape
    covering_matrix = covering_matrix * np.asarray(selection).reshape((1, n_mat))
    indices = np.argwhere(covering_matrix > 0)
    il, jl = np.triu_indices(element_count, 1)
    return [((il[pair_i], jl[pair_i]), mat_i) for pair_i, mat_i in indices]


def compute_optimal_coverings(matrix_list, matrix_preferences=None, max_t=1.0):
    element_count, _ = matrix_list[0].shape
    covering_matrix = create_covering_matrix(matrix_list, matrix_preferences, max_t)
    constraint_rb = np.max(covering_matrix, axis=1, keepdims=True)
    if np.sum(constraint_rb) > 1:
        solution, _ = solve_set_cover_problem(covering_matrix, constraint_rb)
        return get_coverings(element_count, covering_matrix, solution)
    else:
        return []


def create_fake_correlation_matrix(n):
    a = np.triu(np.random.rand(n,n))
    c = a + a.T
    np.fill_diagonal(c, 1.0)
    return c


if __name__ == "__main__":

    cover_mat = np.array([
        [0, 1, 1, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 1, 0, 0]
    ])

    (n, m) = cover_mat.shape
    constraint_rb = np.ones((n, 1))
    (solution, set_count) = solve_set_cover_problem(cover_mat, constraint_rb)
    print(set_count, np.squeeze(np.asarray(solution)))

    array_list = [
        np.array([[1, 0.5],
                  [0.5, 1]]),

        np.array([[0.5, 1],
                  [0.5, 1]])
    ]

    array_list = [
        np.array([[1, 0.6, 1],
                  [0.6, 1, 1],
                  [1, 1, 1]]),

        np.array([[1, 0.5, 0.7],
                  [0.5, 1, 0.5],
                  [0.7, 0.5, 1]]),

        np.array([[1, 0.5, 0.5],
                  [0.5, 1, 1],
                  [0.5, 1, 1]]),

        np.array([[1, 0.5, 1],
                  [0.5, 1, 1],
                  [1, 1, 1]]),
    ]

    t = 1.0
    cov_mat = create_covering_matrix(
        array_list,
        matrix_preferences=None,
        max_t=t
    )
    print(cov_mat)
    pair_list = compute_optimal_coverings(
        array_list,
        matrix_preferences=None,
        max_t=t
    )
    print(pair_list)
