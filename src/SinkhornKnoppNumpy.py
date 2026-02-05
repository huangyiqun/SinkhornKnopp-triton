import numpy as np

np.set_printoptions(precision=3, suppress=True)


def sinkhorn_knopp_numpy(cost_matrix, source, target, reg, eps):
    transport_matrix = np.exp(-cost_matrix / reg)
    transport_matrix /= transport_matrix.sum()

    source = source.reshape(-1, 1)
    target = target.reshape(1, -1)
    row_ratios = []
    col_ratios = []
    err = 1
    while err > eps:
        row_ratio = source / transport_matrix.sum(axis=1, keepdims=True)
        row_ratios.append(np.max(row_ratio - 1))
        transport_matrix *= row_ratio
        col_ratio = target / transport_matrix.sum(axis=0, keepdims=True)
        col_ratios.append(np.max(col_ratio - 1))
        transport_matrix *= col_ratio

        err = np.max(np.abs(transport_matrix.sum(1, keepdims=True) - source))

    min_cost = np.sum(transport_matrix * cost_matrix)
    return transport_matrix, min_cost, row_ratios, col_ratios
