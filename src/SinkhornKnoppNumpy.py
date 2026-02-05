import torch
import triton
import triton.language as tl
import numpy as np
from pprint import pprint

np.set_printoptions(precision=3, suppress=True)


def sinkhorn_knopp_numpy(cost_matrix, source, target, reg, eps):
     # Largest entries of P correspond to movements with lowest cost
    transport_matrix = np.exp(-cost_matrix / reg)
    transport_matrix /= transport_matrix.sum()

    # Source corresponds to rows, target corresponds to colums
    source = source.reshape(-1, 1)
    target = target.reshape(1, -1)
    row_ratios = []
    col_ratios = []
    err = 1
    while err > eps:
        # Over time this both the row_ratio and col_ratio should approach 
        # vectors of all as our transport matrix approximation improves
        row_ratio = source / transport_matrix.sum(axis=1, keepdims=True)
        row_ratios.append(np.max(row_ratio - 1))
        transport_matrix *= row_ratio
        col_ratio = target / transport_matrix.sum(axis=0, keepdims=True)
        col_ratios.append(np.max(col_ratio - 1))
        transport_matrix *= col_ratio

        # Our error is a measure of how well summing the rows of our 
        # transport matrix approximates the target distribution
        # If we've just normalized our columns to sum to our target distribution,
        # and the sum of our rows is still within tolerance of our source distribution, 
        # we've converged!
        err = np.max(np.abs(transport_matrix.sum(1, keepdims=True) - source))
        
    min_cost = np.sum(transport_matrix * cost_matrix)
    return transport_matrix, min_cost, row_ratios, col_ratios

