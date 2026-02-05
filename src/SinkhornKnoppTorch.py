import torch
import triton
import triton.language as tl
import numpy as np
from pprint import pprint

np.set_printoptions(precision=3, suppress=True)

# --- PyTorch Implementation ---
def sinkhorn_knopp_pytorch(cost_matrix, source, target, reg, eps):
    device = cost_matrix.device
    dtype = cost_matrix.dtype
    
    # Largest entries of P correspond to movements with lowest cost
    # Ensure numerical stability by clipping large negative expo nents
    # This is often handled by subtracting the max from cost_matrix first
    # Here we rely on reg being large enough relative to cost_matrix magnitudes
    transport_matrix = torch.exp(-cost_matrix / reg)
    total_mass = transport_matrix.sum()
    transport_matrix /= total_mass

    # Source corresponds to rows, target corresponds to columns
    source = source.unsqueeze(1)  # Shape: (N, 1)
    target = target.unsqueeze(0)  # Shape: (1, M)

    err = float('inf')
    iteration_count = 0
    max_iterations = 1000
    while err > eps and iteration_count < max_iterations:
        # Update u (row scaling factors)
        K_sum = torch.sum(transport_matrix, dim=1, keepdim=True) # Sum over columns (dim=1), shape (N, 1)
        u_new = source / K_sum
        transport_matrix = u_new * transport_matrix # Broadcast multiplication

        # Update v (column scaling factors)
        K_sum = torch.sum(transport_matrix, dim=0, keepdim=True) # Sum over rows (dim=0), shape (1, M)
        v_new = target / K_sum
        transport_matrix = transport_matrix * v_new # Broadcast multiplication

        # Calculate error based on row sums vs source
        err = torch.max(torch.abs(transport_matrix.sum(dim=1, keepdim=True) - source)).item()
        iteration_count += 1
        
    min_cost = torch.sum(transport_matrix * cost_matrix).item()
    return transport_matrix, min_cost

