import numpy as np
import torch

np.set_printoptions(precision=3, suppress=True)


# --- PyTorch Implementation ---
def sinkhorn_knopp_pytorch(cost_matrix, source, target, reg, eps):
    # device = cost_matrix.device
    # dtype = cost_matrix.dtype

    transport_matrix = torch.exp(-cost_matrix / reg)
    total_mass = transport_matrix.sum()
    transport_matrix /= total_mass

    source = source.unsqueeze(1)
    target = target.unsqueeze(0)

    err = float("inf")
    iteration_count = 0
    max_iterations = 1000
    while err > eps and iteration_count < max_iterations:
        K_sum = torch.sum(transport_matrix, dim=1, keepdim=True)
        u_new = source / K_sum
        transport_matrix = u_new * transport_matrix

        K_sum = torch.sum(transport_matrix, dim=0, keepdim=True)
        v_new = target / K_sum
        transport_matrix = transport_matrix * v_new

        err = torch.max(
            torch.abs(transport_matrix.sum(dim=1, keepdim=True) - source)
        ).item()
        iteration_count += 1

    min_cost = torch.sum(transport_matrix * cost_matrix).item()
    return transport_matrix, min_cost
