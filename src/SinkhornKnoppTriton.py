import numpy as np
import torch
import triton
import triton.language as tl

np.set_printoptions(precision=3, suppress=True)


# --- Triton Kernel Implementation ---
@triton.jit
def _sinkhorn_kernel_step_u_kernel(
    transport_ptr,
    source_ptr,
    temp_sum_ptr,
    N,
    M,
    stride_tm_n,
    stride_tm_m,
    reg,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)

    row_start = pid_n * BLOCK_SIZE_N
    # row_end = tl.minimum(row_start + BLOCK_SIZE_N, N)

    transport_block_ptr = tl.make_block_ptr(
        base=transport_ptr,
        shape=(N, M),
        strides=(stride_tm_n, stride_tm_m),
        offsets=(row_start, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_M),
        order=(0, 1),
    )
    source_block_ptr = tl.make_block_ptr(
        base=source_ptr,
        shape=(N,),
        strides=(1,),
        offsets=(row_start,),
        block_shape=(BLOCK_SIZE_N,),
        order=(0,),
    )

    transport_block = tl.load(
        transport_block_ptr, boundary_check=(0, 1), padding_option="zero"
    )
    source_block = tl.load(source_block_ptr, boundary_check=(0,))

    row_sums = tl.sum(transport_block, axis=1)

    u_new = source_block / row_sums

    # m_indices = tl.arange(0, BLOCK_SIZE_N)
    u_new_expanded = tl.broadcast_to(u_new[:, None], (BLOCK_SIZE_N, BLOCK_SIZE_M))

    transport_block_updated = transport_block * u_new_expanded

    tl.store(transport_block_ptr, transport_block_updated, boundary_check=(0, 1))


@triton.jit
def _sinkhorn_kernel_step_v_kernel(
    transport_ptr,
    target_ptr,
    N,
    M,
    stride_tm_n,
    stride_tm_m,
    reg,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)

    col_start = pid_m * BLOCK_SIZE_M
    # col_end = tl.minimum(col_start + BLOCK_SIZE_M, M)

    transport_block_ptr = tl.make_block_ptr(
        base=transport_ptr,
        shape=(N, M),
        strides=(stride_tm_n, stride_tm_m),
        offsets=(0, col_start),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_M),
        order=(0, 1),
    )
    target_block_ptr = tl.make_block_ptr(
        base=target_ptr,
        shape=(M,),
        strides=(1,),
        offsets=(col_start,),
        block_shape=(BLOCK_SIZE_M,),
        order=(0,),
    )

    transport_block = tl.load(
        transport_block_ptr, boundary_check=(0, 1), padding_option="zero"
    )
    target_block = tl.load(target_block_ptr, boundary_check=(0,))

    col_sums = tl.sum(transport_block, axis=0)

    v_new = target_block / col_sums

    v_new_expanded = tl.broadcast_to(v_new[None, :], (BLOCK_SIZE_N, BLOCK_SIZE_M))

    transport_block_updated = transport_block * v_new_expanded

    tl.store(transport_block_ptr, transport_block_updated, boundary_check=(0, 1))


def sinkhorn_knopp_triton(cost_matrix, source, target, reg, eps):
    assert cost_matrix.is_cuda, "Input tensors must be on CUDA device"
    device = cost_matrix.device
    dtype = cost_matrix.dtype
    N, M = cost_matrix.shape

    K = torch.exp(-cost_matrix / reg)
    total_mass = K.sum().item()
    transport_matrix = K / total_mass

    source = source.to(device=device, dtype=dtype).unsqueeze(1)
    target = target.to(device=device, dtype=dtype).unsqueeze(0)

    BLOCK_SIZE_N = 32
    BLOCK_SIZE_M = 32

    grid_u = (triton.cdiv(N, BLOCK_SIZE_N), 1, 1)
    grid_v = (triton.cdiv(M, BLOCK_SIZE_M), 1, 1)

    err = float("inf")
    iteration_count = 0
    max_iterations = 1000

    while err > eps and iteration_count < max_iterations:
        _sinkhorn_kernel_step_u_kernel[grid_u](
            transport_matrix,
            source.flatten(),
            None,
            N,
            M,
            transport_matrix.stride(0),
            transport_matrix.stride(1),
            reg,
            BLOCK_SIZE_N,
            BLOCK_SIZE_M,
        )

        row_sums = transport_matrix.sum(dim=1, keepdim=True)
        err = torch.max(torch.abs(row_sums - source)).item()

        _sinkhorn_kernel_step_v_kernel[grid_v](
            transport_matrix,
            target.flatten(),
            N,
            M,
            transport_matrix.stride(0),
            transport_matrix.stride(1),
            reg,
            BLOCK_SIZE_N,
            BLOCK_SIZE_M,
        )

        iteration_count += 1

    min_cost = torch.sum(transport_matrix * cost_matrix).item()

    return transport_matrix, min_cost
