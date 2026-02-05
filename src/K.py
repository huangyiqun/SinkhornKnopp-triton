import torch
import triton
import triton.language as tl
import numpy as np
from pprint import pprint

np.set_printoptions(precision=3, suppress=True)


def sinkhorn_knopp(cost_matrix, source, target, reg, eps):
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


def main_np():
    source = np.array([1, 1, 1, 1, 1])
    target = np.array([1, 1, 1, 1, 1])
    cost_matrix = np.random.random((5, 5))

    transport_matrix, min_cost, row_ratios, col_ratios = sinkhorn_knopp(
        cost_matrix,
        source,
        target,
        reg=0.001, 
        eps=1e-4
    )

    print("Transport matrix", transport_matrix)
    print("Min cost", min_cost)


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


# --- Triton Kernel Implementation ---
@triton.jit
def _sinkhorn_kernel_step_u_kernel(
    transport_ptr,  # Pointer to transport matrix data
    source_ptr,     # Pointer to source vector
    temp_sum_ptr,   # Pointer to temporary storage for row sums
    N, M,           # Dimensions of the transport matrix
    stride_tm_n, stride_tm_m, # Strides for transport matrix (assuming row-major)
    reg,            # Regularization parameter
    BLOCK_SIZE_N: tl.constexpr, # Block size along N dimension
    BLOCK_SIZE_M: tl.constexpr, # Block size along M dimension
):
    pid_n = tl.program_id(axis=0)
    
    # Define block's row range
    row_start = pid_n * BLOCK_SIZE_N
    row_end = tl.minimum(row_start + BLOCK_SIZE_N, N)
    # Note: num_rows_in_block is not used directly in reshape/broadcast_to now

    # Pointers for the current block of rows
    transport_block_ptr = tl.make_block_ptr(
        base=transport_ptr,
        shape=(N, M),
        strides=(stride_tm_n, stride_tm_m),
        offsets=(row_start, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_M),
        order=(0, 1)  # (M, N) -> row-major access
    )
    source_block_ptr = tl.make_block_ptr(
        base=source_ptr,
        shape=(N,),
        strides=(1,), # Assuming source is contiguous
        offsets=(row_start,),
        block_shape=(BLOCK_SIZE_N,),
        order=(0,)
    )
    # temp_sum_block_ptr might not be necessary if error is checked externally
    # temp_sum_block_ptr = tl.make_block_ptr(
    #     base=temp_sum_ptr,
    #     shape=(N,),
    #     strides=(1,),
    #     offsets=(row_start,),
    #     block_shape=(BLOCK_SIZE_N,),
    #     order=(0,)
    # )

    # Load data for the block
    transport_block = tl.load(transport_block_ptr, boundary_check=(0, 1), padding_option="zero")
    source_block = tl.load(source_block_ptr, boundary_check=(0,))
    
    # Compute row sums for the current block of rows
    # The mask for padding ensures padded elements (which are 0) don't affect the sum
    row_sums = tl.sum(transport_block, axis=1) # Sums along M dimension (axis=1), result shape (BLOCK_SIZE_N,)

    # Compute new scaling factors u = source / row_sum
    u_new = source_block / row_sums

    # Update the transport matrix block: transport[i, j] *= u_new[i]
    # We need to broadcast u_new across the M dimension.
    # u_new has shape (BLOCK_SIZE_N,), we want to multiply each row i by u_new[i].
    # Create a shape (BLOCK_SIZE_N, 1) tensor for broadcasting.
    # We can achieve this by reshaping u_new implicitly during the broadcast operation.
    # Create a tensor of shape (BLOCK_SIZE_N,) representing indices into u_new's rows
    m_indices = tl.arange(0, BLOCK_SIZE_N)
    # Expand u_new to (BLOCK_SIZE_N, BLOCK_SIZE_M) by broadcasting
    # The shape of u_new_expanded will be (BLOCK_SIZE_N, BLOCK_SIZE_M)
    u_new_expanded = tl.broadcast_to(u_new[:, None], (BLOCK_SIZE_N, BLOCK_SIZE_M))
    
    # Perform the multiplication
    transport_block_updated = transport_block * u_new_expanded
    
    # Store the updated values back
    tl.store(transport_block_ptr, transport_block_updated, boundary_check=(0, 1))


@triton.jit
def _sinkhorn_kernel_step_v_kernel(
    transport_ptr,  # Pointer to transport matrix data
    target_ptr,     # Pointer to target vector
    N, M,           # Dimensions of the transport matrix
    stride_tm_n, stride_tm_m, # Strides for transport matrix
    reg,            # Regularization parameter
    BLOCK_SIZE_N: tl.constexpr, # Block size along N dimension
    BLOCK_SIZE_M: tl.constexpr, # Block size along M dimension
):
    pid_m = tl.program_id(axis=0)

    # Define block's column range
    col_start = pid_m * BLOCK_SIZE_M
    col_end = tl.minimum(col_start + BLOCK_SIZE_M, M)
    # Note: num_cols_in_block is not used directly in reshape/broadcast_to now

    # Pointers for the current block of columns
    transport_block_ptr = tl.make_block_ptr(
        base=transport_ptr,
        shape=(N, M),
        strides=(stride_tm_n, stride_tm_m),
        offsets=(0, col_start),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_M),
        order=(0, 1)  # (M, N) -> row-major access
    )
    target_block_ptr = tl.make_block_ptr(
        base=target_ptr,
        shape=(M,),
        strides=(1,), # Assuming target is contiguous
        offsets=(col_start,),
        block_shape=(BLOCK_SIZE_M,),
        order=(0,)
    )

    # Load data for the block
    transport_block = tl.load(transport_block_ptr, boundary_check=(0, 1), padding_option="zero")
    target_block = tl.load(target_block_ptr, boundary_check=(0,))

    # Compute column sums for the current block of columns
    # The mask for padding ensures padded elements (which are 0) don't affect the sum
    col_sums = tl.sum(transport_block, axis=0) # Sums along N dimension (axis=0), result shape (BLOCK_SIZE_M,)

    # Compute new scaling factors v = target / col_sum
    v_new = target_block / col_sums

    # Update the transport matrix block: transport[i, j] *= v_new[j]
    # We need to broadcast v_new across the N dimension.
    # v_new has shape (BLOCK_SIZE_M,), we want to multiply each column j by v_new[j].
    # Create a shape (BLOCK_SIZE_N, BLOCK_SIZE_M) tensor for broadcasting.
    v_new_expanded = tl.broadcast_to(v_new[None, :], (BLOCK_SIZE_N, BLOCK_SIZE_M))
    
    # Perform the multiplication
    transport_block_updated = transport_block * v_new_expanded
    
    # Store the updated values back
    tl.store(transport_block_ptr, transport_block_updated, boundary_check=(0, 1))


def sinkhorn_knopp_triton(cost_matrix, source, target, reg, eps):
    assert cost_matrix.is_cuda, "Input tensors must be on CUDA device"
    device = cost_matrix.device
    dtype = cost_matrix.dtype
    N, M = cost_matrix.shape

    # Initialize transport matrix on GPU
    # Apply regularization and initial normalization
    K = torch.exp(-cost_matrix / reg)
    total_mass = K.sum().item()
    transport_matrix = K / total_mass

    # Ensure source and target are on the same device and correct shape
    source = source.to(device=device, dtype=dtype).unsqueeze(1)  # Shape: (N, 1)
    target = target.to(device=device, dtype=dtype).unsqueeze(0)  # Shape: (1, M)

    # Define block sizes for Triton kernels
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_M = 32

    # Calculate grid dimensions
    grid_u = (triton.cdiv(N, BLOCK_SIZE_N), 1, 1)
    grid_v = (triton.cdiv(M, BLOCK_SIZE_M), 1, 1)

    # Temporary buffer for row/column sums during iterations (not strictly needed for algorithm, but for potential error check)
    # temp_row_sums = torch.empty((N,), dtype=dtype, device=device)

    err = float('inf')
    iteration_count = 0
    max_iterations = 1000

    while err > eps and iteration_count < max_iterations:
        # Step 1: Update u (rows) using Triton kernel
        _sinkhorn_kernel_step_u_kernel[grid_u](
            transport_matrix, source.flatten(), None, # Pass None for temp_sum_ptr if not used
            N, M, transport_matrix.stride(0), transport_matrix.stride(1), reg,
            BLOCK_SIZE_N, BLOCK_SIZE_M
        )
        
        # Calculate error based on row sums vs source (using PyTorch for simplicity after Triton update)
        row_sums = transport_matrix.sum(dim=1, keepdim=True)
        err = torch.max(torch.abs(row_sums - source)).item()

        # Step 2: Update v (columns) using Triton kernel
        _sinkhorn_kernel_step_v_kernel[grid_v](
            transport_matrix, target.flatten(),
            N, M, transport_matrix.stride(0), transport_matrix.stride(1), reg,
            BLOCK_SIZE_N, BLOCK_SIZE_M
        )
        
        iteration_count += 1

    # Final cost calculation using PyTorch (or another Triton kernel if desired)
    min_cost = torch.sum(transport_matrix * cost_matrix).item()
    
    return transport_matrix, min_cost


def main():
    # Use CUDA device for Triton
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("CUDA is required to run the Triton kernel.")
        return

    print(f"Running on device: {device}")
    
    # Convert numpy arrays to PyTorch tensors and move to CUDA
    source_np = np.array([1, 1, 1, 1, 1], dtype=np.float32)
    target_np = np.array([1, 1, 1, 1, 1], dtype=np.float32)
    # Adjusted cost matrix scale and regularization for better numerical stability
    cost_matrix_np = np.random.random((5, 5)).astype(np.float32) * 0.5 # Reduce magnitude
    reg_val = 1e-2 # Increased regularization
    eps_val = 1e-4

    source_torch = torch.from_numpy(source_np).to(device)
    target_torch = torch.from_numpy(target_np).to(device)
    cost_matrix_torch = torch.from_numpy(cost_matrix_np).to(device)

    source_triton = source_torch.clone()
    target_triton = target_torch.clone()
    cost_matrix_triton = cost_matrix_torch.clone()

    # Run PyTorch version
    print("--- Running PyTorch Version ---")
    transport_pytorch, min_cost_pytorch = sinkhorn_knopp_pytorch(
        cost_matrix_torch, source_torch, target_torch, reg=reg_val, eps=eps_val
    )
    print("PyTorch Transport matrix:\n", transport_pytorch.cpu().numpy())
    print("PyTorch Min cost:", min_cost_pytorch)
    print("Output:", source_torch @ transport_pytorch)

    # Run Triton version
    print("\n--- Running Triton Version ---")
    transport_triton, min_cost_triton = sinkhorn_knopp_triton(
        cost_matrix_triton, source_triton, target_triton, reg=reg_val, eps=eps_val
    )
    print("Triton Transport matrix:\n", transport_triton.cpu().numpy())
    print("Triton Min cost:", min_cost_triton)
    print("Output:", source_triton @ transport_triton)

    # Compare results
    print("\n--- Comparison ---")
    print("Transport matrices close?\n", torch.allclose(transport_pytorch, transport_triton, atol=1e-2))
    print("Costs close?\n", abs(min_cost_pytorch - min_cost_triton) < 1e-2)


if __name__ == "__main__":
    main()
