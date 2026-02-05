import numpy as np
import torch
from SinkhornKnopp_triton import (  # sinkhorn_knopp_numpy,
    sinkhorn_knopp_pytorch,
    sinkhorn_knopp_triton,
)

np.set_printoptions(precision=3, suppress=True)


def main():
    # Use CUDA device for Triton
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA is required to run the Triton kernel.")
        return

    print(f"Running on device: {device}")

    # Convert numpy arrays to PyTorch tensors and move to CUDA
    source_np = np.array([1, 1, 1, 1, 1], dtype=np.float32)
    target_np = np.array([1, 1, 1, 1, 1], dtype=np.float32)
    # Adjusted cost matrix scale and regularization for better numerical stability
    cost_matrix_np = (
        np.random.random((5, 5)).astype(np.float32) * 0.5
    )  # Reduce magnitude
    reg_val = 1e-2  # Increased regularization
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
    # print("Output:", source_torch @ transport_pytorch)
    print(
        "Source_matrices * Transport_matrices_torch close?\n",
        torch.allclose(source_torch @ transport_pytorch, target_torch, atol=1e-2),
    )

    # Run Triton version
    print("\n--- Running Triton Version ---")
    transport_triton, min_cost_triton = sinkhorn_knopp_triton(
        cost_matrix_triton, source_triton, target_triton, reg=reg_val, eps=eps_val
    )
    print("Triton Transport matrix:\n", transport_triton.cpu().numpy())
    print("Triton Min cost:", min_cost_triton)
    # print("Output:", source_triton @ transport_triton)
    print(
        "Source_matrices * Transport_matrices_triton close?\n",
        torch.allclose(source_triton @ transport_triton, target_torch, atol=1e-2),
    )


if __name__ == "__main__":
    main()
