# generate_case2_parallel.py
"""
Parallel data generation for case 2 truncated probe simulations.
Generates both local and truncated non-local quantum Fisher information (QFI) datasets.
"""
import os
import json
import numpy as np
from joblib import Parallel, delayed
from common import simulate_local_qfi, simulate_qfi_truncated_2body_grouped
import time

# =============================
# Parameter Selection Rules
# =============================
def select_N_values_for_alpha(alpha):
    """
    Select system sizes (N) based on the power-law exponent (alpha).
    
    Args:
        alpha: Power-law exponent parameter
        
    Returns:
        list: System sizes to simulate for the given alpha value
    """
    if alpha < 1.0:
        return [20, 40, 60, 80, 100]
    return [40, 80]

# =============================
# Parallel Task Simulation
# =============================
def simulate_local_task(alpha, N, t_points, nphi=721):
    """
    Simulate local quantum Fisher information using exact methods.
    
    Args:
        alpha: Power-law exponent
        N: System size
        t_points: Time evolution points
        nphi: Number of rotation angles
        
    Returns:
        dict: QFI values for local probe scenario
    """
    qfi = simulate_local_qfi(N=N, alpha=alpha, t_points=t_points, nphi=nphi)
    return {"N": int(N), "alpha": float(alpha), "qfi": qfi.tolist()}

def simulate_truncated_task(
    alpha, beta, N, t_points,
    n_traj=2000, nphi=201, n_groups=20, seed=12345
):
    """
    Simulate truncated non-local quantum Fisher information using DTWA.
    
    Args:
        alpha: Power-law exponent
        beta: Truncation exponent (R0 = floor(N^beta))
        N: System size
        t_points: Time evolution points
        n_traj: Number of trajectories for DTWA
        nphi: Number of rotation angles
        n_groups: Number of bootstrap groups
        seed: Random seed for reproducibility
        
    Returns:
        dict: QFI values including full curve, bootstrap groups, and truncation radius
    """
    qfi_full, qfi_groups, R0 = simulate_qfi_truncated_2body_grouped(
        N=N,
        alpha=alpha,
        beta=beta,
        t_points=t_points,
        n_traj=n_traj,
        nphi=nphi,
        n_groups=n_groups,
        seed=seed,
        return_R0=True,
    )
    return {
        "N": int(N),
        "alpha": float(alpha),
        "beta": float(beta),
        "R0": int(R0),
        "qfi": qfi_full.tolist(),
        "qfi_groups": qfi_groups.tolist(),
    }

# =============================
# Main Data Generation Function
# =============================
def generate_case2_data_parallel(
    alpha_values,
    beta_values,
    t_points,
    nphi_local=721,
    nphi_nonlocal=201,
    n_traj=2000,
    n_groups=20,
    seed=12345,
    output_name="case2_truncated_parallel",
    n_jobs=90,
):
    """
    Generate complete case 2 dataset with parallel processing for local and truncated probes.
    
    Args:
        alpha_values: List of power-law exponent values
        beta_values: List of truncation exponent values
        t_points: Array of time evolution points
        nphi_local: Number of rotation angles for local probe
        nphi_nonlocal: Number of rotation angles for truncated probe
        n_traj: Number of DTWA trajectories
        n_groups: Number of bootstrap groups
        seed: Random seed for reproducibility
        output_name: Name prefix for output files
        n_jobs: Number of parallel jobs
    """
    os.makedirs("DataF", exist_ok=True)

    # =============================
    # Determine System Sizes for Each Alpha
    # =============================
    alpha_to_Ns = {float(alpha): select_N_values_for_alpha(float(alpha)) for alpha in alpha_values}

    # =============================
    # Parallel Local Probe Simulation
    # =============================
    print("=== Generating local baseline (parallel) ===")
    local_tasks = [(alpha, N, t_points, nphi_local)
                   for alpha in alpha_values
                   for N in alpha_to_Ns[float(alpha)]]

    local_results = Parallel(n_jobs=n_jobs)(
        delayed(simulate_local_task)(*task) for task in local_tasks
    )

    # =============================
    # Parallel Truncated Probe Simulation
    # =============================
    print("=== Generating truncated nonlocal DTWA (parallel) ===")
    truncated_tasks = [
        (alpha, beta, N, t_points, n_traj, nphi_nonlocal, n_groups, seed)
        for alpha in alpha_values
        for N in alpha_to_Ns[float(alpha)]
        for beta in beta_values
    ]

    truncated_results = Parallel(n_jobs=n_jobs)(
        delayed(simulate_truncated_task)(*task) for task in truncated_tasks
    )

    # =============================
    # Save Generated Data
    # =============================
    # Prepare comprehensive results dictionary
    results = {
        "dataset_name": output_name,
        "case": "truncated_probe",
        "alpha_values": list(map(float, alpha_values)),
        "beta_values": list(map(float, beta_values)),
        "alpha_to_Ns": {str(k): v for k, v in alpha_to_Ns.items()},
        "t_points": t_points.tolist(),
        "meta": {
            "local_method": "exact",
            "nonlocal_method": "dtwa_proxy_parallel",
            "nphi_local": int(nphi_local),
            "nphi_nonlocal": int(nphi_nonlocal),
            "n_traj": int(n_traj),
            "seed": int(seed),
            "N_rule": "alpha<1 -> [20,40,60,80,100]; alpha>1 -> [40,80]",
            "n_jobs": int(n_jobs),
            "n_groups": int(n_groups),
            "group_size_mean": float(n_traj / n_groups),
        },
        "local": local_results,
        "truncated_data": truncated_results,
    }

    # Save compressed numpy archive
    np.savez_compressed(
        f"DataF/{output_name}.npz",
        alpha_values=np.array(alpha_values, dtype=float),
        beta_values=np.array(beta_values, dtype=float),
        t_points=np.array(t_points, dtype=float),
        local_data=np.array(local_results, dtype=object),
        truncated_data=np.array(truncated_results, dtype=object),
    )

    # Save JSON metadata and results
    with open(f"DataF/{output_name}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved dataset to DataF/{output_name}.npz and .json")


# =============================
# Example Usage / Entry Point
# =============================
if __name__ == "__main__":
    alpha_values = [0.5, 1.5, 2.5, 3.5]
    beta_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    t_points = np.linspace(0.0, 2.0, 161)

    start = time.time()
    generate_case2_data_parallel(
        alpha_values=alpha_values,
        beta_values=beta_values,
        t_points=t_points,
        nphi_local=201,
        nphi_nonlocal=201,
        n_traj=4000,
        n_groups=20,
        seed=12345,
        output_name="case2_truncated_scan",
        n_jobs=45,
    )
    end = time.time()
    print(f"Program execution time: {end - start:.4f} seconds")