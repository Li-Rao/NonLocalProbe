# generate_benchmark_parallel.py
"""
Parallel benchmark data generation for comparing exact ED and DTWA methods.
Generates local, power-law non-local, and truncated non-local QFI datasets.
"""
import os
import json
import time
import numpy as np
from joblib import Parallel, delayed

from common import (
    simulate_local_qfi,
    simulate_qfi_k_2body,
    simulate_qfi_truncated_2body,
    simulate_qfi_k_2body_exact_smallN,
    simulate_qfi_truncated_2body_exact_smallN,
)


# =============================
# Small-N Benchmark Task
# =============================
def benchmark_task(alpha, N, eta, beta, t_points, nphi_exact, nphi_dtwa, n_traj, seed):
    """
    Single benchmark task computing QFI using both exact (ED) and DTWA methods.
    
    Computes:
    - Local exact QFI
    - Power-law probe exact ED QFI
    - Truncated probe exact ED QFI
    - Power-law probe DTWA QFI
    - Truncated probe DTWA QFI
    
    Args:
        alpha: Power-law exponent
        N: System size (small N for exact ED)
        eta: Power-law probe exponent
        beta: Truncation exponent
        t_points: Time evolution points
        nphi_exact: Number of rotation angles for exact ED
        nphi_dtwa: Number of rotation angles for DTWA
        n_traj: Number of DTWA trajectories
        seed: Random seed for DTWA
        
    Returns:
        dict: Benchmark results for this parameter set
    """
    # ---------- Exact Local QFI ----------
    qfi_local = simulate_local_qfi(
        N=N,
        alpha=alpha,
        t_points=t_points,
        nphi=nphi_exact,
    )

    # ---------- Exact Power-Law Probe QFI (ED) ----------
    qfi_exact_power = simulate_qfi_k_2body_exact_smallN(
        N=N,
        alpha=alpha,
        eta=eta,
        t_points=t_points,
        nphi=nphi_exact,
    )

    # ---------- Exact Truncated Probe QFI (ED) ----------
    qfi_exact_trunc, R0 = simulate_qfi_truncated_2body_exact_smallN(
        N=N,
        alpha=alpha,
        beta=beta,
        t_points=t_points,
        nphi=nphi_exact,
        return_R0=True,
    )

    # ---------- DTWA Power-Law Probe QFI ----------
    qfi_dtwa_power = simulate_qfi_k_2body(
        N=N,
        alpha=alpha,
        eta=eta,
        t_points=t_points,
        n_traj=n_traj,
        nphi=nphi_dtwa,
        seed=seed,
    )

    # ---------- DTWA Truncated Probe QFI ----------
    qfi_dtwa_trunc, R0_dtwa = simulate_qfi_truncated_2body(
        N=N,
        alpha=alpha,
        beta=beta,
        t_points=t_points,
        n_traj=n_traj,
        nphi=nphi_dtwa,
        seed=seed,
        return_R0=True,
    )

    # Verify that R0 values are consistent between exact and DTWA methods
    if int(R0) != int(R0_dtwa):
        raise ValueError(
            f"R0 mismatch for N={N}, beta={beta}: exact={R0}, dtwa={R0_dtwa}"
        )

    return {
        "N": int(N),
        "alpha": float(alpha),
        "eta": float(eta),
        "beta": float(beta),
        "R0": int(R0),
        "qfi_local": qfi_local.tolist(),
        "qfi_exact_powerlaw": qfi_exact_power.tolist(),
        "qfi_dtwa_powerlaw": qfi_dtwa_power.tolist(),
        "qfi_exact_truncated": qfi_exact_trunc.tolist(),
        "qfi_dtwa_truncated": qfi_dtwa_trunc.tolist(),
    }


# =============================
# Main Benchmark Generator
# =============================
def generate_benchmark_parallel(
    alpha_values,
    N_values,
    eta_values,
    beta_values,
    t_points,
    nphi_exact=91,
    nphi_dtwa=201,
    n_traj=2000,
    seed=12345,
    n_jobs=8,
    output_dir="data",
    output_name="benchmark",
):
    """
    Generate comprehensive benchmark dataset comparing exact and DTWA methods.
    
    Args:
        alpha_values: List of power-law exponent values
        N_values: List of system sizes
        eta_values: List of power-law probe exponent values
        beta_values: List of truncation exponent values
        t_points: Array of time evolution points
        nphi_exact: Number of rotation angles for exact ED
        nphi_dtwa: Number of rotation angles for DTWA
        n_traj: Number of DTWA trajectories
        seed: Random seed for reproducibility
        n_jobs: Number of parallel jobs
        output_dir: Output directory for results
        output_name: Name prefix for output files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate all benchmark task combinations
    tasks = [
        (alpha, N, eta, beta, t_points, nphi_exact, nphi_dtwa, n_traj, seed)
        for alpha in alpha_values
        for N in N_values
        for eta in eta_values
        for beta in beta_values
    ]

    print("=== Generating benchmark data (parallel) ===")
    print(f"Total tasks: {len(tasks)}")

    # Execute all benchmark tasks in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(benchmark_task)(*task) for task in tasks
    )

    # Prepare comprehensive results dictionary
    payload = {
        "dataset_name": output_name,
        "case": "benchmark_dual_nonlocal_plus_local",
        "alpha_values": list(map(float, alpha_values)),
        "N_values": list(map(int, N_values)),
        "eta_values": list(map(float, eta_values)),
        "beta_values": list(map(float, beta_values)),
        "t_points": t_points.tolist(),
        "meta": {
            "local_method": "exact_2301_style",
            "powerlaw_exact_method": "ed_smallN",
            "truncated_exact_method": "ed_smallN",
            "powerlaw_dtwa_method": "dtwa_proxy",
            "truncated_dtwa_method": "dtwa_proxy",
            "nphi_exact": int(nphi_exact),
            "nphi_dtwa": int(nphi_dtwa),
            "n_traj": int(n_traj),
            "seed": int(seed),
            "n_jobs": int(n_jobs),
        },
        "results": results,
    }

    # Save compressed numpy archive
    np.savez_compressed(
        f"{output_dir}/{output_name}.npz",
        alpha_values=np.array(alpha_values, dtype=float),
        N_values=np.array(N_values, dtype=int),
        eta_values=np.array(eta_values, dtype=float),
        beta_values=np.array(beta_values, dtype=float),
        t_points=np.array(t_points, dtype=float),
        results=np.array(results, dtype=object),
    )

    # Save JSON metadata and results
    with open(f"{output_dir}/{output_name}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Saved benchmark dataset to {output_dir}/{output_name}.npz and .json")


# =============================
# Example Usage / Entry Point
# =============================
if __name__ == "__main__":
    alpha_values = [0.5]
    N_values = [12, 14]
    eta_values = [0.2]
    beta_values = [0.8]
    t_points = np.linspace(0.0, 2.0, 81)

    start = time.time()

    generate_benchmark_parallel(
        alpha_values=alpha_values,
        N_values=N_values,
        eta_values=eta_values,
        beta_values=beta_values,
        t_points=t_points,
        nphi_exact=91,
        nphi_dtwa=91,
        n_traj=2000,
        seed=12345,
        n_jobs=12,
        output_dir="data",
        output_name="benchmark",
    )

    end = time.time()
    print(f"Program execution time: {end - start:.4f} seconds")