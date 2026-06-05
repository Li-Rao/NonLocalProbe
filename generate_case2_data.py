# generate_case2_parallel.py
import os
import json
import numpy as np
from joblib import Parallel, delayed
from common import simulate_local_qfi, simulate_qfi_truncated_2body_grouped
import time

# -----------------------------
# N 选择规则
# -----------------------------
def select_N_values_for_alpha(alpha):
    if alpha < 1.0:
        return [20, 40, 60, 80, 100]
    return [40, 80]

# -----------------------------
# 并行任务
# -----------------------------
def simulate_local_task(alpha, N, t_points, nphi=721):
    qfi = simulate_local_qfi(N=N, alpha=alpha, t_points=t_points, nphi=nphi)
    return {"N": int(N), "alpha": float(alpha), "qfi": qfi.tolist()}

def simulate_truncated_task(
    alpha, beta, N, t_points,
    n_traj=2000, nphi=201, n_groups=20, seed=12345
):
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

# -----------------------------
# 主函数
# -----------------------------
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
    os.makedirs("DataF", exist_ok=True)

    alpha_to_Ns = {float(alpha): select_N_values_for_alpha(float(alpha)) for alpha in alpha_values}

    # -----------------------------
    # Local probe 并行
    # -----------------------------
    print("=== Generating local baseline (parallel) ===")
    local_tasks = [(alpha, N, t_points, nphi_local)
                   for alpha in alpha_values
                   for N in alpha_to_Ns[float(alpha)]]

    local_results = Parallel(n_jobs=n_jobs)(
        delayed(simulate_local_task)(*task) for task in local_tasks
    )

    # -----------------------------
    # Truncated probe 并行
    # -----------------------------
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

    # -----------------------------
    # 保存数据
    # -----------------------------
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

    # 保存 npz
    np.savez_compressed(
        f"DataF/{output_name}.npz",
        alpha_values=np.array(alpha_values, dtype=float),
        beta_values=np.array(beta_values, dtype=float),
        t_points=np.array(t_points, dtype=float),
        local_data=np.array(local_results, dtype=object),
        truncated_data=np.array(truncated_results, dtype=object),
    )

    # 保存 json
    with open(f"DataF/{output_name}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved dataset to DataF/{output_name}.npz and .json")


# -----------------------------
# Example usage
# -----------------------------
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
    print(f"程序运行时间: {end - start:.4f} 秒")