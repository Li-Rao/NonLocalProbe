import os
import json
import numpy as np

from common import (
    simulate_qfi_k_2body,
    simulate_qfi_truncated_2body,
    simulate_qfi_k_2body_exact_smallN,
    simulate_qfi_truncated_2body_exact_smallN,
)


def generate_benchmark_data(
    alpha=0.5,
    N_values=(8, 10, 12),
    eta_values=(0.5, 1.0, 1.5),
    beta_values=(0.0, 0.5, 1.0),
    t_points=None,
    nphi_exact=91,
    nphi_dtwa=91,
    n_traj=2000,
    seed=12345,
    output_name="benchmark_exact_vs_dtwa",
):
    """
    Generate exact-vs-DTWA benchmark data for both:
      - Case I  : power-law probe
      - Case II : truncated probe

    This is intended for small-N validation only.

    Saved outputs:
      - DataF/<output_name>.npz
      - DataF/<output_name>.json
    """
    if t_points is None:
        t_points = np.linspace(0.0, 0.8, 161)

    os.makedirs("DataF", exist_ok=True)

    results = {
        "dataset_name": output_name,
        "alpha": float(alpha),
        "N_values": list(map(int, N_values)),
        "eta_values": list(map(float, eta_values)),
        "beta_values": list(map(float, beta_values)),
        "t_points": t_points.tolist(),
        "meta": {
            "exact_method": "exact_pure_state_qfi_smallN",
            "dtwa_method": "dtwa_proxy",
            "nphi_exact": int(nphi_exact),
            "nphi_dtwa": int(nphi_dtwa),
            "n_traj": int(n_traj),
            "seed": int(seed),
        },
        "powerlaw_benchmark": [],
        "truncated_benchmark": [],
    }

    # ============================================================
    # Case I: power-law probe
    # ============================================================
    print("\n=== Generating Case I benchmark data (power-law probe) ===")
    for N in N_values:
        for eta in eta_values:
            print(f"[Case I] alpha={alpha}, N={N}, eta={eta} ...")

            qfi_exact = simulate_qfi_k_2body_exact_smallN(
                N=N,
                alpha=alpha,
                eta=eta,
                t_points=t_points,
                nphi=nphi_exact,
            )

            qfi_dtwa = simulate_qfi_k_2body(
                N=N,
                alpha=alpha,
                eta=eta,
                t_points=t_points,
                n_traj=n_traj,
                nphi=nphi_dtwa,
                seed=seed,
            )

            rel_err = np.abs(qfi_dtwa - qfi_exact) / np.maximum(np.abs(qfi_exact), 1e-12)

            # I_Q / (N t)
            qfi_exact_over_nt = np.full_like(qfi_exact, np.nan, dtype=float)
            qfi_dtwa_over_nt = np.full_like(qfi_dtwa, np.nan, dtype=float)

            if len(t_points) > 1:
                qfi_exact_over_nt[1:] = qfi_exact[1:] / (N * t_points[1:])
                qfi_dtwa_over_nt[1:] = qfi_dtwa[1:] / (N * t_points[1:])

            idx_exact = int(np.nanargmax(qfi_exact_over_nt))
            idx_dtwa = int(np.nanargmax(qfi_dtwa_over_nt))

            entry = {
                "N": int(N),
                "alpha": float(alpha),
                "eta": float(eta),
                "qfi_exact": qfi_exact.tolist(),
                "qfi_dtwa": qfi_dtwa.tolist(),
                "rel_err": rel_err.tolist(),
                "qfi_exact_over_nt": qfi_exact_over_nt.tolist(),
                "qfi_dtwa_over_nt": qfi_dtwa_over_nt.tolist(),
                "idx_exact_best_over_nt": idx_exact,
                "idx_dtwa_best_over_nt": idx_dtwa,
                "t_exact_best_over_nt": float(t_points[idx_exact]),
                "t_dtwa_best_over_nt": float(t_points[idx_dtwa]),
                "best_exact_over_nt": float(qfi_exact_over_nt[idx_exact]),
                "best_dtwa_over_nt": float(qfi_dtwa_over_nt[idx_dtwa]),
            }
            results["powerlaw_benchmark"].append(entry)

            print(f"[done] Case I alpha={alpha}, N={N}, eta={eta}")

    # ============================================================
    # Case II: truncated probe
    # ============================================================
    print("\n=== Generating Case II benchmark data (truncated probe) ===")
    for N in N_values:
        for beta in beta_values:
            print(f"[Case II] alpha={alpha}, N={N}, beta={beta} ...")

            qfi_exact, R0_exact = simulate_qfi_truncated_2body_exact_smallN(
                N=N,
                alpha=alpha,
                beta=beta,
                t_points=t_points,
                nphi=nphi_exact,
                return_R0=True,
            )

            qfi_dtwa, R0_dtwa = simulate_qfi_truncated_2body(
                N=N,
                alpha=alpha,
                beta=beta,
                t_points=t_points,
                n_traj=n_traj,
                nphi=nphi_dtwa,
                seed=seed,
                return_R0=True,
            )

            rel_err = np.abs(qfi_dtwa - qfi_exact) / np.maximum(np.abs(qfi_exact), 1e-12)

            # I_Q / (N t)
            qfi_exact_over_nt = np.full_like(qfi_exact, np.nan, dtype=float)
            qfi_dtwa_over_nt = np.full_like(qfi_dtwa, np.nan, dtype=float)

            if len(t_points) > 1:
                qfi_exact_over_nt[1:] = qfi_exact[1:] / (N * t_points[1:])
                qfi_dtwa_over_nt[1:] = qfi_dtwa[1:] / (N * t_points[1:])

            idx_exact = int(np.nanargmax(qfi_exact_over_nt))
            idx_dtwa = int(np.nanargmax(qfi_dtwa_over_nt))

            entry = {
                "N": int(N),
                "alpha": float(alpha),
                "beta": float(beta),
                "R0_exact": int(R0_exact),
                "R0_dtwa": int(R0_dtwa),
                "qfi_exact": qfi_exact.tolist(),
                "qfi_dtwa": qfi_dtwa.tolist(),
                "rel_err": rel_err.tolist(),
                "qfi_exact_over_nt": qfi_exact_over_nt.tolist(),
                "qfi_dtwa_over_nt": qfi_dtwa_over_nt.tolist(),
                "idx_exact_best_over_nt": idx_exact,
                "idx_dtwa_best_over_nt": idx_dtwa,
                "t_exact_best_over_nt": float(t_points[idx_exact]),
                "t_dtwa_best_over_nt": float(t_points[idx_dtwa]),
                "best_exact_over_nt": float(qfi_exact_over_nt[idx_exact]),
                "best_dtwa_over_nt": float(qfi_dtwa_over_nt[idx_dtwa]),
            }
            results["truncated_benchmark"].append(entry)

            print(f"[done] Case II alpha={alpha}, N={N}, beta={beta}, "
                  f"R0_exact={R0_exact}, R0_dtwa={R0_dtwa}")

    # ============================================================
    # Save
    # ============================================================
    np.savez_compressed(
        f"DataF/{output_name}.npz",
        alpha=np.array(alpha, dtype=float),
        N_values=np.array(N_values, dtype=int),
        eta_values=np.array(eta_values, dtype=float),
        beta_values=np.array(beta_values, dtype=float),
        t_points=np.array(t_points, dtype=float),
        powerlaw_benchmark=np.array(results["powerlaw_benchmark"], dtype=object),
        truncated_benchmark=np.array(results["truncated_benchmark"], dtype=object),
    )

    with open(f"DataF/{output_name}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved benchmark dataset to DataF/{output_name}.npz")
    print(f"Saved benchmark metadata to DataF/{output_name}.json")


if __name__ == "__main__":
    t_points = np.linspace(0.0, 0.8, 161)

    generate_benchmark_data(
        alpha=0.5,
        N_values=(8, 10, 12),
        eta_values=(0.5, 1.0, 1.5),
        beta_values=(0.0, 0.5, 1.0),
        t_points=t_points,
        nphi_exact=91,
        nphi_dtwa=91,
        n_traj=2000,
        seed=12345,
        output_name="benchmark_exact_vs_dtwa",
    )