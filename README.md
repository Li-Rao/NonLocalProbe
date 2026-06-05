# NonLocalProbe

QFI calculations for 1D long-range Ising systems. Compares local probes with non-local probes using exact ED and DTWA.

## Quick Start

```bash
# Generate data
python generate_case1_data.py         # case 1: power-law probe
python generate_case2_data.py         # case 2: truncated probe
python generate_benchmarkparallel_data.py  # benchmark: ED vs DTWA

# Visualize (Plot.ipynb expects data in DataF/)
jupyter notebook Plot.ipynb
```

## Core Files

**Data generation:**
- `generate_case1_data.py` → `DataF/case1_powerlaw_scan.npz`
- `generate_case2_data.py` → `DataF/case2_truncated_scan.npz`
- `generate_benchmarkparallel_data.py` → `DataF/benchmark.npz`

**Common utilities:**
- `common.py` - Core QFI functions (local exact, DTWA, ED)

**Analysis:**
- `Plot.ipynb` - Load and visualize case 1/2/benchmark

## Methods

**Local**: Exact covariances for x-polarized state

**DTWA**: Trajectory sampling 
- Random ±1/2 spins from |+x⟩
- Exact evolution for commuting Ising
- Var[pair observable] ≈ QFI proxy
- Scales to large N

**ED**: Full diagonalization (N ≤ 14)

## Output Data

Each case saves two formats to `DataF/`:

```python
# NPZ format (fast)
data = np.load('case1_powerlaw_scan.npz', allow_pickle=True)
# Keys: alpha_values, eta_values, t_points, local_data, nonlocal_data

# Each result entry:
{
    "N": int,
    "alpha": float,
    "eta": float,              # case 1 only
    "beta": float,             # case 2 only
    "R0": int,                 # case 2 only
    "qfi": list[float],        # main QFI curve
    "qfi_groups": list[]       # 20 bootstrap groups for errors
}
```

Benchmark results similar but with both exact and DTWA curves.

## Parameters

| Parameter | Range | Notes |
|-----------|-------|-------|
| alpha | 0.5-3.5 | Power-law exponent |
| eta | 0.2-2.0 | Power-law probe (case 1) |
| beta | 0-1 | Truncation (case 2) |
| N | 20-100 | Auto per alpha |
| n_traj | 2000-4000 | DTWA trajectories |
| n_groups | 20 | Bootstrap groups |

## Notes

- Error bars computed from 20 trajectory groups
- All simulations use fixed seeds (reproducible)
- Parallelized with joblib (default 45 jobs)