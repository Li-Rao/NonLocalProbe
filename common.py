from __future__ import annotations

import numpy as np


# ============================================================
# Basic Geometry and Normalization Utilities
# ============================================================

def prep_normalization_1d(alpha: float, N: int) -> float:
    """
    1D normalization convention:
        N_alpha = N^(1-alpha)  for alpha < 1
        N_alpha = 1            for alpha > 1
    
    Note: alpha = 1 is excluded by design and will raise an error.
    
    Args:
        alpha: Power-law exponent
        N: System size
        
    Returns:
        float: Normalization factor for the given alpha and N
    """
    if np.isclose(alpha, 1.0):
        raise ValueError("alpha = 1 is excluded in this setup.")
    if alpha < 1.0:
        return float(N ** (1.0 - alpha))
    return 1.0


def chain_distance_matrix_open(N: int) -> np.ndarray:
    """
    Open-chain distance matrix d_ij = |i-j|.
    """
    idx = np.arange(N, dtype=float)
    return np.abs(idx[:, None] - idx[None, :])


def coupling_matrix_ising_1d(N: int, alpha: float) -> np.ndarray:
    """
    Preparation Hamiltonian coupling matrix:
        J_ij = 4 / (N_alpha * |i-j|^alpha), i != j
        J_ii = 0
    """
    norm = prep_normalization_1d(alpha, N)
    dist = chain_distance_matrix_open(N)
    with np.errstate(divide="ignore"):
        J = np.where(dist > 0, 4.0 / (norm * dist ** alpha), 0.0)
    np.fill_diagonal(J, 0.0)
    return J

def make_trajectory_groups(n_traj: int, n_groups: int, rng: np.random.Generator):
    """
    Randomly partition trajectories into n_groups groups for bootstrap analysis.
    Returns a list of index arrays where empty groups are filtered out.
    
    Args:
        n_traj: Total number of trajectories
        n_groups: Desired number of groups
        rng: NumPy random generator instance
        
    Returns:
        list: List of index arrays, one per group (excluding empty groups)
        
    Raises:
        ValueError: If n_groups < 2 or n_groups > n_traj
    """
    if n_groups <= 1:
        raise ValueError("n_groups must be >= 2.")
    if n_groups > n_traj:
        raise ValueError("n_groups cannot exceed n_traj.")

    perm = rng.permutation(n_traj)
    groups = np.array_split(perm, n_groups)

    # Avoid empty groups
    groups = [g for g in groups if len(g) > 0]
    return groups

# ============================================================
# Local Probe: Exact 2301-style Implementation
# ============================================================
def exact_local_probe_covariances(
    J: np.ndarray,
    t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (Cyy, Cyz) covariance matrices for x-polarized initial state evolved
    under the commuting Ising ZZ Hamiltonian, following the exact formulas
    from Ref. 2301 (their Eqs. S33-S34).
    
    Here J is the preparation coupling matrix with:
        H_e = sum_{i<j} J_ij S_i^z S_j^z
    
    Args:
        J: Preparation coupling matrix
        t: Evolution time
        
    Returns:
        tuple: (Cyy, Cyz) covariance matrices
    """
    N = J.shape[0]
    Cyy = np.zeros((N, N), dtype=float)
    Cyz = np.zeros((N, N), dtype=float)

    # Diagonal entries are not used in the 2301 off-diagonal formula summations,
    # but we explicitly set them to zero for clarity.
    np.fill_diagonal(Cyy, 0.0)
    np.fill_diagonal(Cyz, 0.0)

    for i in range(N):
        for j in range(i + 1, N):
            prod_minus = 1.0
            prod_plus = 1.0
            prod_i = 1.0
            prod_j = 1.0

            for k in range(N):
                if k == i or k == j:
                    continue
                prod_minus *= np.cos(2.0 * (J[i, k] - J[j, k]) * t)
                prod_plus  *= np.cos(2.0 * (J[i, k] + J[j, k]) * t)
                prod_i     *= np.cos(2.0 * J[i, k] * t)
                prod_j     *= np.cos(2.0 * J[j, k] * t)

            c_yy = 0.5 * prod_minus - 0.5 * prod_plus
            c_yz = -np.sin(2.0 * J[i, j] * t) * (prod_i + prod_j)

            Cyy[i, j] = c_yy
            Cyy[j, i] = c_yy

            Cyz[i, j] = c_yz
            Cyz[j, i] = c_yz

    return Cyy, Cyz


def exact_local_probe_qfi_vs_phi(
    Cyy: np.ndarray,
    Cyz: np.ndarray,
    phi_grid: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    Compute the exact local-probe QFI F_Q(phi) using the 2301 formula:
        F_Q(t) = N + max_phi sum_{i != j}
                 [sin^2(phi) Cyy_ij + 0.5 sin(2phi) Cyz_ij]
    
    and return (fq_phi, best_idx).
    
    Args:
        Cyy: Covariance matrix Cyy
        Cyz: Covariance matrix Cyz
        phi_grid: Array of rotation angles to search over
        
    Returns:
        tuple: (fq_phi, best_idx) where fq_phi is the QFI values for each phi,
               and best_idx is the index of the maximum value
    """
    N = Cyy.shape[0]
    fq_phi = np.zeros_like(phi_grid, dtype=float)

    # Sum off-diagonal elements (diagonal entries are zero by construction)
    sum_cyy_offdiag = np.sum(Cyy)
    sum_cyz_offdiag = np.sum(Cyz)

    for m, phi in enumerate(phi_grid):
        s = np.sin(phi)
        sin2phi = np.sin(2.0 * phi)

        fq_phi[m] = N + (s * s) * sum_cyy_offdiag + 0.5 * sin2phi * sum_cyz_offdiag

    best_idx = int(np.argmax(fq_phi))
    return fq_phi, best_idx

def simulate_local_qfi(
    N: int,
    alpha: float,
    t_points: np.ndarray,
    nphi: int = 721,
) -> np.ndarray:
    """
    Exact local-probe QFI curve F_Q(t) for 1D long-range Ising.

    Returns
    -------
    qfi : np.ndarray of shape (len(t_points),)
    """
    J = coupling_matrix_ising_1d(N, alpha)
    phi_grid = np.linspace(0.0, np.pi, nphi, endpoint=False)

    qfi = np.zeros(len(t_points), dtype=float)

    for it, t in enumerate(t_points):
        Cyy, Cyz = exact_local_probe_covariances(J, float(t))
        fq_phi, best_idx = exact_local_probe_qfi_vs_phi(Cyy, Cyz, phi_grid)
        qfi[it] = fq_phi[best_idx]

    return qfi


# ============================================================
# Dynamical Time Warping Approximation (DTWA) for Non-local Probes
# ============================================================

def dtwa_sample_initial_spins(
    n_traj: int,
    N: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    DTWA sampling for initial state |+x>^{⊗N}:
      sx = +1
      sy, sz = ±1 independently
    
    Args:
        n_traj: Number of trajectories
        N: System size
        rng: NumPy random generator instance
        
    Returns:
        tuple: (sx, sy, sz) spin component arrays of shape (n_traj, N)
    """
    sx = np.ones((n_traj, N), dtype=float) * 0.5

    sy = rng.choice(np.array([-0.5, 0.5]), size=(n_traj, N))
    sz = rng.choice(np.array([-0.5, 0.5]), size=(n_traj, N))
    return sx, sy, sz


def evolve_dtwa_ising_exact_times(
    J: np.ndarray,
    t_points: np.ndarray,
    sx0: np.ndarray,
    sy0: np.ndarray,
    sz0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact DTWA time evolution for commuting Ising dynamics.
    
    Since h_i = sum_j J_ij sz_j is constant along each trajectory:
        sx_i(t) = sx_i(0) cos(2 h_i t) - sy_i(0) sin(2 h_i t)
        sy_i(t) = sy_i(0) cos(2 h_i t) + sx_i(0) sin(2 h_i t)
        sz_i(t) = sz_i(0)
    
    Args:
        J: Coupling matrix
        t_points: Array of evolution times
        sx0, sy0, sz0: Initial spin components of shape (n_traj, N)
        
    Returns:
        tuple: (sx_t, sy_t, sz_t) spin components at all times of shape (n_t, n_traj, N)
    """
    fields = sz0 @ J.T   # shape (n_traj, N)
    angles = fields[None, :, :] * t_points[:, None, None]

    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    sx_t = sx0[None, :, :] * cos_a - sy0[None, :, :] * sin_a
    sy_t = sy0[None, :, :] * cos_a + sx0[None, :, :] * sin_a
    sz_t = np.broadcast_to(sz0[None, :, :], sy_t.shape).copy()

    return sx_t, sy_t, sz_t


# ============================================================
# Probe Matrix Functions
# ============================================================

def probe_matrix_powerlaw(N: int, eta: float) -> np.ndarray:
    """
    Power-law probe interaction matrix:
        g_ij = 1 / |i-j|^eta  for i != j
        g_ii = 0
    
    Args:
        N: System size
        eta: Power-law exponent
        
    Returns:
        ndarray: Probe matrix of shape (N, N)
    """
    dist = chain_distance_matrix_open(N)
    with np.errstate(divide="ignore"):
        g = np.where(dist > 0, dist ** (-eta), 0.0)
    np.fill_diagonal(g, 0.0)
    return g


def probe_matrix_truncated(N: int, beta: float) -> tuple[np.ndarray, int]:
    """
    Truncated probe:
        g_ij = 1  if |i-j| <= R0
             = 0  otherwise

    with
        R0 = max(1, floor(N^beta)), beta in [0,1] for 1D.
    """
    if beta < 0 or beta > 1:
        raise ValueError("For 1D, beta must lie in [0,1].")

    R0 = max(1, int(np.floor(N ** beta)))
    dist = chain_distance_matrix_open(N)
    g = ((dist > 0) & (dist <= R0)).astype(float)
    np.fill_diagonal(g, 0.0)
    return g, R0


# ============================================================
# Nonlocal probe QFI proxy
# ============================================================

def pair_observable_from_spins(
    spin_component: np.ndarray,
    g: np.ndarray,
) -> np.ndarray:
    """
    For each trajectory, compute
        H_S = sum_{i<j} g_ij s_i s_j
    """
    return 0.5 * np.einsum("bi,ij,bj->b", spin_component, g, spin_component, optimize=True)


def fq_proxy_curve_dtwa(
    sy_t: np.ndarray,
    sz_t: np.ndarray,
    g: np.ndarray,
    phi_grid: np.ndarray,
) -> np.ndarray:
    """
    Compute
        FQ_proxy(t) = 4 * max_phi Var[H_S(phi)]
    where
        o_i(phi) = sin(phi) sy_i + cos(phi) sz_i
        H_S(phi) = sum_{i<j} g_ij o_i(phi)o_j(phi)
    """
    n_t = sy_t.shape[0]
    fq_t = np.zeros(n_t, dtype=float)

    for it in range(n_t):
        best_var = -np.inf
        sy = sy_t[it]
        sz = sz_t[it]

        for phi in phi_grid:
            s = np.sin(phi)
            c = np.cos(phi)
            obs = s * sy + c * sz

            hs_vals = pair_observable_from_spins(obs, g)
            var_hs = float(np.var(hs_vals, ddof=0))

            if var_hs > best_var:
                best_var = var_hs

        fq_t[it] = 4.0 * best_var

    return fq_t


def simulate_qfi_k_2body_grouped(
    N: int,
    alpha: float,
    eta: float,
    t_points: np.ndarray,
    n_traj: int = 2000,
    nphi: int = 201,
    n_groups: int = 20,
    seed: int = 12345,
):
    """
    Return:
        qfi_full   : shape (n_t,)
        qfi_groups : shape (n_groups, n_t)
    """
    rng = np.random.default_rng(seed)

    J = coupling_matrix_ising_1d(N, alpha)
    g = probe_matrix_powerlaw(N, eta)

    sx0, sy0, sz0 = dtwa_sample_initial_spins(n_traj, N, rng)
    sx_t, sy_t, sz_t = evolve_dtwa_ising_exact_times(J, t_points, sx0, sy0, sz0)

    phi_grid = np.linspace(0.0, np.pi, nphi, endpoint=False)

    # full curve
    qfi_full = fq_proxy_curve_dtwa(sy_t, sz_t, g, phi_grid)

    # grouped curves
    groups = make_trajectory_groups(n_traj, n_groups, rng)
    qfi_groups = []

    for idx in groups:
        qfi_g = fq_proxy_curve_dtwa(
            sy_t[:, idx, :],
            sz_t[:, idx, :],
            g,
            phi_grid
        )
        qfi_groups.append(qfi_g)

    qfi_groups = np.array(qfi_groups, dtype=float)
    return qfi_full, qfi_groups

def simulate_qfi_truncated_2body_grouped(
    N: int,
    alpha: float,
    beta: float,
    t_points: np.ndarray,
    n_traj: int = 2000,
    nphi: int = 201,
    n_groups: int = 20,
    seed: int = 12345,
    return_R0: bool = False,
):
    rng = np.random.default_rng(seed)

    J = coupling_matrix_ising_1d(N, alpha)
    g, R0 = probe_matrix_truncated(N, beta)

    sx0, sy0, sz0 = dtwa_sample_initial_spins(n_traj, N, rng)
    sx_t, sy_t, sz_t = evolve_dtwa_ising_exact_times(J, t_points, sx0, sy0, sz0)

    phi_grid = np.linspace(0.0, np.pi, nphi, endpoint=False)

    qfi_full = fq_proxy_curve_dtwa(sy_t, sz_t, g, phi_grid)

    groups = make_trajectory_groups(n_traj, n_groups, rng)
    qfi_groups = []

    for idx in groups:
        qfi_g = fq_proxy_curve_dtwa(
            sy_t[:, idx, :],
            sz_t[:, idx, :],
            g,
            phi_grid
        )
        qfi_groups.append(qfi_g)

    qfi_groups = np.array(qfi_groups, dtype=float)

    if return_R0:
        return qfi_full, qfi_groups, R0
    return qfi_full, qfi_groups

def simulate_qfi_k_2body(
    N: int,
    alpha: float,
    eta: float,
    t_points: np.ndarray,
    n_traj: int = 2000,
    nphi: int = 201,
    seed: int = 12345,
) -> np.ndarray:
    """
    Power-law nonlocal probe QFI proxy.

    Parameters
    ----------
    N, alpha : preparation parameters
    eta      : probe exponent
    t_points : time grid

    Returns
    -------
    qfi : np.ndarray
        DTWA-based QFI proxy curve
    """
    rng = np.random.default_rng(seed)

    J = coupling_matrix_ising_1d(N, alpha)
    g = probe_matrix_powerlaw(N, eta)

    sx0, sy0, sz0 = dtwa_sample_initial_spins(n_traj, N, rng)
    sx_t, sy_t, sz_t = evolve_dtwa_ising_exact_times(J, t_points, sx0, sy0, sz0)

    phi_grid = np.linspace(0.0, np.pi, nphi, endpoint=False)
    qfi = fq_proxy_curve_dtwa(sy_t, sz_t, g, phi_grid)
    return qfi


def simulate_qfi_truncated_2body(
    N: int,
    alpha: float,
    beta: float,
    t_points: np.ndarray,
    n_traj: int = 2000,
    nphi: int = 201,
    seed: int = 12345,
    return_R0: bool = False,
):
    """
    Truncated nonlocal probe QFI proxy.

    Parameters
    ----------
    beta : truncation exponent with R0 = floor(N^beta)

    Returns
    -------
    qfi : np.ndarray
    optionally R0 if return_R0=True
    """
    rng = np.random.default_rng(seed)

    J = coupling_matrix_ising_1d(N, alpha)
    g, R0 = probe_matrix_truncated(N, beta)

    sx0, sy0, sz0 = dtwa_sample_initial_spins(n_traj, N, rng)
    sx_t, sy_t, sz_t = evolve_dtwa_ising_exact_times(J, t_points, sx0, sy0, sz0)

    phi_grid = np.linspace(0.0, np.pi, nphi, endpoint=False)
    qfi = fq_proxy_curve_dtwa(sy_t, sz_t, g, phi_grid)

    if return_R0:
        return qfi, R0
    return qfi

import scipy.sparse as sp

# ============================================================
# Exact small-N benchmark (ED-style) for nonlocal probes
# ============================================================

def pauli_x() -> sp.csr_matrix:
    return sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))


def pauli_y() -> sp.csr_matrix:
    return sp.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))


def pauli_z() -> sp.csr_matrix:
    return sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))


def spin_x() -> sp.csr_matrix:
    return 0.5 * pauli_x()


def spin_y() -> sp.csr_matrix:
    return 0.5 * pauli_y()


def spin_z() -> sp.csr_matrix:
    return 0.5 * pauli_z()


def kron_all(ops: list[sp.csr_matrix]) -> sp.csr_matrix:
    out = ops[0]
    for op in ops[1:]:
        out = sp.kron(out, op, format="csr")
    return out


def one_site_operator(N: int, site: int, op: sp.csr_matrix) -> sp.csr_matrix:
    """
    Return operator acting as `op` on `site` and identity elsewhere.
    """
    I2 = sp.identity(2, format="csr", dtype=complex)
    ops = [I2 for _ in range(N)]
    ops[site] = op
    return kron_all(ops)


def build_spin_operators(N: int):
    """
    Precompute Sx_i, Sy_i, Sz_i for all sites.
    """
    sx, sy, sz = spin_x(), spin_y(), spin_z()
    Sx_list, Sy_list, Sz_list = [], [], []

    for i in range(N):
        Sx_list.append(one_site_operator(N, i, sx))
        Sy_list.append(one_site_operator(N, i, sy))
        Sz_list.append(one_site_operator(N, i, sz))

    return Sx_list, Sy_list, Sz_list


def build_preparation_hamiltonian_exact(N: int, alpha: float) -> sp.csr_matrix:
    """
    Exact preparation Hamiltonian
        H_e = sum_{i<j} J_ij S_i^z S_j^z
    using the same J_ij as in coupling_matrix_ising_1d.
    """
    J = coupling_matrix_ising_1d(N, alpha)
    _, _, Sz_list = build_spin_operators(N)

    dim = 2 ** N
    H = sp.csr_matrix((dim, dim), dtype=complex)

    for i in range(N):
        for j in range(i + 1, N):
            if J[i, j] != 0.0:
                H = H + J[i, j] * (Sz_list[i] @ Sz_list[j])

    return H


def build_initial_plus_x_state(N: int) -> np.ndarray:
    """
    |+x>^{⊗N} as a dense state vector.
    """
    plus_x = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    psi = plus_x
    for _ in range(N - 1):
        psi = np.kron(psi, plus_x)
    return psi.astype(np.complex128)


def evolve_state_exact_times(
    H: sp.csr_matrix,
    psi0: np.ndarray,
    t_points: np.ndarray,
) -> list[np.ndarray]:
    """
    Exact time evolution by diagonalizing H once:
        psi(t) = exp(-i H t) psi0
    Suitable for small-N benchmark.
    """
    H_dense = H.toarray()
    evals, evecs = np.linalg.eigh(H_dense)
    coeff0 = evecs.conj().T @ psi0

    psi_t_list = []
    for t in t_points:
        phase = np.exp(-1j * evals * float(t))
        psi_t = evecs @ (phase * coeff0)
        psi_t_list.append(psi_t)

    return psi_t_list


def build_local_direction_operator(
    Sy_list: list[sp.csr_matrix],
    Sz_list: list[sp.csr_matrix],
    phi: float,
) -> list[sp.csr_matrix]:
    """
    H_i(phi) = sin(phi) S_i^y + cos(phi) S_i^z
    """
    s = np.sin(phi)
    c = np.cos(phi)
    return [s * Sy_list[i] + c * Sz_list[i] for i in range(len(Sy_list))]


def build_powerlaw_probe_operator_exact(
    N: int,
    eta: float,
    phi: float,
    Sy_list: list[sp.csr_matrix] | None = None,
    Sz_list: list[sp.csr_matrix] | None = None,
) -> sp.csr_matrix:
    """
    Exact power-law two-body probe:
        H_eta(phi) = sum_{i<j} g_ij H_i(phi) H_j(phi)
    """
    if Sy_list is None or Sz_list is None:
        _, Sy_list, Sz_list = build_spin_operators(N)

    Hloc = build_local_direction_operator(Sy_list, Sz_list, phi)
    g = probe_matrix_powerlaw(N, eta)

    dim = 2 ** N
    Hs = sp.csr_matrix((dim, dim), dtype=complex)

    for i in range(N):
        for j in range(i + 1, N):
            if g[i, j] != 0.0:
                Hs = Hs + g[i, j] * (Hloc[i] @ Hloc[j])

    return Hs


def build_truncated_probe_operator_exact(
    N: int,
    beta: float,
    phi: float,
    Sy_list: list[sp.csr_matrix] | None = None,
    Sz_list: list[sp.csr_matrix] | None = None,
    return_R0: bool = False,
):
    """
    Exact truncated two-body probe:
        H_R0(phi) = sum_{i<j, d_ij<=R0} H_i(phi) H_j(phi)
    """
    if Sy_list is None or Sz_list is None:
        _, Sy_list, Sz_list = build_spin_operators(N)

    Hloc = build_local_direction_operator(Sy_list, Sz_list, phi)
    g, R0 = probe_matrix_truncated(N, beta)

    dim = 2 ** N
    Hs = sp.csr_matrix((dim, dim), dtype=complex)

    for i in range(N):
        for j in range(i + 1, N):
            if g[i, j] != 0.0:
                Hs = Hs + g[i, j] * (Hloc[i] @ Hloc[j])

    if return_R0:
        return Hs, R0
    return Hs


def exact_qfi_pure_state_from_generator(
    psi: np.ndarray,
    Hs: sp.csr_matrix,
) -> float:
    """
    Exact pure-state QFI:
        F_Q = 4 ( <H^2> - <H>^2 )
    """
    Hpsi = Hs @ psi
    mean = np.vdot(psi, Hpsi)
    mean2 = np.vdot(Hpsi, Hpsi)
    var = np.real(mean2 - mean * np.conj(mean))
    return float(max(0.0, 4.0 * var))


def simulate_qfi_k_2body_exact_smallN(
    N: int,
    alpha: float,
    eta: float,
    t_points: np.ndarray,
    nphi: int = 91,
) -> np.ndarray:
    """
    Exact small-N benchmark QFI curve for the power-law nonlocal probe.
    """
    Hprep = build_preparation_hamiltonian_exact(N, alpha)
    psi0 = build_initial_plus_x_state(N)
    psi_t_list = evolve_state_exact_times(Hprep, psi0, t_points)

    _, Sy_list, Sz_list = build_spin_operators(N)
    phi_grid = np.linspace(0.0, np.pi, nphi, endpoint=False)

    # Prebuild probe operators for all phi
    Hs_phi = [
        build_powerlaw_probe_operator_exact(
            N=N,
            eta=eta,
            phi=phi,
            Sy_list=Sy_list,
            Sz_list=Sz_list,
        )
        for phi in phi_grid
    ]

    qfi = np.zeros(len(t_points), dtype=float)

    for it, psi in enumerate(psi_t_list):
        best = -np.inf
        for Hs in Hs_phi:
            fq = exact_qfi_pure_state_from_generator(psi, Hs)
            if fq > best:
                best = fq
        qfi[it] = best

    return qfi


def simulate_qfi_truncated_2body_exact_smallN(
    N: int,
    alpha: float,
    beta: float,
    t_points: np.ndarray,
    nphi: int = 91,
    return_R0: bool = False,
):
    """
    Exact small-N benchmark QFI curve for the truncated nonlocal probe.
    """
    Hprep = build_preparation_hamiltonian_exact(N, alpha)
    psi0 = build_initial_plus_x_state(N)
    psi_t_list = evolve_state_exact_times(Hprep, psi0, t_points)

    _, Sy_list, Sz_list = build_spin_operators(N)
    phi_grid = np.linspace(0.0, np.pi, nphi, endpoint=False)

    Hs_phi = []
    R0_ref = None
    for phi in phi_grid:
        Hs, R0 = build_truncated_probe_operator_exact(
            N=N,
            beta=beta,
            phi=phi,
            Sy_list=Sy_list,
            Sz_list=Sz_list,
            return_R0=True,
        )
        Hs_phi.append(Hs)
        R0_ref = R0

    qfi = np.zeros(len(t_points), dtype=float)

    for it, psi in enumerate(psi_t_list):
        best = -np.inf
        for Hs in Hs_phi:
            fq = exact_qfi_pure_state_from_generator(psi, Hs)
            if fq > best:
                best = fq
        qfi[it] = best

    if return_R0:
        return qfi, R0_ref
    return qfi